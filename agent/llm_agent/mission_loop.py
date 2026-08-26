import os
from math import pi

import geopandas as gpd
from sarenv.analytics.evaluator import ComparativeEvaluator
from sarenv.analytics.metrics import PathEvaluator
from sarenv.core.lost_person import LostPersonLocationGenerator
from shapely.geometry import Point

from agent.llm_agent import greedy_seed  # noqa: F401 - effetto collaterale: seed fisso per generate_greedy_path
from agent.battery_health.health_check import DEFAULT_SEARCHING_CURRENTS_BY_ALGORITHM, MissionLoadProfile, health_check
from agent.llm_agent.battery_setup import initial_battery_state, make_battery_model
from agent.llm_agent.decisions import decide_algorithm, decide_safety, print_tick_status
from agent.llm_agent.llm_client import summarize_feature_probabilities
from agent.llm_agent.reporting import save_trajectory_plot, save_victims_plot
from agent.llm_agent.signal_link import update_signal_status
from agent.llm_agent.trajectory import advance_trajectory, build_final_trajectory
from agent.planning_bridge.path_builder import altitude_for_radius
from agent.planning_bridge.plan import Plan

# SAREnv non fa parte di questo repo (dipendenza esterna, va clonata a parte -
# vedi README) - il percorso del dataset dipende da dove la cloni, quindi è
# sovrascrivibile con la variabile d'ambiente SARENV_DATASET_DIR invece di
# essere fisso nel codice. Default relativo (non un path assoluto personale):
# assume SAREnv clonata come sibling di questo repo, com'è documentato nel
# README.
DATA_DIR = os.environ.get("SARENV_DATASET_DIR", "../SAREnv/sarenv_dataset/19")

STARTUP_BANNER = r"""
                  /\
         /**\
        /****\   /\
       /      \ /**\
      /  /\    /    \        /\    /\  /\      /\            /\/\/\  /\
     /  /  \  /      \      /  \/\/  \/  \  /\/  \/\  /\  /\/ / /  \/  \
    /  /    \/ /\     \    /    \ \  /    \/ /   /  \/  \/  \  /    \   \
   /  /      \/  \/\   \  /      \    /   /    \
__/__/_______/___/__\___\__________________________________________________

  :####:     :##:    ######:     :##:
 :######      ##     #######      ##
 ##:  :#     ####    ##   :##    ####
 ##          ####    ##    ##    ####
 ###:       :#  #:   ##   :##   :#  #:
 :#####:     #::#    #######:    #::#
  .#####:   ##  ##   ######     ##  ##
     :###   ######   ##   ##.   ######
       ##  .######.  ##   ##   .######.
 #:.  :##  :##  ##:  ##   :##  :##  ##:
 #######:  ###  ###  ##    ##: ###  ###
 .#####:   ##:  :##  ##    ### ##:  :##
"""
SEARCH_SPEED_MPS = 8.0
SAFETY_MARGIN_M = 500
TICK_SECONDS = 300
# Abbastanza alto da non tagliare la missione mentre copertura/RUL sono
# ancora sani (verificato empiricamente).
MAX_MISSION_SECONDS = 20000


def load_environment(data_dir=DATA_DIR):
    evaluator = ComparativeEvaluator(
        dataset_directory=data_dir,
        evaluation_sizes=["small"],
        num_drones=1,
        num_lost_persons=100,
        budget=62000,
    )
    _, env_data = next(iter(evaluator.environments.items()))
    item = env_data["item"]
    center_proj = (
        gpd.GeoDataFrame(geometry=[Point(item.center_point)], crs="EPSG:4326")
        .to_crs(env_data["crs"])
        .geometry.iloc[0]
    )
    victims_gdf = env_data["victims"]
    meters_per_bin = evaluator.loader._meter_per_bin
    return item, center_proj, victims_gdf, meters_per_bin


def run_mission(data_dir=DATA_DIR, battery_start_soc=1.0, output_prefix="mission", signal_loss=False):
    """Args:
        data_dir: cartella del dataset SAREnv (es. .../sarenv_dataset/19).
        battery_start_soc: stato di carica iniziale, 0-1 (default 1.0 = piena).
        output_prefix: prefisso dei due PDF salvati a fine missione.
        signal_loss: se False (default), signal_connected resta puramente
            narrativo/informativo come prima - nessun effetto sulle decisioni.
            Se True, quando il segnale è connesso l'LLM non viene interpellato
            e il piano resta congelato (si presume che un eventuale operatore
            a terra sia raggiungibile, anche se non è simulato); quando il
            segnale si perde l'agente "si attiva" e riprende a decidere in
            autonomia come fa sempre ora. Il guardrail deterministico
            (rtb_required in health_check) resta SEMPRE attivo in entrambi i
            casi, indipendentemente dal segnale - è safety-critical, non può
            dipendere da un link radio che può cadere nel momento peggiore.

    Returns:
        dict con le metriche finali della missione.
    """
    print(STARTUP_BANNER)
    item, center_proj, victims_gdf, meters_per_bin = load_environment(data_dir)
    ipp_point = center_proj

    type_probabilities = LostPersonLocationGenerator(item).type_probabilities
    environment_summary = summarize_feature_probabilities(type_probabilities)
    print(f"Environment summary: {environment_summary}")

    # overlap_ratio=0.2: prassi SAR reale (15-30%), compensa i dispersi
    # mancati "per pochissimo" tra due linee di scansione consecutive.
    # altitude_m: proporzionale al raggio della ROI (vedi altitude_for_radius),
    # invariata rispetto al default sugli ambienti dove non serve.
    plan = Plan(
        algorithm="spiral",
        budget_m=62000,
        overlap_ratio=0.2,
        status="SEARCHING",
        altitude_m=altitude_for_radius(item.radius_km),
    )
    model = make_battery_model()
    load_profile = MissionLoadProfile(
        plan=plan,
        searching_currents_by_algorithm=DEFAULT_SEARCHING_CURRENTS_BY_ALGORITHM,
        returning_current=1.7,
    )
    battery_state = initial_battery_state(model, battery_start_soc)
    t = 0

    flown_segments = []
    distance_within_algorithm = 0.0
    last_algorithm = plan.algorithm
    signal_connected = True

    print(f"--- Missione avviata: {plan} ---")

    while plan.status != "RETURN_TO_BASE" and t < MAX_MISSION_SECONDS:
        t += TICK_SECONDS
        signal_connected = update_signal_status(signal_connected, t)

        algorithm_switched = plan.algorithm != last_algorithm
        last_algorithm = plan.algorithm
        path, distance_within_algorithm = advance_trajectory(
            plan, item, center_proj, flown_segments, distance_within_algorithm,
            algorithm_switched, TICK_SECONDS, SEARCH_SPEED_MPS,
        )

        rul_json = health_check(
            plan=plan,
            path=path,
            ipp_point=ipp_point,
            battery_state=battery_state,
            battery_model=model,
            load_profile=load_profile,
            t=t,
            distance_traveled_m=distance_within_algorithm,
            search_speed_mps=SEARCH_SPEED_MPS,
            safety_margin_m=SAFETY_MARGIN_M,
        )
        rul_json["derived"]["signal_connected"] = signal_connected  # informativo

        warmup = model.simulate_to_threshold(
            load_profile, x=battery_state, dt=1, save_freq=TICK_SECONDS, horizon=TICK_SECONDS
        )
        battery_state = warmup.states[-1]

        print_tick_status(t, rul_json, plan, signal_connected)

        if plan.status == "RETURN_TO_BASE":
            print("Guardrail deterministico ha forzato il rientro.")
            break

        agent_active = not (signal_loss and signal_connected)
        if agent_active:
            safety_call = decide_safety(plan, rul_json)
            if safety_call.get("tool") == "no_action" and plan.status != "RETURN_TO_BASE":
                decide_algorithm(
                    plan, item, center_proj, victims_gdf, meters_per_bin, flown_segments, environment_summary, rul_json
                )
        else:
            print(f"[t={t}s] Segnale connesso - agente in stand-by, piano congelato (guardrail comunque attivo).")

    print(f"\n--- Missione conclusa: {plan} ---")

    trajectory_segments = build_final_trajectory(flown_segments, plan, ipp_point)

    evaluator = PathEvaluator(item.heatmap, item.bounds, victims_gdf, plan.fov_deg, plan.altitude_m, meters_per_bin)
    final_metrics = evaluator.calculate_all_metrics(flown_segments, discount_factor=0.999)
    victim_metrics = final_metrics["victim_detection_metrics"]
    total_roi_area_km2 = pi * (item.radius_km ** 2)
    final_coverage_fraction = final_metrics["area_covered"] / total_roi_area_km2 if total_roi_area_km2 > 0 else 0
    print(
        f"Dispersi trovati: {victim_metrics['percentage_found']:.1f}% "
        f"({len(victim_metrics['found_victim_indices'])} su {len(victims_gdf)})"
    )

    save_trajectory_plot(item, trajectory_segments, f"{output_prefix}_trajectory.pdf")
    save_victims_plot(item, victims_gdf, victim_metrics, trajectory_segments, ipp_point, f"{output_prefix}_victims_map.pdf")

    return {
        "data_dir": data_dir,
        "battery_start_soc": battery_start_soc,
        "percentage_found": victim_metrics["percentage_found"],
        "victims_found": len(victim_metrics["found_victim_indices"]),
        "victims_total": len(victims_gdf),
        "coverage_fraction": final_coverage_fraction,
        "final_status": plan.status,
        "mission_duration_s": t,
        "final_algorithm": plan.algorithm,
    }


if __name__ == "__main__":
    run_mission()
