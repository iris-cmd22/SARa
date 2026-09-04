import os
import random
from math import pi

import geopandas as gpd
import numpy as np
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
# ancora sani (verificato empiricamente). Sovrascrivibile via env var per
# esperimenti che devono restare corti apposta (es. sweep di resilienza) -
# default invariato, non tocca il guardrail stesso (n_runs/orizzonte Monte
# Carlo restano quelli di sempre, converge solo più in fretta con RUL basso).
MAX_MISSION_SECONDS = int(os.environ.get("MAX_MISSION_SECONDS", 20000))
# Se la ricerca ha di fatto già coperto l'area/i dispersi (soglie alte
# apposta, per non tagliare corto durante la fase normale di ricerca),
# l'agente continuerebbe comunque a rivalutare la stessa situazione senza
# avanzare - osservato empiricamente (copertura/efficacia identiche per
# tick consecutivi). Forziamo il rientro invece di aspettare inutilmente
# MAX_MISSION_SECONDS.
SEARCH_COMPLETE_COVERAGE = 1.0
SEARCH_COMPLETE_LIKELIHOOD = 0.99


def _check_search_complete(plan, coverage_fraction, likelihood_fraction):
    if coverage_fraction >= SEARCH_COMPLETE_COVERAGE and likelihood_fraction >= SEARCH_COMPLETE_LIKELIHOOD:
        print(
            f"Ricerca sostanzialmente completa (copertura {coverage_fraction:.1%}, "
            f"probabilità {likelihood_fraction:.1%}): rientro forzato."
        )
        plan.status = "RETURN_TO_BASE"


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


def run_mission(data_dir=DATA_DIR, battery_start_soc=1.0, output_prefix="mission", signal_loss=False, fixed_algorithm=None, seed=None):
    """Args:
        data_dir: cartella del dataset SAREnv (es. .../sarenv_dataset/19).
        battery_start_soc: stato di carica iniziale, 0-1 (default 1.0 = piena).
        output_prefix: prefisso dei due PDF salvati a fine missione.
        seed: se impostato, fissa random.seed()/np.random.seed() prima di
            generare l'ambiente - controlla sia il posizionamento dei dispersi
            (LostPersonLocationGenerator in SAREnv, usa random.uniform/choices
            e .sample() di pandas, quindi random+np.random legacy) sia il
            rumore di processo della batteria (ProgPy, np.random.normal/
            uniform in noise_functions.py - anch'esso stato globale legacy,
            non default_rng()). Non tocca il path di greedy, già fissato a
            parte con un seed indipendente in greedy_seed.py (default_rng,
            API diversa). Default None: nessun seeding, comportamento
            invariato rispetto a prima - necessario per confrontare più
            modelli/algoritmi sullo stesso scenario (stesso seed = stessi
            dispersi e stesso rumore batteria per ogni run con quel seed).
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
        fixed_algorithm: se impostato (es. "spiral"), baseline non adattiva
            per il confronto richiesto dal relatore - nessuna chiamata LLM,
            il piano vola con questo algoritmo fisso per l'intera missione.
            Il guardrail deterministico resta SEMPRE attivo, invariato.

    Returns:
        dict con le metriche finali della missione.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

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
        algorithm=fixed_algorithm or "spiral",
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
        battery_model=model,
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

        if fixed_algorithm:
            # Nessuna chiamata LLM, ma la valutazione copertura/probabilità
            # resta attiva (skip_llm=True): senza, lo stop anticipato sarebbe
            # disponibile solo all'agente adattivo, confondendo l'effetto
            # della selezione algoritmica via LLM con quello della capacità
            # di riconoscere che la ricerca è completa - due cose diverse.
            coverage_fraction, likelihood_fraction = decide_algorithm(
                plan, item, center_proj, victims_gdf, meters_per_bin, flown_segments, environment_summary, rul_json,
                skip_llm=True,
            )
            _check_search_complete(plan, coverage_fraction, likelihood_fraction)
            continue

        agent_active = not (signal_loss and signal_connected)
        if agent_active:
            safety_call = decide_safety(plan, rul_json)
            if safety_call.get("tool") == "no_action" and plan.status != "RETURN_TO_BASE":
                coverage_fraction, likelihood_fraction = decide_algorithm(
                    plan, item, center_proj, victims_gdf, meters_per_bin, flown_segments, environment_summary, rul_json
                )
                _check_search_complete(plan, coverage_fraction, likelihood_fraction)
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
    print(f"Copertura finale: {final_coverage_fraction:.1%}")

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
    _seed = os.environ.get("MISSION_SEED")
    run_mission(
        battery_start_soc=float(os.environ.get("BATTERY_START_SOC", 1.0)),
        seed=int(_seed) if _seed is not None else None,
    )
