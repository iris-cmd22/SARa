from math import pi

import numpy as np
from sarenv.analytics.metrics import PathEvaluator

from agent.llm_agent.dispatcher import dispatch
from agent.llm_agent.llm_client import query_algorithm_llm, query_safety_llm
from agent.planning_bridge.path_builder import EXHAUSTIVE_ALGORITHMS, algorithm_efficiency_summary

# Fase esaustiva quando l'area O la probabilità osservata superano la
# rispettiva soglia - la probabilità perché greedy la accumula molto più in
# fretta dell'area grezza (altrimenti resterebbe bloccato in fase rapida per
# sempre).
COVERAGE_PHASE_THRESHOLD = 0.5
LIKELIHOOD_PHASE_THRESHOLD = 0.7


def print_tick_status(t, rul_json, plan, signal_connected):
    print(
        f"\n[t={t}s] RUL={rul_json['battery']['rul_s']:.0f}s  "
        f"dist_ipp={rul_json['derived']['distance_to_ipp_m']:.0f}m  "
        f"rtb_required={rul_json['derived']['rtb_required']}  status={plan.status}  "
        f"segnale={'connesso' if signal_connected else 'PERSO'}"
    )


def decide_safety(plan, rul_json):
    """Chiama query_safety_llm e applica la decisione. Il modello vede SOLO i
    dati di sicurezza - nessun dato di efficacia algoritmo."""
    try:
        safety_call = query_safety_llm(plan, rul_json)
        print(f"LLM (sicurezza) ha scelto: {safety_call}")
        dispatch(safety_call, plan=plan, latest_rul=rul_json)
        print(f"Plan aggiornato: {plan}")
        return safety_call
    except Exception as e:
        print(f"Errore nella decisione di sicurezza (proseguo con il piano invariato): {e}")
        return {"tool": None}


def decide_algorithm(plan, item, center_proj, victims_gdf, meters_per_bin, flown_segments,
                      environment_summary, rul_json, skip_llm=False):
    """Copertura/probabilità osservate (sulla traiettoria persistente, non sul
    solo path di questo tick) decidono la fase; l'LLM sceglie poi l'algoritmo
    migliore tra i soli candidati di quella fase.

    Args:
        skip_llm: se True, calcola solo copertura/probabilità (pura
            geometria via PathEvaluator, nessun LLM) e le ritorna subito -
            usato dalla baseline non adattiva (fixed_algorithm in
            mission_loop.py) cosi' lo stop anticipato su copertura/probabilità
            resta identico tra le due condizioni, e il confronto isola solo
            l'effetto della selezione algoritmica via LLM, non anche la
            capacità di riconoscere che la ricerca è completa.

    Returns:
        (coverage_fraction, likelihood_fraction): usati dal chiamante per
        decidere se la ricerca è di fatto completa (vedi mission_loop.py).
    """
    evaluator = PathEvaluator(item.heatmap, item.bounds, victims_gdf, plan.fov_deg, plan.altitude_m, meters_per_bin)
    coverage_metrics = evaluator.calculate_all_metrics(flown_segments, discount_factor=0.999)
    total_roi_area_km2 = pi * (item.radius_km ** 2)
    coverage_fraction = coverage_metrics["area_covered"] / total_roi_area_km2 if total_roi_area_km2 > 0 else 0

    total_heatmap_probability = float(np.sum(item.heatmap))
    likelihood_fraction = (
        coverage_metrics["total_likelihood_score"] / total_heatmap_probability
        if total_heatmap_probability > 0 else 0
    )
    print(f"Copertura area finora: {coverage_fraction:.1%}  |  probabilità osservata: {likelihood_fraction:.1%}")

    if skip_llm:
        return coverage_fraction, likelihood_fraction

    exhaustive_phase = (
        coverage_fraction >= COVERAGE_PHASE_THRESHOLD or likelihood_fraction >= LIKELIHOOD_PHASE_THRESHOLD
    )
    candidate_algorithms = EXHAUSTIVE_ALGORITHMS if exhaustive_phase else None
    print(f"Fase: {'copertura esaustiva' if exhaustive_phase else 'ricerca rapida (tutti gli algoritmi)'}")

    efficiency_summary = algorithm_efficiency_summary(
        plan, center_proj.x, center_proj.y, item.radius_km * 1000, item.heatmap, item.bounds,
        victims_gdf, meters_per_bin, candidate_algorithms=candidate_algorithms,
    )
    print(f"Efficacia stimata per algoritmo: {efficiency_summary}")
    try:
        algorithm_call = query_algorithm_llm(plan, efficiency_summary, environment_summary)
        print(f"LLM (algoritmo) ha scelto: {algorithm_call}")
        dispatch(algorithm_call, plan=plan, latest_rul=rul_json)
        print(f"Plan aggiornato: {plan}")
    except Exception as e:
        print(f"Errore nella decisione sull'algoritmo (proseguo con il piano invariato): {e}")

    return coverage_fraction, likelihood_fraction
