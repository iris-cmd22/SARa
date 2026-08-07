import dataclasses

from sarenv.analytics.evaluator import PathGeneratorConfig
from sarenv.analytics.metrics import PathEvaluator
from sarenv.analytics.paths import (
    generate_concentric_circles_path,
    generate_greedy_path,
    generate_pizza_zigzag_path,
    generate_spiral_path,
)

ALGORITHMS = {
    "spiral": generate_spiral_path,
    "concentric_circles": generate_concentric_circles_path,
    "pizza_zigzag": generate_pizza_zigzag_path,
    "greedy": generate_greedy_path,
}

# Coprono l'area sistematicamente dal centro verso il bordo; greedy insegue la
# probabilità, non ne fa parte.
EXHAUSTIVE_ALGORITHMS = {"spiral", "concentric_circles", "pizza_zigzag"}

# Retta ancorata ai due estremi osservati nell'esperimento a 36 run (4 livelli
# x 3 batterie x 3 ripetizioni): temperate_flat (radius_km=0.6, SAREnv size
# "small") già copriva quasi il 100% con l'altitudine di default, quindi resta
# invariata lì; dry_mountainous (radius_km=1.6) era il caso peggiore (13-32%
# di copertura), a cui è ancorato il tetto massimo. Valori di RADIUS_* per
# size "small" da sarenv/utils/lost_person_behavior.py.
ALTITUDE_BASE_M = 80.0
ALTITUDE_MAX_M = 160.0
RADIUS_BASELINE_KM = 0.6  # temperate_flat - qui il default basta già
RADIUS_WORST_CASE_KM = 1.6  # dry_mountainous - qui serve il tetto massimo


def altitude_for_radius(radius_km: float) -> float:
    """Altitudine iniziale proporzionale al raggio della ROI, non applicata
    indiscriminatamente a tutti gli scenari: resta ad ALTITUDE_BASE_M per
    ambienti piccoli/flat (dove non serve), cresce verso ALTITUDE_MAX_M solo
    per quelli con ROI estesa (tipicamente mountainous).

    Altitudine più alta => detection_radius più ampio (vedi
    PathEvaluator.detection_radius) => più area vista a parità di metri
    percorsi - compensa una ROI più estesa senza costare corrente
    aggiuntiva, perché MissionLoadProfile calcola la corrente solo da
    plan.status/plan.algorithm, mai da altitude_m.

    Semplificazione dichiarata: nella realtà volare più in alto peggiora la
    risoluzione della camera (rilevamento meno affidabile), un effetto che
    SAREnv/ProgPy non modellano - qui il guadagno è "gratis" solo nella
    simulazione.
    """
    slope = (ALTITUDE_MAX_M - ALTITUDE_BASE_M) / (RADIUS_WORST_CASE_KM - RADIUS_BASELINE_KM)
    altitude = ALTITUDE_BASE_M + slope * max(0.0, radius_km - RADIUS_BASELINE_KM)
    return min(altitude, ALTITUDE_MAX_M)


def build_path(plan, center_x, center_y, max_radius, heatmap, bounds):
    """Genera il percorso SAREnv per l'algoritmo e i parametri correnti di
    `plan` (singolo drone: restituisce solo il primo path)."""
    config = PathGeneratorConfig(
        num_drones=plan.num_drones,
        budget=plan.budget_m,
        fov_degrees=plan.fov_deg,
        altitude_meters=plan.altitude_m,
        overlap_ratio=plan.overlap_ratio,
    )
    params = config.get_params_dict(center_x, center_y, max_radius, heatmap, bounds)
    generator = ALGORITHMS[plan.algorithm]
    return generator(**params)[0]


def algorithm_efficiency_summary(
    plan, center_x, center_y, max_radius, heatmap, bounds,
    victims_gdf, meters_per_bin, discount_factor=0.999, candidate_algorithms=None,
):
    """Punteggio di rilevamento scontato nel tempo per ciascun algoritmo
    candidato, allo stesso budget/altitudine del piano corrente - il lato
    "beneficio" complementare al costo di batteria in MissionLoadProfile.

    candidate_algorithms, se dato, limita il confronto a quel sottoinsieme
    (usato per restringere le opzioni in fase di copertura esaustiva)."""
    algorithms_to_evaluate = candidate_algorithms if candidate_algorithms is not None else ALGORITHMS
    evaluator = PathEvaluator(heatmap, bounds, victims_gdf, plan.fov_deg, plan.altitude_m, meters_per_bin)
    scores = {}
    for algorithm_name in algorithms_to_evaluate:
        hypothetical_plan = dataclasses.replace(plan, algorithm=algorithm_name)
        path = build_path(hypothetical_plan, center_x, center_y, max_radius, heatmap, bounds)
        metrics = evaluator.calculate_all_metrics([path], discount_factor)
        scores[algorithm_name] = metrics["total_time_discounted_score"]
    return scores
