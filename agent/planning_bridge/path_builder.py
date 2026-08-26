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

# Retta ancorata a due estremi osservati - ma NON più agli stessi due della
# prima versione. Primo esperimento a 36 run (4 livelli x 3 batterie x 3
# ripetizioni): temperate_flat (radius_km=0.6) già ~100% di copertura col
# default, invariato. Con la versione iniziale della retta (tetto raggiunto
# solo a radius_km=1.6, dry_mountainous), temperate_mountainous (radius_km=
# 1.1, altitudine risultante 120m) è rimasto il livello peggiore in assoluto
# nel secondo esperimento - media 33.6% di dispersi trovati su 3 batterie,
# PEGGIORE di dry_mountainous (73.7% medio) nonostante una ROI più piccola
# (area minore da coprire) - mentre dry_mountainous con l'altitudine piena
# (160m) performava bene. Prova che 120m a 1.1km non bastava quanto 160m
# bastava a 1.6km: il tetto va raggiunto prima. Retaratura: il tetto massimo
# si raggiunge già a radius_km=1.1 (non più 1.6) - dry_flat (radius_km=1.3)
# ne beneficia di conseguenza (resta comunque sotto il tetto prima, ora lo
# tocca anche lui, coerente con la sua debolezza osservata a batteria bassa).
# Valori di RADIUS_* per size "small" da sarenv/utils/lost_person_behavior.py.
ALTITUDE_BASE_M = 80.0
ALTITUDE_MAX_M = 160.0
RADIUS_BASELINE_KM = 0.6  # temperate_flat - qui il default basta già
RADIUS_WORST_CASE_KM = 1.1  # temperate_mountainous - qui serve già il tetto massimo


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
