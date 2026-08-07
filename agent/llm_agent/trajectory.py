import dataclasses
from math import radians, tan

from shapely.geometry import LineString, Point
from shapely.ops import substring, unary_union

from agent.planning_bridge.path_builder import build_path

# Budget fisso usato SOLO per generare la forma del percorso, mai per la
# sicurezza (quella resta su plan.budget_m). spiral/concentric/pizza sono
# prefisso-stabili al variare del budget, greedy no (route diversa, non solo
# più lunga) - generarlo sempre con questo valore evita che la rotta di
# greedy cambi ad ogni oscillazione del budget di sicurezza.
PATH_SHAPE_REFERENCE_BUDGET_M = 200000


def connect_segments_for_plotting(segments):
    """Collega con brevi segmenti gli spezzoni non contigui di flown_segments,
    solo per una traiettoria visivamente continua nei grafici. I connettori
    non entrano mai in calculate_all_metrics: conterebbero come "cercato" un
    tratto in realtà solo teletrasportato da un tick all'altro."""
    if not segments:
        return []
    connected = [segments[0]]
    for prev, curr in zip(segments, segments[1:]):
        prev_end = Point(prev.coords[-1])
        curr_start = Point(curr.coords[0])
        if prev_end.distance(curr_start) > 1e-6:
            connected.append(LineString([prev_end, curr_start]))
        connected.append(curr)
    return connected


def skip_distance_past_covered_ground(path, flown_segments, detection_radius, step_m=200):
    """Distanza lungo `path` da cui in poi il tratto è terreno NUOVO rispetto
    a flown_segments. SAREnv non ha memoria (ogni path riparte cieco dal
    centro); usiamo quello che sappiamo già per non ripercorrere terreno già
    esplorato da un algoritmo precedente. Euristica a passi fissi - non
    ottima se il path rientra in una zona già coperta dopo averla lasciata,
    ma elimina la parte peggiore dello spreco."""
    if not flown_segments:
        return 0.0
    already_covered = unary_union([segment.buffer(detection_radius) for segment in flown_segments])
    distance = 0.0
    while distance < path.length:
        if not already_covered.contains(path.interpolate(distance)):
            return distance
        distance += step_m
    return path.length


def advance_trajectory(plan, item, center_proj, flown_segments, distance_within_algorithm,
                        algorithm_switched, tick_seconds, search_speed_mps):
    """Genera il path di riferimento del tick e avanza l'accumulo della
    traiettoria persistente, aggiungendo solo il tratto nuovo dal
    contachilometri precedente a quello di questo tick.

    Se l'algoritmo è appena cambiato, il contachilometri non riparte da 0 ma
    salta avanti fino al primo terreno nuovo (vedi skip_distance_past_covered_
    ground) - il path stesso riparte sempre dal centro, limite di SAREnv non
    aggirabile.

    Ritorna (path, nuovo_contachilometri).
    """
    shape_plan = dataclasses.replace(plan, budget_m=PATH_SHAPE_REFERENCE_BUDGET_M)
    path = build_path(shape_plan, center_proj.x, center_proj.y, item.radius_km * 1000, item.heatmap, item.bounds)

    if algorithm_switched:
        detection_radius = plan.altitude_m * tan(radians(plan.fov_deg / 2))
        distance_within_algorithm = skip_distance_past_covered_ground(path, flown_segments, detection_radius)

    new_distance = min(distance_within_algorithm + tick_seconds * search_speed_mps, path.length)
    if new_distance > distance_within_algorithm:
        flown_segments.append(substring(path, distance_within_algorithm, new_distance))
    return path, new_distance


def build_final_trajectory(flown_segments, plan, ipp_point):
    """Traiettoria persistente accumulata tick per tick, non l'ultimo path
    troncato - un crollo del budget nell'ultimo tratto non cancella i km già
    esplorati. Connettori cosmetici solo per il disegno; il rilevamento dei
    dispersi usa flown_segments "grezzi"."""
    trajectory_segments = connect_segments_for_plotting(flown_segments)
    if plan.status == "RETURN_TO_BASE" and trajectory_segments:
        final_position = Point(trajectory_segments[-1].coords[-1])
        trajectory_segments.append(LineString([final_position, ipp_point]))
    return trajectory_segments
