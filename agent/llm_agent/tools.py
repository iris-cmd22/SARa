from agent.planning_bridge.plan import Plan

VALID_ALGORITHMS = {"spiral", "concentric_circles", "pizza_zigzag", "greedy"}


def set_search_algorithm(plan: Plan, algorithm: str) -> None:
    """Tool: l'LLM sceglie solo `algorithm`, plan viene iniettato dal dispatcher."""
    if algorithm not in VALID_ALGORITHMS:
        raise ValueError(f"Algoritmo non valido: {algorithm}")
    plan.algorithm = algorithm


def set_budget(plan: Plan, latest_rul: dict, budget_m: float) -> None:
    """Tool: l'LLM sceglie solo `budget_m`; validato contro il range residuo
    sicuro prima di applicarlo, mai fidato ciecamente."""
    derived = latest_rul["derived"]
    max_safe_budget = derived["remaining_range_m"] - derived["distance_to_ipp_m"] - derived["safety_margin_m"]
    if budget_m > max_safe_budget:
        raise ValueError(f"Budget richiesto ({budget_m}) supera il range residuo sicuro ({max_safe_budget})")
    plan.budget_m = budget_m


def set_altitude(plan: Plan, altitude_m: float) -> None:
    """Tool: l'LLM sceglie solo `altitude_m`."""
    plan.altitude_m = altitude_m


def request_return_to_base(plan: Plan) -> None:
    """Tool: l'LLM sceglie volontariamente di terminare la ricerca e tornare.
    Il guardrail deterministico resta comunque attivo in parallelo e
    indipendente da questa scelta - il rischio è asimmetrico: rientrare troppo
    presto è subottimale, non rientrare quando serve sarebbe pericoloso, ed è
    compito del guardrail impedirlo sempre, a prescindere da questa scelta."""
    plan.status = "RETURN_TO_BASE"
