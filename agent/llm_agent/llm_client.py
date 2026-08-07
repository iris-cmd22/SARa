import json

import requests

from agent.battery_health.health_check import DEFAULT_SEARCHING_CURRENTS_BY_ALGORITHM
from agent.llm_agent.grammars import (
    RISK_POSTURE_GRAMMAR,
    RISK_POSTURE_WITH_RETURN_GRAMMAR,
    build_algorithm_grammar,
)

SERVER_URL = "http://127.0.0.1:8080"

# L'LLM non sceglie mai un tool a scelta libera (test sistematici hanno
# dimostrato classificazione inaffidabile tra 3+ opzioni, indipendentemente
# dal prompt). Ogni decisione è o deterministica in Python, o una
# classificazione tra poche etichette fisse vincolate da grammatica (vedi
# grammars.py).
RATIO_BY_RISK_POSTURE = {"conservative": 4.0, "balanced": 3.0, "aggressive": 2.0}

RISK_POSTURE_CRITERIA = (
    "The current budget is outside the healthy range and must be adjusted (the direction - "
    "increase or decrease - and the exact new value are computed separately, not by you). "
    "Your only job: classify the risk posture as 'conservative', 'balanced', or 'aggressive', "
    "based on how uncertain the RUL estimate is. Compare rul_distribution.p10 and "
    "rul_distribution.p90 against p50: a wide spread relative to p50 means high uncertainty, "
    "pick 'conservative' (larger safety margin). A narrow spread means low uncertainty, "
    "'aggressive' is acceptable (smaller margin, more budget for search). Otherwise 'balanced'."
)

RISK_POSTURE_EXAMPLES = (
    "Example 1:\nrul_distribution={\"p10\": 200, \"p50\": 1000, \"p90\": 2500}\n"
    'Response: {"risk_posture": "conservative"}\n\n'
    "Example 2:\nrul_distribution={\"p10\": 900, \"p50\": 1000, \"p90\": 1100}\n"
    'Response: {"risk_posture": "aggressive"}\n\n'
)

# Solo per la fascia ratio < 2 (margine già stretto): il guardrail resta
# comunque l'autorità ultima sul rientro imposto, questa etichetta permette
# solo di anticiparlo volontariamente.
RETURN_NOW_ADDENDUM = (
    "A fourth option is available here: if the RUL estimate is both low AND highly "
    "uncertain (p10 far below p50), continuing to search - even with a reduced budget - "
    "may not be worth the risk. In that case classify as 'return_now' instead, to end "
    "the search voluntarily. Use it only when the uncertainty itself is the concern, "
    "not just because the ratio is tight."
)

RETURN_NOW_EXAMPLE = (
    "Example 3:\nrul_distribution={\"p10\": 80, \"p50\": 900, \"p90\": 2200}\n"
    'Response: {"risk_posture": "return_now"}\n\n'
)

ALGORITHM_CRITERIA = (
    "Pick the algorithm with the highest value in detection_score_by_algorithm using "
    "set_search_algorithm, or no_action if current_algorithm already has the highest value."
)

ALGORITHM_EXAMPLES = (
    "Example 1:\ncurrent_algorithm=spiral, "
    'detection_score_by_algorithm={"spiral": 8, "greedy": 22}\n'
    'Response: {"tool": "set_search_algorithm", "arguments": {"algorithm": "greedy"}}\n\n'
    "Example 2:\ncurrent_algorithm=greedy, "
    'detection_score_by_algorithm={"spiral": 8, "greedy": 22}\n'
    'Response: {"tool": "no_action", "arguments": {}}\n\n'
)


def build_risk_posture_prompt(
    plan, rul_json: dict, range_to_budget_ratio: float, spendable_range_m: float, allow_return: bool = False
) -> str:
    """Prompt per la sola classificazione della risk posture. allow_return
    (True solo per ratio < 2) aggiunge 'return_now' come quarta etichetta -
    mai per ratio > 4, dove "tornare" non avrebbe senso."""
    derived = rul_json["derived"]
    battery = rul_json["battery"]
    criteria = RISK_POSTURE_CRITERIA + (f"\n\n{RETURN_NOW_ADDENDUM}" if allow_return else "")
    examples = RISK_POSTURE_EXAMPLES + (RETURN_NOW_EXAMPLE if allow_return else "")
    return (
        f"{criteria}\n\n"
        f"{examples}"
        "Current mission state:\n"
        f"current_budget_m={plan.budget_m}\n"
        f"remaining_range_m={derived['remaining_range_m']:.0f}\n"
        f"distance_to_ipp_m={derived['distance_to_ipp_m']:.0f}\n"
        f"safety_margin_m={derived['safety_margin_m']:.0f}\n"
        f"spendable_range_m={spendable_range_m:.0f}\n"
        f"range_to_budget_ratio={range_to_budget_ratio:.2f}\n"
        f"rul_distribution={battery['rul_distribution']}\n\n"
        "Respond ONLY with the JSON of the classification."
    )


def summarize_feature_probabilities(type_probabilities: dict, top_n: int = 3) -> str:
    """Riassunto testuale delle probabilità per tipo di feature (SAREnv
    LostPersonLocationGenerator.type_probabilities) - dati reali dell'ambiente
    corrente, non un placeholder."""
    if not type_probabilities:
        return "no feature data available"
    total = sum(type_probabilities.values())
    if total <= 0:
        return "no feature data available"
    ranked = sorted(type_probabilities.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    parts = [f"{name} ({value / total:.0%})" for name, value in ranked]
    return "highest-probability feature types: " + ", ".join(parts)


def build_algorithm_prompt(plan, efficiency_summary: dict, environment_summary: str = "") -> str:
    """Prompt con solo l'efficacia degli algoritmi - nessun dato di sicurezza,
    chiamato solo quando la sicurezza è già a posto."""
    environment_line = f"Search area: {environment_summary}\n" if environment_summary else ""
    return (
        "Available tools: set_search_algorithm (arguments: algorithm), "
        "no_action (arguments: none)\n\n"
        f"{ALGORITHM_CRITERIA}\n\n"
        f"{ALGORITHM_EXAMPLES}"
        "Current mission state:\n"
        f"current_algorithm={plan.algorithm}\n"
        f"searching_current_by_algorithm={DEFAULT_SEARCHING_CURRENTS_BY_ALGORITHM}\n"
        f"detection_score_by_algorithm={efficiency_summary}\n"
        f"{environment_line}\n"
        "Decide one action among the available tools and respond ONLY with the JSON of the call."
    )


def extract_json_object(text: str) -> dict:
    """Estrae il primo oggetto JSON bilanciato dal testo, ignorando rumore
    intorno (banner, echo del prompt, statistiche di timing)."""
    start = text.index("{")
    depth = 0
    for i, char in enumerate(text[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("Nessun oggetto JSON bilanciato trovato nell'output del modello")


def _query_llm(prompt: str, grammar: str, n_predict: int = 100, timeout_s: int = 60) -> dict:
    """Interroga llama-server (processo persistente, modello già in RAM), con
    la grammatica GBNF passata come parametro esteso (non json_schema, per
    evitare il bug noto di llama.cpp su json_schema+chatml)."""
    response = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": prompt}],
            "grammar": grammar,
            "max_tokens": n_predict,
            "temperature": 0.2,
            "seed": 42,
        },
        timeout=timeout_s,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return extract_json_object(content)


def query_safety_llm(plan, rul_json: dict) -> dict:
    """Decisione di sicurezza a zone (range_to_budget_ratio = spendable_range_m
    / budget_m, dove spendable_range_m riserva già la tratta di rientro e il
    margine di sicurezza):

      2 <= ratio <= 4  -> no_action, deterministico (margine sano)
      ratio < 2 o > 4  -> LLM classifica la risk posture; Python calcola
                          budget_m = spendable_range_m / target_ratio

    Solo nella fascia ratio < 2 la classificazione ammette anche 'return_now'
    (rientro volontario anticipato) - il rientro IMPOSTO resta comunque
    sempre compito esclusivo del guardrail rtb_required in health_check(),
    indipendente da questa funzione.
    """
    derived = rul_json["derived"]
    spendable_range_m = (
        derived["remaining_range_m"] - derived["distance_to_ipp_m"] - derived["safety_margin_m"]
    )
    range_to_budget_ratio = (
        spendable_range_m / plan.budget_m if plan.budget_m > 0 else float("inf")
    )
    if 2 <= range_to_budget_ratio <= 4:
        return {"tool": "no_action", "arguments": {}}

    allow_return = range_to_budget_ratio < 2
    classification = _query_llm(
        build_risk_posture_prompt(
            plan, rul_json, range_to_budget_ratio, spendable_range_m, allow_return=allow_return
        ),
        RISK_POSTURE_WITH_RETURN_GRAMMAR if allow_return else RISK_POSTURE_GRAMMAR,
        n_predict=20,
    )
    posture = classification.get("risk_posture")
    if posture == "return_now":
        return {"tool": "request_return_to_base", "arguments": {}}
    target_ratio = RATIO_BY_RISK_POSTURE.get(posture, 3.0)
    new_budget_m = spendable_range_m / target_ratio
    return {"tool": "set_budget", "arguments": {"budget_m": new_budget_m}}


def query_algorithm_llm(plan, efficiency_summary: dict, environment_summary: str = "") -> dict:
    grammar = build_algorithm_grammar(efficiency_summary.keys())
    return _query_llm(build_algorithm_prompt(plan, efficiency_summary, environment_summary), grammar)
