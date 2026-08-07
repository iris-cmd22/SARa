import inspect

from agent.llm_agent import tools as tools_module

TOOLS = {
    "set_search_algorithm": tools_module.set_search_algorithm,
    "set_budget": tools_module.set_budget,
    "set_altitude": tools_module.set_altitude,
    "request_return_to_base": tools_module.request_return_to_base,
}

# Mai scelti dall'LLM: se il modello li infila per sbaglio in "arguments", li ignoriamo.
INJECTED_PARAMS = {"plan", "latest_rul"}


def dispatch(tool_call: dict, plan, latest_rul: dict):
    """Interpreta l'output JSON del modello e chiama il tool corrispondente.

    Filtra gli argomenti proposti dal modello tenendo solo le chiavi che la
    funzione bersaglio accetta davvero (via inspect.signature) - il modello,
    anche con la grammatica attiva, a volte infila chiavi extra."""
    tool_name = tool_call.get("tool")
    if tool_name == "no_action":
        return None

    if tool_name not in TOOLS:
        raise ValueError(f"Tool sconosciuto: {tool_name!r}")

    func = TOOLS[tool_name]
    accepted_params = set(inspect.signature(func).parameters.keys())

    llm_args = tool_call.get("arguments") or {}
    call_kwargs = {
        key: value for key, value in llm_args.items()
        if key in accepted_params and key not in INJECTED_PARAMS
    }
    if "plan" in accepted_params:
        call_kwargs["plan"] = plan
    if "latest_rul" in accepted_params:
        call_kwargs["latest_rul"] = latest_rul

    return func(**call_kwargs)
