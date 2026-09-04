from dataclasses import dataclass

import numpy as np

from agent.planning_bridge.plan import Plan

# Più virate/manovre => più corrente. spiral (curva continua) è il più
# economico, greedy (cammino erratico in salita sul gradiente) il più
# costoso. Valori plausibili, non misurati - assunzione di modellazione.
DEFAULT_SEARCHING_CURRENTS_BY_ALGORITHM = {
    "spiral": 2.2,
    "concentric_circles": 2.5,
    "pizza_zigzag": 3.2,
    "greedy": 3.6,
}

# Potenza media del calcolo di bordo (LLM + guardrail su Jetson Orin),
# misurata da tegrastats su 2 ambienti/6 run reali (analysis/data/pooled_tegrastats.csv,
# report Sezione RQ5 - Risorse ed energia): 8.56W (env6) / 8.58W (env44), media
# pooled 8.57W. Richiesta esplicita del relatore: il modello di batteria deve
# scaricarsi anche per il consumo energetico del calcolo IA di bordo, non solo
# per il movimento del drone (DEFAULT_SEARCHING_CURRENTS_BY_ALGORITHM sopra) -
# prima di questa modifica il calcolo era invisibile al bilancio energetico
# simulato. Costante, non letta live da tegrastats: il modello gira anche su
# QEMU (nessun tegrastats disponibile) e comunque la potenza di calcolo non
# dipende in modo osservabile dall'ambiente SAR (stesso modello, stesso
# hardware, Sezione RQ3 - throughput sostanzialmente identico tra env6/env44).
DEFAULT_COMPUTE_POWER_W = 8.57


@dataclass
class MissionLoadProfile:
    """Profilo di carico per ProgPy: callable (t, x=None) che
    simulate_to_threshold usa come future_loading_eqn. La corrente totale è
    la somma di due contributi indipendenti: il movimento (dipende dallo
    status del Plan condiviso - searching/returning - e, in ricerca,
    dall'algoritmo scelto, mai da un calcolo derivato dal tempo) e il calcolo
    di bordo (potenza costante misurata, convertita in corrente tramite la
    tensione istantanea della batteria)."""

    plan: Plan
    searching_currents_by_algorithm: dict
    returning_current: float
    battery_model: object
    compute_power_w: float = DEFAULT_COMPUTE_POWER_W

    def phase_at(self, t: float) -> str:
        return "returning" if self.plan.status == "RETURN_TO_BASE" else "searching"

    def _movement_current(self, t: float) -> float:
        if self.phase_at(t) == "returning":
            return self.returning_current
        return self.searching_currents_by_algorithm[self.plan.algorithm]

    def _compute_current(self, x) -> float:
        # ProgPy chiama future_loading_eqn(t) senza x al primissimo passo
        # (prima che lo stato sia inizializzato): in quel caso usiamo lo
        # stato iniziale x0 dei parametri del modello, che a quell'istante
        # coincide esattamente con lo stato vero della batteria.
        state = x if x is not None else self.battery_model.parameters["x0"]
        voltage_v = self.battery_model.output(state)["v"]
        return self.compute_power_w / voltage_v

    def __call__(self, t, x=None):
        return {"i": self._movement_current(t) + self._compute_current(x)}


def distance_to_ipp(path, ipp_point, distance_traveled_m: float) -> float:
    """Distanza in linea d'aria tra la posizione stimata del drone lungo
    `path` e l'IPP.

    distance_traveled_m va calcolata dal chiamante in base al tempo trascorso
    sotto l'algoritmo CORRENTE, non al tempo totale di missione: `path`
    riparte sempre dal centro ad ogni rigenerazione, quindi il tempo totale
    piazzerebbe la posizione in un punto arbitrario ad ogni cambio algoritmo.
    """
    current_position = path.interpolate(min(distance_traveled_m, path.length))
    return current_position.distance(ipp_point)


def rtb_required(remaining_range_m: float, distance_to_ipp_m: float, safety_margin_m: float) -> bool:
    """Guardrail deterministico: True se il range residuo non basta a coprire
    la distanza di rientro più un margine di sicurezza. Mai delegato all'LLM."""
    return remaining_range_m < distance_to_ipp_m + safety_margin_m


def state_of_charge(battery_model, battery_state) -> float:
    """Stato di carica (0-1), stessa formula usata internamente da ProgPy."""
    params = battery_model.parameters
    return (params["CMax"] - params["qMax"] + battery_state["qb"]) / params["CMax"]


# Tetto sulla durata simulata di ogni run Monte Carlo, non sull'accuratezza
# del RUL riportato: quando la batteria è sana il RUL vero può richiedere
# decine di migliaia di passi di integrazione per arrivare alla soglia EOD -
# precisione che qui non serve, perché in quella fascia range_to_budget_ratio
# è comunque ben oltre le soglie decisionali (verificato: anche troncato, il
# ratio resta correttamente nella fascia ">4"). Quando il RUL vero è basso
# (il caso che conta per la sicurezza) il cap non ha alcun effetto: la
# simulazione arriva alla soglia reale in pochi passi, ben prima del tetto.
# 12000s equivalgono a ~96km di range residuo a 8 m/s, ampio margine sopra i
# budget tipici osservati (2.000-60.000m).
RUL_SIMULATION_HORIZON_S = 12000


def rul_estimate_and_distribution(battery_model, load_profile, battery_state, n_runs: int = 20):
    """RUL puntuale e distribuzione (p10/p50/p90 in secondi) dalla STESSA
    serie di simulazioni Monte Carlo - non due batch indipendenti.

    Prima chiamavamo simulate_to_threshold una volta per il RUL puntuale
    (usato da remaining_range_m/rtb_required, quindi safety-critical) e altre
    20 volte, separatamente, per la distribuzione (usata dall'LLM per
    risk_posture) - con process_noise attivo sul modello, i due valori erano
    campioni indipendenti che potevano differire enormemente (verificato:
    quasi 2× tra loro sullo stesso stato di batteria). Guardrail e LLM
    finivano per ragionare su due numeri scorrelati che dovrebbero
    rappresentare la stessa grandezza. Un solo batch di n_runs campioni:
    il RUL puntuale è ora la mediana (p50) della stessa distribuzione
    riportata, e si risparmia anche una simulazione ridondante per tick.

    Richiede battery_model creato con process_noise != 0, altrimenti ogni
    run è identica e i percentili collassano sullo stesso valore.
    """
    samples = [
        battery_model.simulate_to_threshold(
            load_profile, x=battery_state, dt=1, save_freq=60, horizon=RUL_SIMULATION_HORIZON_S
        ).times[-1]
        for _ in range(n_runs)
    ]
    p10, p50, p90 = (float(v) for v in np.percentile(samples, [10, 50, 90]))
    return p50, {"p10": p10, "p50": p50, "p90": p90}


def health_check(plan, path, ipp_point, battery_state, battery_model, load_profile,
                  t, distance_traveled_m, search_speed_mps, safety_margin_m):
    """Ciclo centrale di health monitoring: prevede il RUL in avanti dallo
    stato attuale della batteria, calcola la posizione del drone e la
    distanza dall'IPP, applica il guardrail (mutando plan.status se serve),
    e assembla il JSON del RUL da passare all'agente."""
    rul_s, rul_dist = rul_estimate_and_distribution(battery_model, load_profile, battery_state)

    voltage_v = battery_model.output(battery_state)["v"]
    soc = state_of_charge(battery_model, battery_state)

    # Potenza prevista dal modello di batteria allo stato ATTUALE (non una
    # nuova simulazione Monte Carlo: load_profile(t, x) e' la stessa funzione
    # che ProgPy chiama internamente ad ogni passo di simulazione, qui
    # applicata una volta sola allo stato vero) - corrente totale (movimento +
    # calcolo di bordo) per la tensione istantanea. Confrontabile direttamente
    # con tegrastats.log (mW) per rispondere alla domanda aperta col relatore:
    # il modello di batteria e' un proxy sufficiente per l'energia, o serve
    # tegrastats come fonte autorevole? (vedi analysis/plot_predicted_vs_real_power.py)
    predicted_current_a = load_profile(t, battery_state)["i"]
    predicted_power_w = predicted_current_a * voltage_v

    distance_to_ipp_m = distance_to_ipp(path, ipp_point, distance_traveled_m)
    remaining_range_m = rul_s * search_speed_mps

    needs_rtb = rtb_required(remaining_range_m, distance_to_ipp_m, safety_margin_m)
    if needs_rtb:
        plan.status = "RETURN_TO_BASE"

    return {
        "timestamp_s": t,
        "battery": {
            "soc": soc,
            "voltage_v": voltage_v,
            "rul_s": rul_s,
            "rul_distribution": rul_dist,
            "predicted_power_w": predicted_power_w,
        },
        "derived": {
            "remaining_range_m": remaining_range_m,
            "distance_to_ipp_m": distance_to_ipp_m,
            "safety_margin_m": safety_margin_m,
            "rtb_required": needs_rtb,
        },
    }
