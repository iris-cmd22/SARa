from progpy.models import BatteryCircuit


def make_battery_model():
    """Batteria scalata ×5 rispetto al default (capacità = 5 celle
    equivalenti in parallelo): capacitanze scalate ×5, resistenze divise per
    5, stessa chimica di cella (V0/VEOD/parametri termici invariati)."""
    return BatteryCircuit(
        qMax=39281.627,
        CMax=38885,
        x0={"tb": 292.1, "qb": 39281.627, "qcp": 0, "qcs": 0},
        Cbp0=-1150, Cbp1=6.0, Cbp2=10399.5, Cbp3=135.27863,
        Cs=1171.935, Ccp=74.1115,
        Rp=2000, Rs=0.01077852, Rcp0=0.01395552, Rcp1=3.01056e-18,
        process_noise={"tb": 0.01, "qb": 100.0, "qcp": 2.5, "qcs": 2.5},
    )


def initial_battery_state(model, battery_start_soc):
    """A piena carica usa x0 così com'è. Per una carica parziale calcola il qb
    corrispondente invertendo state_of_charge: soc = (CMax-qMax+qb)/CMax."""
    if battery_start_soc >= 1.0:
        return model.parameters["x0"]
    params = model.parameters
    state = dict(params["x0"])
    state["qb"] = params["qMax"] - (1 - battery_start_soc) * params["CMax"]
    return state
