import random

# Evento puramente narrativo: nessuna funzione di sicurezza legge lo stato del
# segnale, serve solo a mostrare nel log quando l'agente sta operando senza
# contatto con la base. Non è un problema di prognostica (non degrada verso
# una soglia), va e viene.
SIGNAL_LOSS_PROBABILITY_PER_TICK = 0.05
SIGNAL_RECOVERY_PROBABILITY_PER_TICK = 0.3


def update_signal_status(signal_connected, t):
    if signal_connected:
        if random.random() < SIGNAL_LOSS_PROBABILITY_PER_TICK:  # nosec B311 - simulazione narrativa, non security-critical
            print(f"[t={t}s] Segnale radio perso - l'agente opera in autonomia locale.")
            return False
        return True
    if random.random() < SIGNAL_RECOVERY_PROBABILITY_PER_TICK:  # nosec B311 - vedi sopra
        print(f"[t={t}s] Segnale radio riacquisito.")
        return True
    return False
