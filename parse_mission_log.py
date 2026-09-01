"""
Converte il log della missione (mission_run.log o mission_run.json) in tick_metrics.csv.

Uso:
    python parse_mission_log.py mission_run.json tick_metrics.csv
    python parse_mission_log.py mission_run.log tick_metrics.csv

Due formati di input supportati (autodetect dal contenuto):

1. journald JSON (preferito - niente overhead di "date" per riga, timestamp al
   microsecondo, isola pulito le righe della missione dal resto del journal):
       python -u -m agent.llm_agent.mission_loop 2>&1 | systemd-cat -t mission_loop --priority=info
       journalctl -t mission_loop -o json --no-pager > mission_run.json

2. Wrapper testuale (fallback se systemd-cat/journalctl non sono disponibili):
       python -u -m agent.llm_agent.mission_loop 2>&1 | \
           while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%3N)" "$line"; done > mission_run.log
   Formato riga: <unix_timestamp> <riga di output originale>

Marcatori usati (stringhe esatte da agent/llm_agent/decisions.py e mission_loop.py):
- "[t=Ns] RUL="                                  -> inizio tick N (fine guardrail)
- "LLM (sicurezza) ha scelto:"                    -> ramo safety_llm
- "Errore nella decisione di sicurezza"           -> fallback safety
- "Copertura area finora:"                        -> ramo algorithm_llm (decide_algorithm chiamato)
- "Fase: copertura esaustiva"                     -> sotto-fase esaustiva
- "Fase: ricerca rapida"                          -> sotto-fase rapida (valuta TUTTI gli algoritmi)
- "Errore nella decisione sull'algoritmo"         -> fallback algorithm
- "Guardrail deterministico ha forzato il rientro." -> tick_type = rtb
- "Ricerca sostanzialmente completa" -> tick_type = search_complete (solo run patchate
  con lo stop anticipato su copertura/probabilita', vedi mission_loop.py)
- "Segnale connesso - agente in stand-by"         -> tick_type = frozen
"""
import csv
import json
import re
import sys

TICK_START_RE = re.compile(r"\[t=(\d+)s\] RUL=")


def parse_journal_json(path):
    events = []  # (timestamp: float, text: str)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            entry = json.loads(raw)
            message = entry.get("MESSAGE", "")
            if isinstance(message, list):
                # journald a volte codifica messaggi non-UTF8 come array di byte
                message = bytes(message).decode("utf-8", errors="replace")
            ts_us = entry.get("__REALTIME_TIMESTAMP")
            if ts_us is None:
                continue
            events.append((int(ts_us) / 1_000_000, message))
    return events


def parse_text_log(path):
    events = []  # (timestamp: float, text: str)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            parts = raw.split(" ", 1)
            if len(parts) != 2:
                continue
            ts_str, text = parts
            try:
                ts = float(ts_str)
            except ValueError:
                continue
            events.append((ts, text))
    return events


def parse_log(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline().strip()
    if first_line.startswith("{"):
        return parse_journal_json(path)
    return parse_text_log(path)


MISSION_START_RE = re.compile(r"--- Missione avviata:")


def keep_last_mission(events):
    """journalctl -t mission_loop include TUTTI gli avvii passati (il tag e' persistente
    tra riavvii della VM) - senza questo filtro, tick di missioni diverse vengono trattati
    come se fossero sequenziali nella stessa run, producendo guardrail_s/tick_total_s
    assurdi ai punti di giunzione. Tiene solo l'ultimo avvio (quello piu' recente)."""
    start_indices = [i for i, (_, text) in enumerate(events) if MISSION_START_RE.search(text)]
    if len(start_indices) <= 1:
        return events
    return events[start_indices[-1]:]


def classify_and_extract(events):
    tick_starts = [i for i, (_, text) in enumerate(events) if TICK_START_RE.search(text)]
    rows = []

    for k, start_idx in enumerate(tick_starts):
        ts_start, text_start = events[start_idx]
        t_match = TICK_START_RE.search(text_start)
        t_value = int(t_match.group(1))

        end_idx = tick_starts[k + 1] - 1 if k + 1 < len(tick_starts) else len(events) - 1
        window = events[start_idx:end_idx + 1]
        window_text = "\n".join(text for _, text in window)
        decision_end_ts = events[end_idx][0]

        if "Guardrail deterministico ha forzato il rientro." in window_text:
            tick_type = "rtb"
        elif "Ricerca sostanzialmente completa" in window_text:
            tick_type = "search_complete"
        elif "Segnale connesso - agente in stand-by" in window_text:
            tick_type = "frozen"
        elif "Copertura area finora:" in window_text:
            tick_type = "algorithm_llm"
        elif "LLM (sicurezza) ha scelto:" in window_text or "Errore nella decisione di sicurezza" in window_text:
            tick_type = "safety_llm"
        else:
            tick_type = "unknown"

        if tick_type == "algorithm_llm":
            if "Fase: copertura esaustiva" in window_text:
                algorithm_phase = "copertura_esaustiva"
            elif "Fase: ricerca rapida" in window_text:
                algorithm_phase = "ricerca_rapida"
            else:
                algorithm_phase = "unknown"
        else:
            algorithm_phase = ""

        fallback_safety = "Errore nella decisione di sicurezza" in window_text
        fallback_algorithm = "Errore nella decisione sull'algoritmo" in window_text

        if k == 0:
            guardrail_s = ""
            tick_total_s = ""
        else:
            # rows[-1]["_decision_end_ts"] è l'ultima riga stampata dal tick precedente,
            # non il suo marcatore di inizio "[t=...] RUL=" - va usato per ogni k>0,
            # senza un caso speciale per k==1 (che introdurrebbe di nuovo l'errore).
            prev_decision_end_ts = rows[-1]["_decision_end_ts"]
            guardrail_s = ts_start - prev_decision_end_ts
            tick_total_s = decision_end_ts - prev_decision_end_ts

        llm_block_s = decision_end_ts - ts_start

        rows.append({
            "t": t_value,
            "tick_type": tick_type,
            "algorithm_phase": algorithm_phase,
            "fallback_safety": fallback_safety,
            "fallback_algorithm": fallback_algorithm,
            "guardrail_s": guardrail_s,
            "llm_block_s": llm_block_s,
            "tick_total_s": tick_total_s,
            "_decision_end_ts": decision_end_ts,  # colonna interna, rimossa prima di scrivere il CSV
        })

    return rows


def write_csv(rows, out_path):
    fieldnames = [
        "t", "tick_type", "algorithm_phase", "fallback_safety", "fallback_algorithm",
        "guardrail_s", "llm_block_s", "tick_total_s",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python parse_mission_log.py mission_run.log tick_metrics.csv")
        sys.exit(1)

    events = parse_log(sys.argv[1])
    events = keep_last_mission(events)
    rows = classify_and_extract(events)
    write_csv(rows, sys.argv[2])

    n = len(rows)
    fallbacks = sum(1 for r in rows if r["fallback_safety"] or r["fallback_algorithm"])
    print(f"{n} tick estratti -> {sys.argv[2]}")
    print(f"Fallback totali: {fallbacks}")
    from collections import Counter
    print("Distribuzione tick_type:", dict(Counter(r["tick_type"] for r in rows)))
