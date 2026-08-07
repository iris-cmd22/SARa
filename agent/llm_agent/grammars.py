"""Tutto ciò che riguarda la FORMA vincolata dell'output del modello (file
GBNF statici + costruzione dinamica) - separato da llm_client.py, che si
occupa di COSA chiedere (prompt) e COME chiederlo (HTTP)."""

from pathlib import Path

# Path relativo al file, non assoluto: i .gbnf vivono nel repo (gbnf/), non in
# una cartella llama.cpp esterna - così il progetto funziona su qualunque
# macchina/percorso in cui viene clonato.
_GBNF_DIR = Path(__file__).parent / "gbnf"
RISK_POSTURE_GRAMMAR_PATH = _GBNF_DIR / "risk_posture.gbnf"
RISK_POSTURE_WITH_RETURN_GRAMMAR_PATH = _GBNF_DIR / "risk_posture_with_return.gbnf"

with open(RISK_POSTURE_GRAMMAR_PATH) as _f:
    RISK_POSTURE_GRAMMAR = _f.read()

with open(RISK_POSTURE_WITH_RETURN_GRAMMAR_PATH) as _f:
    RISK_POSTURE_WITH_RETURN_GRAMMAR = _f.read()


def build_algorithm_grammar(candidate_algorithm_names) -> str:
    """Grammatica GBNF costruita a runtime: 'algorithm' può assumere SOLO uno
    dei nomi effettivamente proposti in questo tick (le chiavi già filtrate
    per fase di efficiency_summary) - un file statico con tutti e 4 gli
    algoritmi permetterebbe al modello di scegliere 'greedy' anche in fase
    esaustiva, dove non dovrebbe essere un'opzione."""
    algorithm_alternatives = " | ".join(
        f'"\\"{name}\\""' for name in sorted(candidate_algorithm_names)
    )
    return (
        'root ::= set-algorithm-call | no-action-call\n'
        'set-algorithm-call ::= "{" ws "\\"tool\\"" ws ":" ws "\\"set_search_algorithm\\"" ws "," ws '
        '"\\"arguments\\"" ws ":" ws "{" ws "\\"algorithm\\"" ws ":" ws algorithm ws "}" ws "}"\n'
        'no-action-call ::= "{" ws "\\"tool\\"" ws ":" ws "\\"no_action\\"" ws "," ws "\\"arguments\\"" ws ":" '
        'ws "{" ws "}" ws "}"\n'
        f'algorithm ::= {algorithm_alternatives}\n'
        'ws ::= [ \\t\\n]*\n'
    )
