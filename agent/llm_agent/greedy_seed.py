import numpy as np

# SAREnv.generate_greedy_path non è deterministico (RNG interno senza seed,
# non esposto come parametro) - due chiamate identiche producono percorsi
# diversi. Senza toccare i file di SAREnv, intercettiamo qui la funzione
# globale numpy.random.default_rng così che ogni sua chiamata (inclusa quella
# interna a SAREnv) usi un seed fisso. Importare questo modulo applica la
# patch come effetto collaterale - va importato prima di qualunque
# generazione di path.
GREEDY_PATH_RNG_SEED = 42
_original_default_rng = np.random.default_rng
np.random.default_rng = lambda *args, **kwargs: _original_default_rng(GREEDY_PATH_RNG_SEED)
