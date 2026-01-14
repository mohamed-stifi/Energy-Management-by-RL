"""Configuration centralisée pour tout le projet."""

import os
from pathlib import Path

# Dossier racine du projet
PROJECT_ROOT = Path(__file__).parent.parent

# Dossiers DATA
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_OFFLINE_DIR = DATA_DIR / "offline_dataset"

# Dossiers RÉSULTATS
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_LOGS_DIR = RESULTS_DIR / "logs"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"

# Créer les dossiers s'ils n'existent pas
for directory in [DATA_RAW_DIR, DATA_OFFLINE_DIR, RESULTS_LOGS_DIR, RESULTS_FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ⚠️ IMPORTANT : Configurer Sinergym pour utiliser data/raw/
# Cette ligne DOIT être exécutée AVANT l'import de Sinergym
os.environ['SINERGYM_LOGS_DIR'] = str(DATA_RAW_DIR.absolute())
print(f"[CONFIG] Sinergym logs directory: {os.environ['SINERGYM_LOGS_DIR']}")