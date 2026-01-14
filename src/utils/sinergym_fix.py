"""Fix pour rediriger Sinergym vers data/raw/ DÈS LE DÉPART."""

import os
import shutil
from pathlib import Path


def setup_sinergym_dirs():
    """Configure Sinergym pour créer DIRECTEMENT dans data/raw/."""
    
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_RAW = PROJECT_ROOT / "data" / "raw"
    
    # Créer le dossier s'il n'existe pas
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    
    # 🔑 CLEF : Dire à Sinergym où CRÉER les envs dès le départ
    os.environ['SINERGYM_LOGS_DIR'] = str(DATA_RAW.absolute())
    
    # Change le working directory de Sinergym
    # (Sinergym crée des sous-dossiers à partir du cwd)
    original_cwd = os.getcwd()
    os.chdir(str(DATA_RAW))
    os.chdir(original_cwd)  # Revenir au projet root
    
    # Cleanup : si des Eplus restent à la racine, les déplacer
    root_eplus_dirs = list(Path('/').glob('Eplus-*-res*'))
    if root_eplus_dirs:
        print(f"⚠️ Déplacement de {len(root_eplus_dirs)} dossiers oubliés de / vers data/raw/")
        for dir_path in root_eplus_dirs:
            try:
                dest = DATA_RAW / dir_path.name
                if not dest.exists():
                    shutil.move(str(dir_path), str(dest))
                    print(f"   ✅ {dir_path.name}")
            except Exception as e:
                print(f"   ❌ {dir_path.name}: {e}")
    
    return DATA_RAW


if __name__ == '__main__':
    setup_sinergym_dirs()
    print("\n✅ Sinergym correctement configuré")