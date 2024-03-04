from pathlib import Path


REPO_ROOT_DIR = Path(__file__).parent

DATA_DIR = REPO_ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

APP_DIR = REPO_ROOT_DIR / "app"
APP_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
