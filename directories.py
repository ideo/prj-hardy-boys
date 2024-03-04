from pathlib import Path


REPO_ROOT_DIR = Path(__file__).parent

DATA_DIR = REPO_ROOT_DIR / "data"
APP_DIR = REPO_ROOT_DIR / "app"

DATA_DIR.mkdir(parents=True, exist_ok=True)
APP_DIR.mkdir(parents=True, exist_ok=True)
