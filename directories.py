from pathlib import Path


REPO_ROOT_DIR = Path(__file__).parent
DATA_DIR = REPO_ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)