import sys
from pathlib import Path


def get_base_dir() -> Path:
    """
    Works in dev and when packaged with PyInstaller.

    - Dev: project root (two levels up from scripts/)
    - PyInstaller: folder containing the .exe (where we also ship /data)
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()
CENTERS_PATH = BASE_DIR / "data" / "centers.csv"
