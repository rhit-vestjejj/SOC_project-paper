"""Shared configuration and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TAB_DIR = OUTPUT_DIR / "tables"

for d in (DATA_DIR, FIG_DIR, TAB_DIR):
    d.mkdir(parents=True, exist_ok=True)

START_DATE = "2005-01-01"
END_DATE = "2025-12-31"

ROLLING_WINDOW = 60
DEFAULT_K = 2.0
DEFAULT_TAU = 0.4

K_GRID = [1.5, 2.0, 2.5, 3.0]
TAU_GRID = [0.3, 0.4, 0.5, 0.6]

RANDOM_SEED = 42
