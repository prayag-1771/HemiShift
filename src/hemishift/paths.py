"""Project paths and device selection. Every other module takes its locations from here."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("HEMISHIFT_DATA", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
HARMONISED_DIR = DATA_DIR / "harmonised"
RESULTS_DIR = ROOT / "results"
CONFIG_DIR = ROOT / "configs"


def device():
    """Torch device chosen by HEMISHIFT_DEVICE=cpu|cuda; defaults to CUDA when available.

    Pin one device per experiment: results computed on different devices are not compared.
    """
    import torch

    wanted = os.environ.get("HEMISHIFT_DEVICE", "").lower()
    if wanted == "cpu":
        return torch.device("cpu")
    if wanted == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "HEMISHIFT_DEVICE=cuda but CUDA is unavailable; is a +cpu torch build installed?"
        )
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
