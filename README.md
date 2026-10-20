# HemiShift

Decoding a stroke patient's intent to move from EEG, for brain-computer-interface rehabilitation.
After a stroke, the "I'm trying to move" signal weakens and shifts toward the healthy hemisphere.
HemiShift teaches decoders that shift by turning healthy EEG into stroke-like EEG (LesionSim),
then tests them on hospitals they have never seen.

- Plan, phases and gates: [PLAN.md](PLAN.md)
- Prior work and novelty evidence: [LITERATURE.md](LITERATURE.md)

## Setup (Windows, Python 3.11, NVIDIA GPU)

```bash
python -m venv .venv
# torch and torchaudio first, from the CUDA index. A plain `pip install torch` gives a CPU-only
# build, and braindecode would otherwise pull a PyPI torchaudio that can't load with this torch.
.venv/Scripts/python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --retries 10 --timeout 120
.venv/Scripts/python -m pip install -r requirements.txt
# exact versions of the tested environment: requirements.lock.txt
.venv/Scripts/python -m pip install -e .
.venv/Scripts/python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

`HEMISHIFT_DEVICE=cpu|cuda` pins the device. `HEMISHIFT_DATA` moves the data folder (default `data/`).

## Data

```bash
.venv/Scripts/python -I scripts/download_all.py              # all non-deferred cohorts (~22 GB)
.venv/Scripts/python -I scripts/download_all.py liu2025      # deferred cohort, 21.8 GB
```

Cohorts, licences and sizes are in `src/hemishift/data/registry.py`. MOABB downloads go to
`data/raw/moabb`. This also sets MNE's global `MNE_DATA` config on this machine.
