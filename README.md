# CASO

Replays cost-aware optimization policies on a measured bank of LLM-generated patches and computes the reported contrasts.

Requirements: Python >= 3.10, numpy, scipy.

Data: the attempt bank and archived results (Zenodo archive, unpacked as DATA_DIR).

```
./reproduce.sh DATA_DIR [WORKERS]
```

Recomputes the confirmatory contrasts and replays two policies on three opportunities over 200 permutations; fails unless both match the archived results exactly. `analysis/` holds the scripts for every other analysis; run them with `PYTHONPATH=src:analysis`.
