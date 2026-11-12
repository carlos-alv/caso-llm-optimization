#!/usr/bin/env bash
set -euo pipefail
DATA="$(cd "${1:?usage: reproduce.sh DATA_DIR [WORKERS]}" && pwd)"
W="${2:-3}"
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:$PWD/analysis" PYTHONDONTWRITEBYTECODE=1
out="$(mktemp -d)"
python3 analysis/confirmatory_analysis.py --opportunity-means "$DATA/results/final-7482/primary/opportunity_means.csv" \
  --split holdout --out "$out/contrasts_holdout.json" > /dev/null
python3 - "$out/contrasts_holdout.json" "$DATA/results/final-7482/primary/contrasts_holdout.json" <<'PY'
import json, sys
new, ref = (json.load(open(p)) for p in sys.argv[1:3])
def walk(a, b):
    if isinstance(a, dict):
        for k in a:
            if k in b: yield from walk(a[k], b[k])
    elif isinstance(a, list):
        for x, y in zip(a, b): yield from walk(x, y)
    elif isinstance(a, float) and isinstance(b, float):
        yield abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
r = list(walk(new, ref))
print(f"contrasts: {sum(r)}/{len(r)} identical")
sys.exit(0 if r and all(r) else 1)
PY
cands=$(sed "s|^|$DATA/|" "$DATA/data/candidates/ORDER.txt" | tr "\n" " ")
evals=$(sed "s|^|$DATA/|" "$DATA/data/evaluations/ORDER.txt" | tr "\n" " ")
python3 analysis/replay_variants.py --candidates $cands --evaluations $evals --reports "$DATA/data/reports" \
  --scenarios configs/cost_scenarios_pilot.json --shares "$DATA/data/fork_shares.json" \
  --thresholds "$DATA/data/controls/thresholds.json" --variant primary \
  --instances scipy__scipy-13566 sympy__sympy-16134 scipy__scipy-10467 \
  --policies P2_FIXED_LARGE P6_CASO_ADAPTIVE --demands 1e4 1e7 --budgets 0.2 --workers "$W" --out "$out/replay" > /dev/null
python3 analysis/validate_replay.py --replay "$out/replay/opportunity_means.csv" \
  --primary "$DATA/results/final-7482/primary/opportunity_means.csv"
echo "REPRODUCTION CHECK PASSED"
