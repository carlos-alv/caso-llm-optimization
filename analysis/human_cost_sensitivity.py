import argparse
import json
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pilot_analysis import SIZE, _run_seed, base_time_per_call
from caso.bank import load_bank
from caso.pilot_analysis import run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', nargs='+', required=True)
    ap.add_argument('--evaluations', nargs='+', required=True)
    ap.add_argument('--reports', required=True)
    ap.add_argument('--scenarios', required=True)
    ap.add_argument('--shares', required=True)
    ap.add_argument('--thresholds', required=True)
    ap.add_argument('--demand-exponents', nargs='+', type=int, default=[3, 4, 5, 6, 7])
    ap.add_argument('--k', nargs='+', type=float, default=[0, 10, 50])
    ap.add_argument('--permutations', type=int, default=200)
    ap.add_argument('--perm-seed', type=int, default=20261027)
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    attempts = load_bank(map(Path, a.candidates), map(Path, a.evaluations), reject_persistent_state=True, instance_thresholds=json.loads(Path(a.thresholds).read_text()))
    reports = {p.stem: json.loads(p.read_text()) for p in Path(a.reports).glob('*.json') if not p.name.startswith('_')}
    base = {iid: base_time_per_call(reports[iid]) for iid in {x.instance_id for x in attempts}}
    scen = json.loads(Path(a.scenarios).read_text())
    gpu = {s['scenario_id'].rsplit('-', 1)[-1]: s['rates']['gpu_hour_fully_loaded_eur']['value'] for s in scen}
    cpu = scen[0]['rates']['cpu_core_hour_fully_loaded_eur']['value']
    shares = json.loads(Path(a.shares).read_text())
    attempts = [replace(x, hotspot_share=shares[x.instance_id][x.hotspot_id]) if x.hotspot_id in shares.get(x.instance_id, {}) else x for x in attempts]
    present = {x.model_key for x in attempts}
    job = partial(run, attempts, base, {k: v for k, v in SIZE.items() if k in present}, {'median': gpu['median']}, cpu, [10.0 ** k for k in a.demand_exponents], [0.2], horizon=3, rate=0.0, ablations=False)
    seeds = [a.perm_seed + k for k in range(a.permutations)]
    acc = defaultdict(lambda: {'npv': 0.0, 'adopted': 0, 'n': 0})
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for chunk in pool.map(_run_seed, [job] * len(seeds), seeds):
            for r in chunk:
                if r['split'] != 'holdout':
                    continue
                key = (float(r['demand_per_year']), r['policy_id'])
                acc[key]['npv'] += float(r['npv_eur'])
                acc[key]['adopted'] += float(r['value_eur']) > 0
                acc[key]['n'] += 1
    out = []
    for (dem, pol), v in sorted(acc.items()):
        n_opp = v['n'] / len(seeds)
        for k in a.k:
            out.append({'demand_per_year': dem, 'policy_id': pol, 'K_eur': k, 'mean_npv_eur': (v['npv'] - k * v['adopted']) / v['n'], 'mean_adopted_per_permutation': v['adopted'] / len(seeds), 'n_opportunities': n_opp})
    Path(a.out).write_text(json.dumps({'permutations': a.permutations, 'perm_seed': a.perm_seed, 'note': 'ex post fixed cost per adopted patch; decisions unchanged', 'rows': out}, indent=1))
    print('done', len(out))
if __name__ == '__main__':
    main()
