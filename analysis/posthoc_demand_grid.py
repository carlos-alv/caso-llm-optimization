import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pilot_analysis import SIZE, base_time_per_call
import caso.pilot_analysis as pa
from caso.bank import load_bank

def run_holdout(attempts, base_time_per_call_s, size_class, gpu_price, cpu_price, demands, budget, horizon, order_seed):
    acc = defaultdict(lambda: [0.0, 0, 0, 0])
    pool = [a for a in attempts if a.split in ('dev', 'calibration')]
    priors = pa.priors_from(pool, size_class)
    prices = pa.Prices(gpu_price, cpu_price)
    cost_model = pa.CostModel(priors, prices)
    for iid in sorted({a.instance_id for a in attempts if a.split == 'holdout'}):
        bank = pa.bank_for(attempts, iid, prices, order_seed)
        c0 = base_time_per_call_s[iid] / 3600 * cpu_price
        for demand in demands:
            multiplier = demand * pa.annuity(0.0, horizon)
            state = pa.OpportunityState(iid, c0, multiplier, pa.actions_for(attempts, iid), (), budget)
            for policy in pa.build_policies(cost_model, priors, ablations=False):
                r = pa.replay(policy, state, bank)
                a = acc[policy.policy_id, demand]
                a[0] += r.npv_eur
                a[1] += r.value_eur > 0
                a[2] += bool(r.censored)
                a[3] += 1
    return dict(acc)

def main():
    ap = argparse.ArgumentParser()
    for k in ('candidates', 'evaluations'):
        ap.add_argument(f'--{k}', nargs='+', required=True)
    for k in ('reports', 'scenarios', 'shares', 'thresholds', 'out'):
        ap.add_argument(f'--{k}', required=True)
    ap.add_argument('--kmin', type=int, required=True)
    ap.add_argument('--kmax', type=int, required=True)
    ap.add_argument('--permutations', type=int, default=200)
    ap.add_argument('--perm-seed', type=int, default=20261027)
    ap.add_argument('--workers', type=int, default=28)
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
    demands = [10.0 ** (3 + k / 8) for k in range(a.kmin, a.kmax + 1)]
    job = partial(run_holdout, attempts, base, {k: v for k, v in SIZE.items() if k in present}, gpu['median'], cpu, demands, 0.2, 3)
    seeds = [a.perm_seed + k for k in range(a.permutations)]
    total = defaultdict(lambda: [0.0, 0, 0, 0])
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for acc in ex.map(job, seeds):
            for k, v in acc.items():
                t = total[k]
                for i in range(4):
                    t[i] += v[i]
    rows = [{'policy_id': p, 'demand_per_year': d, 'mean_npv_eur': v[0] / v[3], 'adoption_rate': v[1] / v[3], 'censored_rate': v[2] / v[3], 'n': v[3]} for (p, d), v in sorted(total.items())]
    Path(a.out).write_text(json.dumps({'permutations': a.permutations, 'budget_eur': 0.2, 'kmin': a.kmin, 'kmax': a.kmax, 'rows': rows}, indent=1))
    print('done', len(rows))
if __name__ == '__main__':
    main()
