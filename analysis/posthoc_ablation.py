import argparse
import csv
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
from caso.decision import expected_improvement_eur, posterior_for
from caso.inference import hierarchical_bootstrap, leave_one_repo_out
from caso.policies import Candidate, P3PerformanceFirst, P6CasoVOI, PolicyDecision, _stop

def _caso_scored(policy, state):
    scored = []
    for c in policy._feasible(state):
        validity, log_speedup = posterior_for(policy.priors[c.action.model_key], state.history, c.action.model_key)
        gain, _ = expected_improvement_eur(state, validity.mean, log_speedup, amdahl_share=c.action.hotspot_share, n_samples=4000, seed=state.step)
        scored.append((c, validity, log_speedup, gain, gain - c.expected_cost_eur))
    return scored

class P6NoStop(P6CasoVOI):
    policy_id = 'P6N_CASO_NO_STOP'

    def decide(self, state):
        usable = [x for x in _caso_scored(self, state) if x[0].feasible]
        if not usable:
            return _stop('budget exhausted')
        c, v, _, gain, margin = max(usable, key=lambda x: x[4])
        return PolicyDecision(c.action, 'max margin, no economic stop', [], margin)

class P3WithStop(P3PerformanceFirst):
    policy_id = 'P3S_PERF_WITH_STOP'

    def decide(self, state):
        usable = [x for x in _caso_scored(self, state) if x[0].feasible]
        if not usable or max((x[4] for x in usable)) <= 0:
            return _stop('no action with positive expected marginal value')
        c = max(usable, key=lambda x: (x[1].mean * x[2].mean, x[0].action.hotspot_share))[0]
        return PolicyDecision(c.action, 'highest expected speedup, CASO stop', [], None)
_orig = pa.build_policies

def build_policies(cost_model, priors, seed=20261025, ablations=False):
    return _orig(cost_model, priors, seed=seed, ablations=ablations) + [P6NoStop(cost_model, priors), P3WithStop(cost_model, priors)]
pa.build_policies = build_policies

def _aggregate(job, seed):
    acc = defaultdict(float)
    for r in job(order_seed=seed):
        if r['split'] == 'holdout':
            acc[r['task_id'], r['policy_id'], float(r['demand_per_year']), float(r['budget_eur'])] += float(r['npv_eur'])
    return dict(acc)

def repo_of(task_id):
    owner, rest = task_id.split('__', 1)
    return f"{owner}/{rest.rsplit('-', 1)[0]}"

def main():
    ap = argparse.ArgumentParser()
    for k in ('candidates', 'evaluations'):
        ap.add_argument(f'--{k}', nargs='+', required=True)
    for k in ('reports', 'scenarios', 'shares', 'thresholds', 'out'):
        ap.add_argument(f'--{k}', required=True)
    ap.add_argument('--permutations', type=int, default=200)
    ap.add_argument('--perm-seed', type=int, default=20261027)
    ap.add_argument('--workers', type=int, default=14)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    attempts = load_bank(map(Path, a.candidates), map(Path, a.evaluations), reject_persistent_state=True, instance_thresholds=json.loads(Path(a.thresholds).read_text()))
    reports = {p.stem: json.loads(p.read_text()) for p in Path(a.reports).glob('*.json') if not p.name.startswith('_')}
    base = {iid: base_time_per_call(reports[iid]) for iid in {x.instance_id for x in attempts}}
    scen = json.loads(Path(a.scenarios).read_text())
    gpu = {s['scenario_id'].rsplit('-', 1)[-1]: s['rates']['gpu_hour_fully_loaded_eur']['value'] for s in scen}
    cpu = scen[0]['rates']['cpu_core_hour_fully_loaded_eur']['value']
    shares = json.loads(Path(a.shares).read_text())
    attempts = [replace(x, hotspot_share=shares[x.instance_id][x.hotspot_id]) if x.hotspot_id in shares.get(x.instance_id, {}) else x for x in attempts]
    present = {x.model_key for x in attempts}
    demands, budgets = ([10000.0, 10000000.0], [0.02, 0.05, 0.2])
    job = partial(pa.run, attempts, base, {k: v for k, v in SIZE.items() if k in present}, {'median': gpu['median']}, cpu, demands, budgets, horizon=3, rate=0.0, ablations=False)
    seeds = [a.perm_seed + k for k in range(a.permutations)]
    total = defaultdict(float)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for acc in pool.map(_aggregate, [job] * len(seeds), seeds):
            for k, v in acc.items():
                total[k] += v
    means = {k: v / len(seeds) for k, v in total.items()}
    with open(out / 'opportunity_means_holdout.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['task_id', 'policy_id', 'demand_per_year', 'budget_eur', 'mean_npv_eur'])
        for (t, p, d, b), v in sorted(means.items()):
            w.writerow([t, p, d, b, v])
    tasks = sorted({k[0] for k in means})
    contrasts = {'H1': ('P6_CASO_ADAPTIVE', 'P3_PERFORMANCE_FIRST', 10000000.0), 'H3': ('P6_CASO_ADAPTIVE', 'P3_PERFORMANCE_FIRST', 10000.0), 'H1b': ('P6_CASO_ADAPTIVE', 'P8_COST_AWARE_BO_STOP', 10000000.0), 'H1c': ('P6_CASO_ADAPTIVE', 'P5_COST_HEURISTIC', 10000000.0), 'H5': ('P6_CASO_ADAPTIVE', 'P4_RANDOM_ROUTER', 10000000.0)}
    for d in (10000.0, 10000000.0):
        tag = '1e4' if d == 10000.0 else '1e7'
        contrasts[f'STOP_in_CASO_{tag}'] = ('P6_CASO_ADAPTIVE', 'P6N_CASO_NO_STOP', d)
        contrasts[f'STOP_in_P3_{tag}'] = ('P3S_PERF_WITH_STOP', 'P3_PERFORMANCE_FIRST', d)
        contrasts[f'SELECTION_given_STOP_{tag}'] = ('P6_CASO_ADAPTIVE', 'P3S_PERF_WITH_STOP', d)
    res = {'note': 'exploratory post-hoc; holdout; median price; 200 permutations', 'policy_means': {}, 'contrasts': {}}
    for b in budgets:
        for d in demands:
            for p in sorted({k[1] for k in means}):
                res['policy_means'][f'{p}|{d:g}|{b}'] = sum((means[t, p, d, b] for t in tasks)) / len(tasks)
        for name, (p, r, d) in contrasts.items():
            diffs = defaultdict(list)
            for t in tasks:
                diffs[repo_of(t)].append(means[t, p, d, b] - means[t, r, d, b])
            hb = hierarchical_bootstrap(diffs, reps=5000)
            res['contrasts'][f'{name}|{b}'] = {'policy': p, 'reference': r, 'demand': d, 'budget': b, **hb, 'leave_one_repo_out': leave_one_repo_out(diffs)}
    (out / 'posthoc_ablation.json').write_text(json.dumps(res, indent=1, default=str))
    print('done')
if __name__ == '__main__':
    main()
