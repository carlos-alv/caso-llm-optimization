import argparse
import csv
import json
import math
import statistics
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
from caso.decision import BetaPosterior, NormalPosterior, expected_improvement_eur
from caso.inference import hierarchical_bootstrap, leave_one_repo_out
from caso.policies import P5CostHeuristic, P6CasoVOI, PolicyDecision, _stop
CONF = {}

class P6Confirmed(P6CasoVOI):
    policy_id = 'P6C_CASO_CONFIRMED'

    def decide(self, state):
        scored = []
        for c in self._feasible(state):
            beta, norm = CONF[c.action.model_key]
            for o in state.history:
                if o.model_key == c.action.model_key:
                    beta = beta.updated(bool(o.improvement_confirmed))
                    if o.improvement_confirmed and o.speedup and (o.speedup > 0):
                        norm = norm.updated(math.log(o.speedup))
            gain, _ = expected_improvement_eur(state, beta.mean, norm, amdahl_share=c.action.hotspot_share, n_samples=4000, seed=state.step)
            scored.append((c, gain - c.expected_cost_eur))
        usable = [x for x in scored if x[0].feasible]
        if not usable:
            return _stop('no action within remaining budget')
        c, margin = max(usable, key=lambda x: x[1])
        if margin <= 0:
            return _stop('no action with positive expected marginal value')
        return PolicyDecision(c.action, 'max confirmed-improvement margin', [], margin)

class P5Full(P5CostHeuristic):
    policy_id = 'P5F_CASCADE_FULL'

    def decide(self, state):
        cands = self._feasible(state)
        usable = sorted((c for c in cands if c.feasible), key=lambda c: c.expected_cost_eur)
        if not usable:
            return _stop('budget exhausted', cands)
        failures = {}
        for o in state.history:
            failures[o.model_key] = failures.get(o.model_key, 0) + (not o.improvement_confirmed)
        for c in usable:
            if failures.get(c.action.model_key, 0) < self.k:
                same = [u for u in usable if u.action.model_key == c.action.model_key]
                return PolicyDecision(max(same, key=lambda u: u.action.hotspot_share).action, 'cascade, no satisficing', cands)
        return _stop('every model failed k times', cands)
_orig = pa.build_policies

def build_policies(cost_model, priors, seed=20261025, ablations=False):
    return _orig(cost_model, priors, seed=seed, ablations=ablations) + [P6Confirmed(cost_model, priors), P5Full(cost_model, priors)]
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
    pool = [x for x in attempts if x.split in ('dev', 'calibration')]
    conf_logs = [math.log(x.speedup) for x in pool if x.improvement_confirmed and x.speedup and (x.speedup > 0)]
    mu, sd = (statistics.mean(conf_logs), statistics.stdev(conf_logs))
    for m in SIZE:
        own = [x for x in pool if x.model_key == m]
        nconf = sum((x.improvement_confirmed for x in own))
        CONF[m] = (BetaPosterior(1 + nconf, 1 + len(own) - nconf), NormalPosterior(mu, sd ** 2, sd))
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
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for acc in ex.map(_aggregate, [job] * len(seeds), seeds):
            for k, v in acc.items():
                total[k] += v
    means = {k: v / len(seeds) for k, v in total.items()}
    with open(out / 'opportunity_means_holdout.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['task_id', 'policy_id', 'demand_per_year', 'budget_eur', 'mean_npv_eur'])
        for (t, p, d, b), v in sorted(means.items()):
            w.writerow([t, p, d, b, v])
    tasks = sorted({k[0] for k in means})
    contrasts = {}
    for d, tag in ((10000.0, '1e4'), (10000000.0, '1e7')):
        contrasts[f'P6C_minus_P6_{tag}'] = ('P6C_CASO_CONFIRMED', 'P6_CASO_ADAPTIVE', d)
        contrasts[f'P6C_minus_P2_{tag}'] = ('P6C_CASO_CONFIRMED', 'P2_FIXED_LARGE', d)
        contrasts[f'P2_minus_P6_{tag}'] = ('P2_FIXED_LARGE', 'P6_CASO_ADAPTIVE', d)
        contrasts[f'P6_minus_P5F_{tag}'] = ('P6_CASO_ADAPTIVE', 'P5F_CASCADE_FULL', d)
        contrasts[f'P5F_minus_P5_{tag}'] = ('P5F_CASCADE_FULL', 'P5_COST_HEURISTIC', d)
    res = {'note': 'exploratory post-hoc; holdout; median price; 200 permutations', 'confirmed_priors': {m: {'p_confirmed': CONF[m][0].mean} for m in CONF}, 'confirmed_log_speedup_pool': {'mean': mu, 'sd': sd, 'n': len(conf_logs)}, 'policy_means': {}, 'contrasts': {}}
    for b in budgets:
        for d in demands:
            for p in sorted({k[1] for k in means}):
                res['policy_means'][f'{p}|{d:g}|{b}'] = sum((means[t, p, d, b] for t in tasks)) / len(tasks)
        for name, (p, r, d) in contrasts.items():
            diffs = defaultdict(list)
            for t in tasks:
                diffs[repo_of(t)].append(means[t, p, d, b] - means[t, r, d, b])
            res['contrasts'][f'{name}|{b}'] = {'policy': p, 'reference': r, 'demand': d, 'budget': b, **hierarchical_bootstrap(diffs, reps=5000), 'leave_one_repo_out': leave_one_repo_out(diffs)}
    (out / 'posthoc_policies2.json').write_text(json.dumps(res, indent=1, default=str))
    print('done')
if __name__ == '__main__':
    main()
