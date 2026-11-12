import argparse
import csv
import json
from collections import defaultdict
from caso.inference import hierarchical_bootstrap, holm, leave_one_repo_out
CONTRASTS = {'H1': ('P6_CASO_ADAPTIVE', 'P3_PERFORMANCE_FIRST', 'base', 'greater'), 'H3': ('P6_CASO_ADAPTIVE', 'P3_PERFORMANCE_FIRST', 10000.0, 'greater'), 'H1b': ('P6_CASO_ADAPTIVE', 'P8_COST_AWARE_BO_STOP', 'base', 'two-sided'), 'H1c': ('P6_CASO_ADAPTIVE', 'P5_COST_HEURISTIC', 'base', 'greater'), 'H5': ('P6_CASO_ADAPTIVE', 'P4_RANDOM_ROUTER', 'base', 'greater')}

def repo_of(task_id):
    owner, rest = task_id.split('__', 1)
    return f"{owner}/{rest.rsplit('-', 1)[0]}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--opportunity-means', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--price', default='median')
    ap.add_argument('--budget', type=float, default=0.2)
    ap.add_argument('--base-demand', type=float, default=10000000.0)
    ap.add_argument('--p6-policy', default='P6_CASO_ADAPTIVE', help='final P6 variant (protocol §4.3)')
    ap.add_argument('--reps', type=int, default=5000)
    ap.add_argument('--exclude', nargs='*', default=[], help='instances left out (sensitivity, e.g. OOM batch)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    npv = {}
    for r in csv.DictReader(open(a.opportunity_means)):
        if r['split'] == a.split and r['price'] == a.price and (float(r['budget_eur']) == a.budget) and (r['task_id'] not in a.exclude):
            npv[float(r['demand_per_year']), r['task_id'], r['policy_id']] = float(r['mean_npv_eur'])
    results = {}
    for name, (policy, ref, demand, sided) in CONTRASTS.items():
        policy = a.p6_policy if policy == 'P6_CASO_ADAPTIVE' else policy
        d = a.base_demand if demand == 'base' else demand
        diffs = defaultdict(list)
        for (dem, task, pol), value in sorted(npv.items()):
            if dem == d and pol == policy and ((dem, task, ref) in npv):
                diffs[repo_of(task)].append(value - npv[dem, task, ref])
        if not diffs:
            results[name] = {'error': f'no paired rows for {policy} vs {ref} at {d}'}
            continue
        boot = hierarchical_bootstrap(diffs, reps=a.reps)
        results[name] = {'policy': policy, 'reference': ref, 'demand_per_year': d, 'sided': sided, **boot, 'p': boot['p_one_sided_greater'] if sided == 'greater' else boot['p_two_sided'], 'leave_one_repo_out': leave_one_repo_out(diffs), 'per_repo_mean': {r: sum(v) / len(v) for r, v in diffs.items()}}
    secondary = {k: v['p'] for k, v in results.items() if k != 'H1' and 'p' in v}
    out = {'excluded_instances': a.exclude, 'split': a.split, 'status': 'DRY RUN (exploratory)' if a.split != 'holdout' else 'confirmatory', 'price': a.price, 'budget_eur': a.budget, 'base_demand': a.base_demand, 'H1_supported': bool('ci95' in results.get('H1', {}) and results['H1']['ci95'][0] > 0), 'holm_secondary': holm(secondary) if secondary else {}, 'contrasts': results}
    with open(a.out, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ('split', 'status', 'H1_supported', 'holm_secondary')}))
if __name__ == '__main__':
    main()
