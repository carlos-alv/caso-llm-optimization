import argparse
import csv
import json
import sys
from collections import defaultdict

def close(a, b, rel=1e-09, abs_=1e-12):
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replay', required=True)
    ap.add_argument('--primary', help='final-7482/primary/opportunity_means.csv')
    ap.add_argument('--posthoc', nargs='*', default=[], help='opportunity_means_holdout.csv of jobs 8143/8182')
    ap.add_argument('--grid', nargs='*', default=[], help='demand_grid.json files (50-opportunity means)')
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.replay)))
    got = {(r['task_id'], r['policy_id'], float(r['demand_per_year']), float(r['budget_eur'])): float(r['mean_npv_eur']) for r in rows}
    n_perm = {int(r['n_permutations']) for r in rows}
    assert n_perm == {200}, f'incomplete replay: {n_perm}'
    refs = {}
    if a.primary:
        refs['primary'] = {(r['task_id'], r['policy_id'], float(r['demand_per_year']), float(r['budget_eur'])): float(r['mean_npv_eur']) for r in csv.DictReader(open(a.primary)) if r['split'] == 'holdout' and r['price'] == 'median'}
    for f in a.posthoc:
        refs[f] = {(r['task_id'], r['policy_id'], float(r['demand_per_year']), float(r['budget_eur'])): float(r['mean_npv_eur']) for r in csv.DictReader(open(f))}
    bad = 0
    for name, ref in refs.items():
        n = 0
        for k, v in got.items():
            if k in ref:
                n += 1
                if not close(v, ref[k]):
                    bad += 1
                    print('MISMATCH', name, k, v, ref[k])
        print(f'{name}: {n} rows compared')
    if a.grid:
        tasks = {k[0] for k in got}
        if len(tasks) == 50:
            means = defaultdict(list)
            for (t, p, d, b), v in got.items():
                if b == 0.2:
                    means[p, d].append(v)
            n = 0
            for g in a.grid:
                for r in json.load(open(g))['rows']:
                    key = (r['policy_id'], r['demand_per_year'])
                    match = [m for (p, d), m in means.items() if p == key[0] and close(d, key[1], rel=1e-12)]
                    if match:
                        n += 1
                        mean = sum(match[0]) / len(match[0])
                        if not close(mean, r['mean_npv_eur']):
                            bad += 1
                            print('MISMATCH grid', key, mean, r['mean_npv_eur'])
            print(f'grid: {n} rows compared')
    if bad:
        sys.exit(f'{bad} mismatches')
    print('validation ok')
if __name__ == '__main__':
    main()
