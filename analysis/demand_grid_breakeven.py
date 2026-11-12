import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
DEMAND_INDEPENDENT = ('P1_FIXED_SMALL', 'P2_FIXED_LARGE', 'P3_PERFORMANCE_FIRST', 'P4_RANDOM_ROUTER', 'P5_COST_HEURISTIC', 'P7_SYSLMMATIC_REPLICA')

def close(a, b, rel=1e-09):
    return abs(a - b) <= rel * max(abs(a), abs(b))

def crossings(points):
    out = []
    for (d0, v0), (d1, v1) in zip(points, points[1:]):
        if v0 <= 0 < v1 or v0 > 0 >= v1:
            out.append((d0 + (0 - v0) * (d1 - d0) / (v1 - v0), 'up' if v1 > 0 else 'down'))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', nargs='+', required=True)
    ap.add_argument('--primary', required=True)
    ap.add_argument('--human-cost', required=True)
    ap.add_argument('--k', nargs='+', type=float, default=[0, 10, 50])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows, meta = ([], [])
    for g in a.grid:
        d = json.loads(Path(g).read_text())
        meta.append({k: v for k, v in d.items() if k != 'rows'} | {'file': g})
        rows += d['rows']
    assert all((m['permutations'] == 200 and m['budget_eur'] == 0.2 for m in meta)), meta
    by_pol = defaultdict(list)
    for r in rows:
        by_pol[r['policy_id']].append(r)
    for p in by_pol:
        by_pol[p].sort(key=lambda r: r['demand_per_year'])
    demands = sorted({r['demand_per_year'] for r in rows})
    assert all((len(v) == len(demands) for v in by_pol.values())), 'grid incomplete'
    assert all((r['n'] == 10000 for r in rows)), 'expected 200 permutations x 50 opportunities'

    def at(policy, demand):
        hits = [r for r in by_pol.get(policy, ()) if close(r['demand_per_year'], demand)]
        return hits[0] if hits else None
    val_npv, max_npv_diff = ([], 0.0)
    with open(a.primary) as fh:
        for r in csv.DictReader(fh):
            if r['split'] != 'holdout' or float(r['budget_eur']) != 0.2:
                continue
            g = at(r['policy_id'], float(r['demand_per_year']))
            if g is None:
                continue
            diff = abs(g['mean_npv_eur'] - float(r['mean_npv_eur']))
            max_npv_diff = max(max_npv_diff, diff)
            val_npv.append({'policy_id': r['policy_id'], 'demand_per_year': float(r['demand_per_year']), 'grid': g['mean_npv_eur'], 'primary': float(r['mean_npv_eur']), 'abs_diff': diff})
    hc = json.loads(Path(a.human_cost).read_text())['rows']
    max_adopt_diff, n_adopt = (0.0, 0)
    for r in hc:
        if r['K_eur'] != 0:
            continue
        g = at(r['policy_id'], r['demand_per_year'])
        if g is None:
            continue
        n_adopt += 1
        max_adopt_diff = max(max_adopt_diff, abs(g['adoption_rate'] * r['n_opportunities'] - r['mean_adopted_per_permutation']))
    affinity = {}
    for p in DEMAND_INDEPENDENT:
        pts = [(r['demand_per_year'], r['mean_npv_eur']) for r in by_pol[p]]
        (d0, v0), (d1, v1) = (pts[0], pts[-1])
        slope = (v1 - v0) / (d1 - d0)
        res = max((abs(v - (v0 + slope * (d - d0))) for d, v in pts))
        adopt = {round(r['adoption_rate'], 12) for r in by_pol[p]}
        cens = {round(r['censored_rate'], 12) for r in by_pol[p]}
        affinity[p] = {'max_abs_residual_eur': res, 'intercept_eur': v0 - slope * d0, 'slope_eur_per_exec': slope, 'adoption_constant': len(adopt) == 1, 'censoring_constant': len(cens) == 1}
    breakeven = []
    for p, rs in sorted(by_pol.items()):
        if p == 'P0_STOP':
            continue
        for k in a.k:
            pts = [(r['demand_per_year'], r['mean_npv_eur'] - k * r['adoption_rate']) for r in rs]
            cr = crossings(pts)
            ups = [d for d, s in cr if s == 'up']
            if pts[0][1] > 0:
                first = None
                status = 'positive at the lowest grid demand'
            elif not ups:
                first = None
                status = 'not positive on the grid'
            else:
                first = ups[0]
                status = 'crossing'
            breakeven.append({'policy_id': p, 'K_eur': k, 'status': status, 'breakeven_demand': first, 'n_sign_changes': len(cr), 'all_crossings': cr, 'exact_interpolation': p in DEMAND_INDEPENDENT, 'npv_at_grid_min': pts[0][1], 'npv_at_grid_max': pts[-1][1]})
    decades = [10.0 ** e for e in range(3, 8)]
    censoring = {p: {f'{d:.0e}': at(p, d)['censored_rate'] for d in decades} | {'min_over_grid': min((r['censored_rate'] for r in rs)), 'max_over_grid': max((r['censored_rate'] for r in rs))} for p, rs in sorted(by_pol.items())}
    adoption = {p: {f'{d:.0e}': at(p, d)['adoption_rate'] for d in decades} for p in sorted(by_pol)}
    out = {'grid_meta': meta, 'n_demands': len(demands), 'demand_min': demands[0], 'demand_max': demands[-1], 'validation': {'mean_npv_vs_primary': {'n': len(val_npv), 'max_abs_diff_eur': max_npv_diff}, 'adoption_vs_human_cost_8083': {'n': n_adopt, 'max_abs_diff_patches': max_adopt_diff}, 'affinity_demand_independent': affinity, 'details': val_npv}, 'breakeven': breakeven, 'censoring_rate': censoring, 'adoption_rate': adoption}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f'validation: npv vs primary n={len(val_npv)} max|diff|={max_npv_diff:.3g} EUR; adoption vs 8083 n={n_adopt} max|diff|={max_adopt_diff:.3g} patches')
    for p, v in affinity.items():
        print(f"  affine {p}: max residual {v['max_abs_residual_eur']:.3g} EUR, adoption const {v['adoption_constant']}, censoring const {v['censoring_constant']}")
    print('break-even demand (executions/year), first upward crossing:')
    for b in breakeven:
        be = '-' if b['breakeven_demand'] is None else f"{b['breakeven_demand']:,.0f}"
        print(f"  {b['policy_id']:<24} K={b['K_eur']:>4g}  {be:>12}  {b['status']:<36} sign changes={b['n_sign_changes']} exact={b['exact_interpolation']}")
    print('censoring rate by demand:')
    for p, c in censoring.items():
        print(f'  {p:<24} ' + '  '.join((f'{k}={v:.3f}' for k, v in c.items())))
    print('adoption rate by demand:')
    for p, c in adoption.items():
        print(f'  {p:<24} ' + '  '.join((f'{k}={v:.3f}' for k, v in c.items())))
if __name__ == '__main__':
    main()
