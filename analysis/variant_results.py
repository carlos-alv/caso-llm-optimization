import argparse
import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
BASE9 = ('P0_STOP', 'P1_FIXED_SMALL', 'P2_FIXED_LARGE', 'P3_PERFORMANCE_FIRST', 'P4_RANDOM_ROUTER', 'P5_COST_HEURISTIC', 'P6_CASO_ADAPTIVE', 'P7_SYSLMMATIC_REPLICA', 'P8_COST_AWARE_BO_STOP')
SPEND = ('P1_FIXED_SMALL', 'P2_FIXED_LARGE', 'P3_PERFORMANCE_FIRST', 'P4_RANDOM_ROUTER', 'P5_COST_HEURISTIC', 'P6_CASO_ADAPTIVE', 'P7_SYSLMMATIC_REPLICA', 'P8_COST_AWARE_BO_STOP', 'P2S_LARGE_WITH_STOP')
DEMAND_INDEPENDENT = ('P1_FIXED_SMALL', 'P2_FIXED_LARGE', 'P3_PERFORMANCE_FIRST', 'P4_RANDOM_ROUTER', 'P5_COST_HEURISTIC', 'P7_SYSLMMATIC_REPLICA')
CONTRASTS = {'H1': ('P6_CASO_ADAPTIVE', 'P3_PERFORMANCE_FIRST', 32), 'H3': ('P6_CASO_ADAPTIVE', 'P3_PERFORMANCE_FIRST', 8), 'H1b': ('P6_CASO_ADAPTIVE', 'P8_COST_AWARE_BO_STOP', 32), 'H1c': ('P6_CASO_ADAPTIVE', 'P5_COST_HEURISTIC', 32), 'H5': ('P6_CASO_ADAPTIVE', 'P4_RANDOM_ROUTER', 32), 'P2-P6': ('P2_FIXED_LARGE', 'P6_CASO_ADAPTIVE', 32)}
S3_OPPS = ('scipy__scipy-18799', 'scipy__scipy-11982')
S4_OPPS = S3_OPPS + ('scipy__scipy-10921', 'scikit-learn__scikit-learn-19606', 'sympy__sympy-10919')

def K(d):
    k = (math.log10(float(d)) - 3) * 8
    assert abs(k - round(k)) < 1e-06, d
    return int(round(k))

def D(k):
    return 10.0 ** (3 + k / 8)

def close(a, b, rel=1e-09, abs_=1e-12):
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))

def repo_of(task_id):
    owner, rest = task_id.split('__', 1)
    return f"{owner}/{rest.rsplit('-', 1)[0]}"

def load_pub(path, holdout_filter=True):
    out = {}
    for r in csv.DictReader(open(path)):
        if holdout_filter and 'split' in r and (r['split'] != 'holdout' or r['price'] != 'median'):
            continue
        out[r['task_id'], r['policy_id'], K(r['demand_per_year']), float(r['budget_eur'])] = float(r['mean_npv_eur'])
    return out

def load_new(path):
    out = {}
    for r in csv.DictReader(open(path)):
        assert int(r['n_permutations']) == 200, (path, r['n_permutations'])
        out[r['task_id'], r['policy_id'], K(r['demand_per_year']), float(r['budget_eur'])] = {k: float(r[k]) for k in ('mean_npv_eur', 'mean_cost_eur', 'adoption_rate', 'censored_rate', 'mean_steps', 'share_over_budget', 'max_overshoot_eur')}
    return out

def npv_of(new):
    return {k: v['mean_npv_eur'] for k, v in new.items()}

def check(name, got, ref, keys=None):
    n = 0
    for k, v in got.items():
        if (keys is None or k in keys) and k in ref:
            n += 1
            assert close(v, ref[k]), (name, k, v, ref[k])
    assert n > 0, f'{name}: nothing compared'
    print(f'validated {name}: {n} rows')

def splice(base, new, opps):
    out = {k: v for k, v in base.items() if k[0] not in opps}
    out.update({k: v for k, v in new.items() if k[0] in opps})
    return out

def crossing(points):
    for (d0, v0), (d1, v1) in zip(points, points[1:]):
        if v0 <= 0 < v1:
            return d0 + (0 - v0) * (d1 - d0) / (v1 - v0)
    return None if points[-1][1] <= 0 else points[0][0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--caso-src', required=True)
    ap.add_argument('--fase3', required=True)
    ap.add_argument('--summary', required=True, help='posthoc_summary.py output')
    ap.add_argument('--runs', required=True, help='JSON mapping run names to opportunity_means.csv paths')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.caso_src)
    from caso.inference import hierarchical_bootstrap
    f3 = Path(a.fase3)
    runs = {k: load_new(v) for k, v in json.loads(Path(a.runs).read_text()).items()}
    prim = load_pub(f3 / 'final-7482/primary/opportunity_means.csv')
    abl = load_pub(f3 / 'posthoc-abl-8143/opportunity_means_holdout.csv', holdout_filter=False)
    pol2 = load_pub(f3 / 'posthoc-pol2-8182/opportunity_means_holdout.csv', holdout_filter=False)
    s1 = load_pub(f3 / 'review-S1-8047/S1/opportunity_means.csv')
    s2 = load_pub(f3 / 'review-S2-8048/S2/opportunity_means.csv')
    tasks = sorted({k[0] for k in prim})
    assert len(tasks) == 50
    grid = {}
    for j in (8183, 8184):
        for r in json.load(open(f3 / f'demand-grid-{j}/demand_grid.json'))['rows']:
            grid[r['policy_id'], K(r['demand_per_year'])] = (r['mean_npv_eur'], r['adoption_rate'])
    r10 = json.loads(Path(a.summary).read_text())
    F_setup = r10['B_setup_cost']['mean']
    res = {'F_setup': F_setup}
    for k, v in s1.items():
        if k[0] not in S3_OPPS and k in prim:
            assert close(v, prim[k]), ('S1 differs from primary outside S3_OPPS', k)
    print('validated S1 == primary outside', S3_OPPS)
    check('audit primary grid vs primary analysis', npv_of(runs['audit_primary_grid']), prim)
    for name in ('audit_S3_grid', 'audit_S3_decades'):
        check(f'{name} on scipy-11982 vs S1', npv_of(runs[name]), s1, keys={k for k in runs[name] if k[0] == 'scipy__scipy-11982'})
    check('audit S4 on scipy-11982 vs S2', npv_of(runs['audit_S4_decades']), s2, keys={k for k in runs['audit_S4_decades'] if k[0] == 'scipy__scipy-11982'})
    if 'spend' in runs:
        check('spend run vs ablation job (budgets 0.02/0.05)', npv_of(runs['spend']), abl)
    p2s = npv_of(runs['p2s_grid'])
    for k in sorted({k[2] for k in p2s}):
        m = st.mean((p2s[t, 'P2_FIXED_LARGE', k, 0.2] for t in tasks))
        assert close(m, grid['P2_FIXED_LARGE', k][0]), (k, m)
    print('validated P2 on the P2S grid vs the fine grid')

    def means(npv, policies, ks, budget=0.2):
        return {(p, k): st.mean((npv[t, p, k, budget] for t in tasks)) for p in policies for k in ks if all(((t, p, k, budget) in npv for t in tasks))}

    def contrasts(npv, spec=CONTRASTS, budget=0.2):
        out = {}
        for n, (p, ref, k) in spec.items():
            if not all(((t, p, k, budget) in npv and (t, ref, k, budget) in npv for t in tasks)):
                continue
            diffs = defaultdict(list)
            for t in tasks:
                diffs[repo_of(t)].append(npv[t, p, k, budget] - npv[t, ref, k, budget])
            hb = hierarchical_bootstrap(diffs, reps=5000)
            out[n] = {'estimate': hb['estimate'], 'ci95': hb['ci95'], 'p_one_sided_greater': hb['p_one_sided_greater'], 'p_two_sided': hb['p_two_sided']}
        return out

    def breakevens(npv_k, adopt_k, policies, F=0.0, Ks=(0, 10, 50)):
        out = {}
        for Kh in Ks:
            for p in policies:
                pts = sorted(((D(k), npv_k[p, k] - F - Kh * adopt_k.get((p, k), 0.0)) for pp, k in npv_k if pp == p))
                out[f'{p}|K={Kh}'] = crossing(pts)
        return out
    s3_dec = splice(s1, {**npv_of(runs['audit_S3_grid']), **npv_of(runs['audit_S3_decades'])}, S3_OPPS)
    s3_dec = {k: v for k, v in s3_dec.items() if k[3] == 0.2}
    res['S3_means'] = {f'{p}|{k}': v for (p, k), v in means(s3_dec, BASE9, (0, 8, 16, 24, 32)).items()}
    res['S3_contrasts'] = contrasts(s3_dec)
    pg, sg = (runs['audit_primary_grid'], runs['audit_S3_grid'])
    s3_grid, s3_adopt, prim_grid, prim_adopt = ({}, {}, {}, {})
    for (p, k), (m, ad) in grid.items():
        prim_grid[p, k], prim_adopt[p, k] = (m, ad)
        if all(((t, p, k, 0.2) in pg and (t, p, k, 0.2) in sg for t in S3_OPPS)):
            s3_grid[p, k] = m + sum((sg[t, p, k, 0.2]['mean_npv_eur'] - pg[t, p, k, 0.2]['mean_npv_eur'] for t in S3_OPPS)) / 50
            s3_adopt[p, k] = ad + sum((sg[t, p, k, 0.2]['adoption_rate'] - pg[t, p, k, 0.2]['adoption_rate'] for t in S3_OPPS)) / 50
    for k in (0, 8, 16, 24, 32):
        for p in BASE9:
            assert close(s3_grid[p, k], res['S3_means'][f'{p}|{k}'], rel=1e-09, abs_=1e-09), (p, k)
    print('validated S3 grid splice vs S3 decade means')
    pols = [p for p in BASE9 if p != 'P0_STOP']
    res['primary_breakeven'] = breakevens(prim_grid, prim_adopt, pols)
    res['primary_breakeven_setup'] = breakevens(prim_grid, prim_adopt, pols, F=F_setup, Ks=(0,))
    res['S3_breakeven'] = breakevens(s3_grid, s3_adopt, pols)
    res['S3_breakeven_setup'] = breakevens(s3_grid, s3_adopt, pols, F=F_setup, Ks=(0,))
    s3_new = npv_of(runs['audit_S3_decades'])
    s3_post = splice({**abl, **pol2}, s3_new, S3_OPPS)
    spec = {'STOP_in_CASO_1e4': ('P6_CASO_ADAPTIVE', 'P6N_CASO_NO_STOP', 8), 'STOP_in_P3_1e4': ('P3S_PERF_WITH_STOP', 'P3_PERFORMANCE_FIRST', 8), 'SEL_given_STOP_1e7': ('P6_CASO_ADAPTIVE', 'P3S_PERF_WITH_STOP', 32), 'P6C-P6_1e7': ('P6C_CASO_CONFIRMED', 'P6_CASO_ADAPTIVE', 32), 'P2-P6_1e7': ('P2_FIXED_LARGE', 'P6_CASO_ADAPTIVE', 32), 'H1c_1e7': ('P6_CASO_ADAPTIVE', 'P5_COST_HEURISTIC', 32), 'H3': ('P6_CASO_ADAPTIVE', 'P3_PERFORMANCE_FIRST', 8)}
    res['S3_posthoc'] = {b: contrasts(s3_post, spec, budget=b) for b in (0.02, 0.05, 0.2)}
    res['S3_posthoc_means'] = {f'{p}|{k}|{b}': v for b in (0.02, 0.05, 0.2) for (p, k), v in means(s3_post, sorted({x[1] for x in s3_post}), (8, 32), budget=b).items()}
    s4 = splice({k: v for k, v in s2.items() if k[3] == 0.2}, npv_of(runs['audit_S4_decades']), S4_OPPS)
    res['S4_means'] = {f'{p}|{k}': v for (p, k), v in means(s4, BASE9, (0, 8, 16, 24, 32)).items()}
    res['S4_contrasts'] = contrasts(s4)
    p2s_all = runs['p2s_grid']
    pg_npv = {(p, k): st.mean((p2s_all[t, p, k, 0.2]['mean_npv_eur'] for t in tasks)) for t, p, k, b in p2s_all if t == tasks[0]}
    pg_ad = {(p, k): st.mean((p2s_all[t, p, k, 0.2]['adoption_rate'] for t in tasks)) for p, k in pg_npv}
    res['P2S_grid_means'] = {f'{p}|{k}': v for (p, k), v in pg_npv.items()}
    res['P2S_breakeven'] = breakevens(pg_npv, pg_ad, ('P2S_LARGE_WITH_STOP',))
    res['P2S_breakeven_setup'] = breakevens(pg_npv, pg_ad, ('P2S_LARGE_WITH_STOP',), F=F_setup, Ks=(0,))
    full = {**prim, **{k: v['mean_npv_eur'] for k, v in p2s_all.items()}, **abl}
    spec2 = {'P2S-P2_1e4': ('P2S_LARGE_WITH_STOP', 'P2_FIXED_LARGE', 8), 'P2S-P2_1e7': ('P2S_LARGE_WITH_STOP', 'P2_FIXED_LARGE', 32), 'P2S-P6_1e4': ('P2S_LARGE_WITH_STOP', 'P6_CASO_ADAPTIVE', 8), 'P2S-P6_1e7': ('P2S_LARGE_WITH_STOP', 'P6_CASO_ADAPTIVE', 32), 'P2S-P8_1e7': ('P2S_LARGE_WITH_STOP', 'P8_COST_AWARE_BO_STOP', 32), 'P2S-P3S_1e4': ('P2S_LARGE_WITH_STOP', 'P3S_PERF_WITH_STOP', 8), 'P2S-P3S_1e7': ('P2S_LARGE_WITH_STOP', 'P3S_PERF_WITH_STOP', 32), 'P2S-P7_1e4': ('P2S_LARGE_WITH_STOP', 'P7_SYSLMMATIC_REPLICA', 8)}
    res['P2S_contrasts_primary'] = contrasts(full, spec2)
    if 'spend' in runs:
        full_small = {**abl, **npv_of(runs['spend'])}
        res['P2S_contrasts_small'] = {b: contrasts(full_small, spec2, budget=b) for b in (0.02, 0.05)}
    s3p = {k: v['mean_npv_eur'] for k, v in runs['audit_S3_p2s_grid'].items()}
    p2s_s3 = splice({k: v['mean_npv_eur'] for k, v in p2s_all.items() if k[1] == 'P2S_LARGE_WITH_STOP'}, s3p, S3_OPPS)
    ks = sorted({k[2] for k in p2s_s3})
    s3g_p2s = {('P2S_LARGE_WITH_STOP', k): st.mean((p2s_s3[t, 'P2S_LARGE_WITH_STOP', k, 0.2] for t in tasks)) for k in ks}
    ad_s3 = {('P2S_LARGE_WITH_STOP', k): pg_ad['P2S_LARGE_WITH_STOP', k] + sum((runs['audit_S3_p2s_grid'][t, 'P2S_LARGE_WITH_STOP', k, 0.2]['adoption_rate'] - p2s_all[t, 'P2S_LARGE_WITH_STOP', k, 0.2]['adoption_rate'] for t in S3_OPPS)) / 50 for k in ks}
    res['P2S_S3_means'] = {f'P2S|{k}': v for (_, k), v in s3g_p2s.items()}
    res['P2S_S3_breakeven'] = breakevens(s3g_p2s, ad_s3, ('P2S_LARGE_WITH_STOP',))
    res['P2S_S3_breakeven_setup'] = breakevens(s3g_p2s, ad_s3, ('P2S_LARGE_WITH_STOP',), F=F_setup, Ks=(0,))
    res['P2S_grid_adoption'] = {f'{p}|{k}': v for (p, k), v in pg_ad.items()}
    full_s3 = {**s3_dec, **{k: v for k, v in p2s_s3.items()}, **{k: v for k, v in s3_post.items() if k[3] == 0.2}}
    res['P2S_contrasts_S3'] = contrasts(full_s3, spec2)
    if 'spend' in runs:
        spend = {}
        pub_cost = {}
        for r in csv.DictReader(open(f3 / 'final-7482/primary/summary_permutations.csv')):
            if r['split'] == 'holdout' and r['price'] == 'median' and (float(r['budget_eur']) == 0.2):
                pub_cost[r['policy_id'], K(r['demand_per_year'])] = float(r['mean_cost_eur'])
        for p in SPEND:
            for k in (8, 32):
                row = {}
                for b in (0.02, 0.05):
                    v = [runs['spend'][t, p, k, b] for t in tasks]
                    row[b] = {'mean_cost': st.mean((x['mean_cost_eur'] for x in v)), 'share_over': st.mean((x['share_over_budget'] for x in v)), 'max_over': max((x['max_overshoot_eur'] for x in v)), 'mean_steps': st.mean((x['mean_steps'] for x in v))}
                if p == 'P2S_LARGE_WITH_STOP' or p == 'P2_FIXED_LARGE':
                    v = [runs['p2s_grid'][t, p, k, 0.2] for t in tasks]
                    row[0.2] = {'mean_cost': st.mean((x['mean_cost_eur'] for x in v)), 'share_over': st.mean((x['share_over_budget'] for x in v)), 'max_over': max((x['max_overshoot_eur'] for x in v)), 'mean_steps': st.mean((x['mean_steps'] for x in v))}
                else:
                    row[0.2] = {'mean_cost': pub_cost[p, k]}
                spend[f'{p}|{k}'] = row
        res['spend'] = spend
    res['price'] = {}
    for lab in ('low', 'high'):
        if f'price_{lab}' not in runs:
            continue
        npv = npv_of(runs[f'price_{lab}'])
        m = means(npv, BASE9, (0, 4, 8, 12, 16, 32))
        be = {}
        for p in BASE9[1:]:
            if p in DEMAND_INDEPENDENT:
                b_ = (m[p, 8] - m[p, 0]) / (D(8) - D(0))
                a0 = m[p, 0] - b_ * D(0)
                assert all((close(m[p, k], a0 + b_ * D(k), rel=1e-06, abs_=1e-09) for k in (4, 12, 16, 32))), (lab, p)
                be[p] = -a0 / b_ if b_ > 0 and a0 < 0 else None
            else:
                be[p] = crossing(sorted(((D(k), m[p, k]) for k in (0, 4, 8, 12, 16, 32))))
        res['price'][lab] = {'means': {f'{p}|{k}': v for (p, k), v in m.items()}, 'contrasts': contrasts(npv), 'breakeven': be}
    if 'cpu_check' in runs:
        check('CPU factor 1.0 vs primary analysis', npv_of(runs['cpu_check']), prim)
    res['cpu'] = {}
    P2S, P2, P6 = ('P2S_LARGE_WITH_STOP', 'P2_FIXED_LARGE', 'P6_CASO_ADAPTIVE')
    for lab in ('half', 'double'):
        if f'cpu_{lab}' not in runs:
            continue
        npv = npv_of(runs[f'cpu_{lab}'])
        ks = (0, 4, 8, 12, 32)
        pols = BASE9 + (P2S,)
        m = means(npv, pols, ks)
        be = {}
        for p in pols[1:]:
            if p in DEMAND_INDEPENDENT:
                b_ = (m[p, 8] - m[p, 0]) / (D(8) - D(0))
                a0 = m[p, 0] - b_ * D(0)
                assert all((close(m[p, k], a0 + b_ * D(k), rel=1e-06, abs_=1e-09) for k in (4, 12, 32))), (lab, p)
                be[p] = -a0 / b_ if b_ > 0 and a0 < 0 else None
            else:
                be[p] = crossing(sorted(((D(k), m[p, k]) for k in ks)))
        spec_cpu = dict(CONTRASTS)
        spec_cpu.update({'P2S-P2_1e4': (P2S, P2, 8), 'P2S-P2_1e7': (P2S, P2, 32), 'P2S-P6_1e4': (P2S, P6, 8), 'P2S-P6_1e7': (P2S, P6, 32)})
        res['cpu'][lab] = {'means': {f'{p}|{k}': v for (p, k), v in m.items()}, 'contrasts': contrasts(npv, spec_cpu), 'breakeven': be}
    res['threshold'] = {}
    for T, name in ((1.2, 'thresh_1.2'), (1.5, 'thresh_1.5')):
        if name not in runs:
            continue
        new = npv_of(runs[name])
        opps = {k[0] for k in new}
        npv = splice({k: v for k, v in prim.items() if k[3] == 0.2}, new, opps)
        m = means(npv, BASE9, (0, 8, 32))
        be = {}
        for p in DEMAND_INDEPENDENT:
            b_ = (m[p, 8] - m[p, 0]) / (D(8) - D(0))
            a0 = m[p, 0] - b_ * D(0)
            be[p] = -a0 / b_ if b_ > 0 and a0 < 0 else None
        res['threshold'][str(T)] = {'affected': len(opps), 'means': {f'{p}|{k}': v for (p, k), v in m.items()}, 'contrasts': contrasts(npv), 'breakeven_affine': be, 'caso_p8_positive_at_1e4': {p: m[p, 8] > 0 for p in ('P6_CASO_ADAPTIVE', 'P8_COST_AWARE_BO_STOP')}}
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print('ok')
if __name__ == '__main__':
    main()
