import argparse
import csv
import glob
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
BASE = 10000000.0
HORIZON = 3
POLICIES = ('P1_FIXED_SMALL', 'P2_FIXED_LARGE', 'P3_PERFORMANCE_FIRST', 'P4_RANDOM_ROUTER', 'P5_COST_HEURISTIC', 'P6_CASO_ADAPTIVE', 'P7_SYSLMMATIC_REPLICA', 'P8_COST_AWARE_BO_STOP')
DEMAND_INDEPENDENT = ('P1_FIXED_SMALL', 'P2_FIXED_LARGE', 'P3_PERFORMANCE_FIRST', 'P4_RANDOM_ROUTER', 'P5_COST_HEURISTIC', 'P7_SYSLMMATIC_REPLICA')
AUDITED = {'sympy__sympy-14772', 'scipy__scipy-13566', 'scipy__scipy-11982'}
CONTRASTS = {'H1': ('P6_CASO_ADAPTIVE', 'P3_PERFORMANCE_FIRST', BASE, 'primary'), 'H3': ('P6_CASO_ADAPTIVE', 'P3_PERFORMANCE_FIRST', 10000.0, 'primary'), 'H1b': ('P6_CASO_ADAPTIVE', 'P8_COST_AWARE_BO_STOP', BASE, 'primary'), 'H1c': ('P6_CASO_ADAPTIVE', 'P5_COST_HEURISTIC', BASE, 'primary'), 'H5': ('P6_CASO_ADAPTIVE', 'P4_RANDOM_ROUTER', BASE, 'primary'), 'P2-P6': ('P2_FIXED_LARGE', 'P6_CASO_ADAPTIVE', BASE, 'primary'), 'STOP_in_P3_1e4': ('P3S_PERF_WITH_STOP', 'P3_PERFORMANCE_FIRST', 10000.0, 'ablation'), 'STOP_in_CASO_1e4': ('P6_CASO_ADAPTIVE', 'P6N_CASO_NO_STOP', 10000.0, 'ablation')}

def repo_of(task_id):
    owner, rest = task_id.split('__', 1)
    return f"{owner}/{rest.rsplit('-', 1)[0]}"

def close(a, b, rel=1e-09, abs_=1e-12):
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))

def crossings(points):
    out = []
    for (d0, v0), (d1, v1) in zip(points, points[1:]):
        if v0 <= 0 < v1 or v0 > 0 >= v1:
            out.append((d0 + (0 - v0) * (d1 - d0) / (v1 - v0), 'up' if v1 > 0 else 'down'))
    return out

def first_up(points):
    ups = [d for d, kind in crossings(points) if kind == 'up']
    if ups:
        return ups[0]
    return None if points[-1][1] <= 0 else points[0][0]

def unit_cpu_s(r):
    s = 0.0
    for k in ('apply', 'tests'):
        if isinstance(r.get(k), dict):
            s += r[k].get('cpu_core_s') or 0.0
    for b in r.get('bench') or []:
        s += b.get('cpu_core_s') or 0.0
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--caso-src', required=True, help='src/ of the frozen code snapshot')
    ap.add_argument('--primary', required=True, help='final-7482/primary')
    ap.add_argument('--ablation', required=True, help='posthoc-abl-8143/opportunity_means_holdout.csv')
    ap.add_argument('--grid', nargs='+', required=True)
    ap.add_argument('--controls', required=True)
    ap.add_argument('--reports', required=True, help='verification reports of the held-out instances')
    ap.add_argument('--reviews', nargs='*', default=[], help='patch-review files (S3/S4) for bounds without replay')
    ap.add_argument('--review-base', nargs='*', default=[], help='opportunity_means.csv of the replayed sensitivity each review extends (S1 for S3, S2 for S4)')
    ap.add_argument('--evaluations', nargs='+', required=True)
    ap.add_argument('--scenarios', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.caso_src)
    from caso.inference import hierarchical_bootstrap, leave_one_repo_out
    prim = Path(a.primary)
    meta = json.loads((prim / 'meta.json').read_text())
    p_cpu = meta['cpu_price_eur_per_vcpu_h']
    p_gpu_med = meta['gpu_prices_eur_per_h']['median']
    scen = json.loads(Path(a.scenarios).read_text())
    gpu_prices = {s['scenario_id'].rsplit('-', 1)[-1]: s['rates']['gpu_hour_fully_loaded_eur']['value'] for s in scen}
    assert close(gpu_prices['median'], p_gpu_med) and close(scen[0]['rates']['cpu_core_hour_fully_loaded_eur']['value'], p_cpu)
    out = {'note': 'exploratory post-processing for extension; no replay and no new measurements', 'prices': {'gpu_eur_per_h': gpu_prices, 'cpu_eur_per_vcpu_h': p_cpu}}
    npv = {}
    for r in csv.DictReader(open(prim / 'opportunity_means.csv')):
        if r['split'] == 'holdout' and r['price'] == 'median' and (float(r['budget_eur']) == 0.2):
            assert int(r['n_permutations']) == 200
            npv['primary', float(r['demand_per_year']), r['task_id'], r['policy_id']] = float(r['mean_npv_eur'])
    for r in csv.DictReader(open(a.ablation)):
        if float(r['budget_eur']) == 0.2:
            npv['ablation', float(r['demand_per_year']), r['task_id'], r['policy_id']] = float(r['mean_npv_eur'])
    tasks = sorted({k[2] for k in npv if k[0] == 'primary'})
    assert len(tasks) == 50
    for t in tasks:
        for d in (10000.0, BASE):
            for p in ('P3_PERFORMANCE_FIRST', 'P6_CASO_ADAPTIVE'):
                assert close(npv['ablation', d, t, p], npv['primary', d, t, p], rel=1e-09, abs_=1e-09), (t, d, p)
    oracle, c0, rows_default = ({}, {}, [])
    for r in csv.DictReader(open(prim / 'replay_rows.csv')):
        if r['split'] != 'holdout' or r['price'] != 'median' or float(r['budget_eur']) != 0.2:
            continue
        rows_default.append(r)
        oracle[float(r['demand_per_year']), r['task_id']] = float(r['oracle_npv_eur'])
        c0[r['task_id']] = float(r['c0_eur'])
    mean_oracle_base = st.mean((oracle[BASE, t] for t in tasks))

    def contrast(name, excluded):
        p, ref, d, src = CONTRASTS[name]
        diffs = defaultdict(list)
        for t in tasks:
            if t not in excluded:
                diffs[repo_of(t)].append(npv[src, d, t, p] - npv[src, d, t, ref])
        hb = hierarchical_bootstrap(diffs, reps=5000)
        return {'estimate': hb['estimate'], 'ci95': hb['ci95'], 'p_one_sided_greater': hb['p_one_sided_greater'], 'p_two_sided': hb['p_two_sided'], 'n': hb['n_opportunities'], 'loo': leave_one_repo_out(diffs)}
    rank_oracle = sorted(tasks, key=lambda t: -oracle[BASE, t])
    rank_caso = sorted(tasks, key=lambda t: -npv['primary', BASE, t, 'P6_CASO_ADAPTIVE'])
    total_pos = sum((max(0.0, oracle[BASE, t]) for t in tasks))
    A = {'ranking_oracle_top10': [(t, oracle[BASE, t], oracle[BASE, t] / total_pos) for t in rank_oracle[:10]], 'ranking_caso_top10': [(t, npv['primary', BASE, t, 'P6_CASO_ADAPTIVE']) for t in rank_caso[:10]], 'variants': {}}
    full = {n: contrast(n, set()) for n in CONTRASTS}
    pub = json.loads((prim / 'contrasts_holdout.json').read_text())['contrasts']
    for n in ('H1', 'H3', 'H1b', 'H1c', 'H5'):
        assert close(full[n]['estimate'], pub[n]['estimate'], rel=1e-09), n
        assert close(full[n]['ci95'][0], pub[n]['ci95'][0], rel=1e-09) and close(full[n]['ci95'][1], pub[n]['ci95'][1], rel=1e-09), n
    for rank_name, ranking in (('oracle', rank_oracle), ('caso', rank_caso)):
        for k in (0, 1, 3, 5):
            excl = set(ranking[:k])
            keep = [t for t in tasks if t not in excl]
            means = {f'{p}|{d:g}': st.mean((npv['primary', d, t, p] for t in keep)) for p in POLICIES for d in (10000.0, BASE)}
            means.update({f'{p}|{d:g}': st.mean((npv['ablation', d, t, p] for t in keep)) for p in ('P3S_PERF_WITH_STOP', 'P6N_CASO_NO_STOP') for d in (10000.0, BASE)})
            means['ORACLE|1e+07'] = st.mean((oracle[BASE, t] for t in keep))
            A['variants'][f'{rank_name}|{k}'] = {'excluded': sorted(excl), 'n': len(keep), 'means': means, 'contrasts': {n: full[n] if k == 0 else contrast(n, excl) for n in CONTRASTS}}
    decades = (1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0)
    A['breakeven_without_top_k'] = {}
    for k in (0, 1, 3, 5):
        keep = [t for t in tasks if t not in set(rank_oracle[:k])]
        row = {}
        for p in POLICIES:
            m = {d: st.mean((npv['primary', d, t, p] for t in keep)) for d in decades}
            if p in DEMAND_INDEPENDENT:
                b = (m[10000.0] - m[1000.0]) / (10000.0 - 1000.0)
                a0 = m[1000.0] - b * 1000.0
                assert all((close(m[d], a0 + b * d, rel=1e-06, abs_=1e-09) for d in decades)), (p, k)
                row[p] = -a0 / b if b > 0 and a0 < 0 else None
            else:
                pos = [d for d in decades if m[d] > 0]
                row[p] = {'first_positive_decade': pos[0] if pos else None, 'means': m}
        A['breakeven_without_top_k'][k] = row
    out['A_leave_top_k_out'] = A
    ctl = [json.loads(line) for line in open(a.controls)]
    null_s, units = (defaultdict(float), defaultdict(lambda: defaultdict(int)))
    for r in ctl:
        i = r['instance_id']
        if i not in tasks:
            continue
        if r['unit'] == 'profile_fork':
            null_s[i] += r['profile'].get('total_s') or 0.0
            units[i]['profile_fork'] += 1
        elif r['unit'] in ('aa_plain', 'aa_tested', 'placebo'):
            null_s[i] += unit_cpu_s(r)
            units[i][r['unit']] += 1
    base_s = {}
    for t in tasks:
        rep = json.loads((Path(a.reports) / f'{t}.json').read_text())
        s = sum((rep['stages'][k].get('cpu_core_s') or 0.0 for k in ('extract', 'import', 'tests_base', 'profile')))
        s += sum((rep['measurements'][k].get('cpu_core_s') or 0.0 for k in ('warm_base', 'base_1', 'base_2')))
        base_s[t] = s
    null_eur = sorted((v / 3600 * p_cpu for v in null_s.values()))
    setup_eur = {t: (base_s[t] + null_s.get(t, 0.0)) / 3600 * p_cpu for t in tasks}
    vals = sorted(setup_eur.values())
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]
    F_mean = st.mean(vals)
    B = {'null_units_only': {'n': len(null_eur), 'median': st.median(null_eur), 'mean': st.mean(null_eur), 'max': max(null_eur)}, 'baseline_steps_only': {'median': st.median((v / 3600 * p_cpu for v in base_s.values())), 'mean': st.mean((v / 3600 * p_cpu for v in base_s.values()))}, 'n': len(vals), 'median': st.median(vals), 'mean': F_mean, 'p90': q(vals, 0.9), 'max': max(vals), 'min': min(vals), 'units': {i: dict(u) for i, u in units.items()}, 'breakeven': {}}
    grid = defaultdict(list)
    for gf in a.grid:
        g = json.loads(Path(gf).read_text())
        assert g['permutations'] == 200 and g['budget_eur'] == 0.2
        for r in g['rows']:
            grid[r['policy_id']].append((r['demand_per_year'], r['mean_npv_eur']))
    for p in grid:
        grid[p].sort()
    for label, F in (('F0', 0.0), ('F_mean_measured', F_mean), ('F_max_measured', max(vals))):
        B['breakeven'][label] = {'F': F, **{p: first_up([(d, v - F) for d, v in grid[p]]) for p in POLICIES}}
    B['npv_1e4_minus_F_mean'] = {p: st.mean((npv['primary', 10000.0, t, p] for t in tasks)) - F_mean for p in POLICIES}
    out['B_setup_cost'] = B
    C = {'mean_cost_eur': {}, 'overshoot_default_order': {}}
    for r in csv.DictReader(open(prim / 'summary_permutations.csv')):
        if r['split'] == 'holdout' and r['price'] == 'median' and (float(r['budget_eur']) == 0.2) and (float(r['demand_per_year']) <= 10000000.0):
            C['mean_cost_eur'][f"{r['policy_id']}|{float(r['demand_per_year']):g}"] = float(r['mean_cost_eur'])
    for p in POLICIES + ('P6A_CASO_NO_AMDAHL',):
        for d in (10000.0, BASE):
            rs = [r for r in rows_default if r['policy_id'] == p and float(r['demand_per_year']) == d]
            over = [float(r['total_cost_eur']) - 0.2 for r in rs]
            C['overshoot_default_order'][f'{p}|{d:g}'] = {'n': len(rs), 'share_over_budget': sum((o > 1e-12 for o in over)) / len(rs), 'max_overshoot': max(over), 'mean_steps': st.mean((int(r['steps']) for r in rs))}
    out['C_realized_spend'] = C
    bank = [r for r in csv.DictReader(open(prim / 'bank_attempts.csv')) if r['instance_id'] in tasks]
    assert len(bank) == 2304, len(bank)
    ev = {}
    for fn in a.evaluations:
        for f in sorted(glob.glob(fn)):
            with open(f) as fh:
                for line in fh:
                    r = json.loads(line)
                    if r.get('instance_id') in tasks and r.get('status') == 'benchmarked':
                        ev[r['candidate_id']] = r
    conf0 = [r for r in bank if r['improvement_confirmed'] == 'True']
    assert len(conf0) == 153, len(conf0)
    s12 = {}
    for r in bank:
        e = ev.get(r['candidate_id'])
        if e is None:
            continue
        s1 = (e.get('speedup') or {}).get('speedup')
        s2 = ((e.get('remeasure') or {}).get('speedup') or {}).get('speedup')
        s12[r['candidate_id']] = (s1, s2, bool(e.get('stage1_confirmed')), bool((e.get('remeasure') or {}).get('passed')), bool(e.get('persistent_state_flags')), e.get('speedup_final'))
    for r in conf0:
        s1, s2, st1, st2, ps, sf = s12[r['candidate_id']]
        assert st1 and st2 and (not ps) and s1 and s2, r['candidate_id']
        assert close(float(r['speedup']), sf, rel=1e-09), r['candidate_id']

    def attempt_cost(r, p_gpu):
        return float(r['gpu_s']) / 3600 * p_gpu + float(r['cpu_core_s']) / 3600 * p_cpu

    def oracle_mean(confirmed_ids, demand, p_gpu):
        best = defaultdict(float)
        for r in bank:
            t = r['instance_id']
            v = c0[t] * (1 - 1 / float(r['speedup'])) * demand * HORIZON if r['candidate_id'] in confirmed_ids else 0.0
            best[t] = max(best[t], v - attempt_cost(r, p_gpu))
        return st.mean((max(0.0, best[t]) for t in tasks))
    ids0 = {r['candidate_id'] for r in conf0}
    o_base = oracle_mean(ids0, BASE, p_gpu_med)
    assert close(o_base, mean_oracle_base, rel=1e-06), (o_base, mean_oracle_base)
    assert close(oracle_mean(ids0, 10000.0, p_gpu_med), st.mean((oracle[10000.0, t] for t in tasks)), rel=1e-06)
    pooled = {cid for cid, (s1, s2, st1, st2, ps, sf) in s12.items() if st1 and st2 and (not ps)}
    D = {'variants': {}}
    variants = [('primary (tau_i)', ids0), ('pooled tau only (no post-evaluation null units)', pooled)]
    for T in (1.1, 1.2, 1.5, 2.0):
        variants.append((f'min speedup {T:.2f} in both stages', {cid for cid in ids0 if min(s12[cid][0], s12[cid][1]) >= T}))
    for name, ids in variants:
        opps = {cid.split(':')[0] for cid in ids}
        D['variants'][name] = {'confirmed': len(ids), 'opportunities': len(opps), 'oracle_1e7': oracle_mean(ids, BASE, p_gpu_med), 'oracle_1e4': oracle_mean(ids, 10000.0, p_gpu_med), 'oracle_share_1e7': oracle_mean(ids, BASE, p_gpu_med) / o_base}
    best_s = defaultdict(float)
    for r in conf0:
        best_s[r['instance_id']] = max(best_s[r['instance_id']], float(r['speedup']))
    D['oracle_share_by_best_speedup'] = {f'>={x}': sum((max(0.0, oracle[BASE, t]) for t in tasks if best_s.get(t, 0) >= x)) / total_pos for x in (1.2, 1.5, 2, 5, 10)}
    out['D_thresholds'] = D
    E = {}
    for lab, pg in gpu_prices.items():
        tot = sum((attempt_cost(r, pg) for r in bank))
        gpu_part = sum((float(r['gpu_s']) / 3600 * pg for r in bank))
        per_model = {}
        for m in sorted({r['model_key'] for r in bank}):
            rs = [r for r in bank if r['model_key'] == m]
            n_conf = sum((r['improvement_confirmed'] == 'True' for r in rs))
            per_model[m] = {'mean_cost_per_attempt': st.mean((attempt_cost(r, pg) for r in rs)), 'cost_per_confirmed': sum((attempt_cost(r, pg) for r in rs)) / n_conf}
        E[lab] = {'gpu_eur_per_h': pg, 'bank_total_eur': tot, 'gpu_share': gpu_part / tot, 'mean_cost_per_attempt': tot / len(bank), 'per_model': per_model, 'oracle_1e4': oracle_mean(ids0, 10000.0, pg), 'oracle_1e7': oracle_mean(ids0, BASE, pg)}
    costs_med = sorted((attempt_cost(r, p_gpu_med) for r in bank))
    E['attempt_cost_quantiles_median_price'] = {'median': st.median(costs_med), 'p95': costs_med[int(0.95 * len(costs_med))], 'max': costs_med[-1]}
    out['E_gpu_price_bank_level'] = E
    per_opp = defaultdict(list)
    for r in conf0:
        t = r['instance_id']
        per_opp[t].append((c0[t] * (1 - 1 / float(r['speedup'])) * BASE * HORIZON - attempt_cost(r, p_gpu_med), float(r['speedup']), r['candidate_id']))
    rank = sorted(per_opp, key=lambda t: -max((x[0] for x in per_opp[t])))
    cum, F_rows = (0.0, [])
    for t in rank:
        cum += max(0.0, oracle[BASE, t])
        F_rows.append({'instance': t, 'audited': t in AUDITED, 'oracle_1e7': oracle[BASE, t], 'cum_share': cum / total_pos, 'n_confirmed': len(per_opp[t]), 'best_speedup': max((x[1] for x in per_opp[t])), 'patches': sorted(per_opp[t], reverse=True)})
    out['F_audit_ranking'] = F_rows
    G = {}
    worst = 0.2 + costs_med[-1]
    for rf, base_file in zip(a.reviews, a.review_base):
        rev = json.loads(Path(rf).read_text())
        rejected = {x['candidate_id'] for x in rev['reviews']}
        ids = ids0 - rejected
        left = defaultdict(int)
        for cid in ids:
            left[cid.split(':')[0]] += 1
        affected = sorted({cid.split(':')[0] for cid in rejected} - {'scipy__scipy-11982'})
        wiped = [t for t in affected if left[t] == 0]
        partial = [t for t in affected if left[t] > 0]
        base_npv = {}
        for r in csv.DictReader(open(base_file)):
            if r['split'] == 'holdout' and r['price'] == 'median' and (float(r['budget_eur']) == 0.2):
                base_npv[float(r['demand_per_year']), r['task_id'], r['policy_id']] = float(r['mean_npv_eur'])
        o_new = {}
        for d in (10000.0, BASE):
            best = defaultdict(float)
            for r in bank:
                t = r['instance_id']
                v = c0[t] * (1 - 1 / float(r['speedup'])) * d * HORIZON if r['candidate_id'] in ids else 0.0
                best[t] = max(best[t], v - attempt_cost(r, p_gpu_med))
            o_new[d] = {t: max(0.0, best[t]) for t in tasks}
        g = {'rejected': len(rejected), 'base': base_file, 'wiped': wiped, 'partially_affected': partial, 'oracle_1e7': st.mean(o_new[BASE].values()), 'oracle_1e4': st.mean(o_new[10000.0].values()), 'means': {}, 'contrasts_mid': {}}

        def bounds(d, t):
            return (-worst, 0.0) if t in wiped else (-worst, o_new[d][t])
        for d in (10000.0, BASE):
            for p in POLICIES:
                s = sum((base_npv[d, t, p] for t in tasks if t not in affected))
                lo = s + sum((bounds(d, t)[0] for t in affected))
                hi = s + sum((bounds(d, t)[1] for t in affected))
                g['means'][f'{p}|{d:g}'] = [lo / 50, hi / 50]
        for n, (p, ref, d, src) in CONTRASTS.items():
            if src != 'primary':
                continue
            diffs = defaultdict(list)
            for t in tasks:
                diffs[repo_of(t)].append(0.0 if t in affected else base_npv[d, t, p] - base_npv[d, t, ref])
            hb = hierarchical_bootstrap(diffs, reps=5000)
            half = sum((max((abs(x) for x in bounds(d, t))) for t in affected)) / 50
            g['contrasts_mid'][n] = {'estimate': hb['estimate'], 'ci95': hb['ci95'], 'half_width_bound': half, 'p_one_sided_greater': hb['p_one_sided_greater']}
        G[rev['variant']] = g
    out['G_audit_bounds'] = G
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print('ok')
if __name__ == '__main__':
    main()
