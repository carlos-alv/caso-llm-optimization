import argparse
import csv
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from caso.bank import load_bank
from caso.pilot_analysis import opportunity_means, run, summarize, summarize_permutations
SIZE = {'qwen7b': 'small', 'gptoss20b': 'small', 'qwen32b': 'large', 'gptoss120b': 'large'}

def base_time_per_call(report):
    m = report['measurements']
    means = [statistics.mean(m[k]['times_s']) / m[k]['number'] for k in ('base_1', 'base_2')]
    return statistics.mean(means)

def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def _run_seed(job, seed):
    return [{k: v for k, v in r.items() if k != 'actions'} for r in job(order_seed=seed)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', nargs='+', required=True)
    ap.add_argument('--evaluations', nargs='+', required=True)
    ap.add_argument('--reports', required=True)
    ap.add_argument('--scenarios', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--rate', type=float, default=0.0)
    ap.add_argument('--strict-format', action='store_true', help='sensitivity: whole-function rewrites count as patch_invalid')
    ap.add_argument('--review', help="patch review JSON; verdict 'reject' -> rejected_by_review")
    ap.add_argument('--reject-persistent-state', action='store_true', help='automatic gate rule: patches flagged by caso.patch_review are rejected')
    ap.add_argument('--prices', nargs='+', help='subset of GPU price labels (default: all)')
    ap.add_argument('--demand-exponents', nargs='+', type=int, default=list(range(3, 13)))
    ap.add_argument('--budgets', nargs='+', type=float, default=[0.02, 0.05, 0.2])
    ap.add_argument('--permutations', type=int, default=0, help='also replay R seeded permutations of the within-action bank order')
    ap.add_argument('--perm-seed', type=int, default=20261027)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--ablations', action='store_true', help='add P6A_CASO_NO_AMDAHL (exploratory)')
    ap.add_argument('--thresholds', help='JSON {instance: tau_i} (protocol v1.0 per-instance rule)')
    ap.add_argument('--shares', help="JSON {instance: {hotspot_id: share}} overriding the bank's hotspot shares (e.g. from a fork-consistent profile)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rejected = []
    if a.review:
        rejected = [r['candidate_id'] for r in json.loads(Path(a.review).read_text())['reviews'] if r['verdict'] == 'reject']
    attempts = load_bank(map(Path, a.candidates), map(Path, a.evaluations), strict_format=a.strict_format, rejected_by_review=rejected, reject_persistent_state=a.reject_persistent_state, instance_thresholds=json.loads(Path(a.thresholds).read_text()) if a.thresholds else None)
    reports = {p.stem: json.loads(p.read_text()) for p in Path(a.reports).glob('*.json') if not p.name.startswith('_')}
    base = {iid: base_time_per_call(reports[iid]) for iid in {x.instance_id for x in attempts}}
    scen = json.loads(Path(a.scenarios).read_text())
    gpu = {s['scenario_id'].rsplit('-', 1)[-1]: s['rates']['gpu_hour_fully_loaded_eur']['value'] for s in scen}
    cpu = scen[0]['rates']['cpu_core_hour_fully_loaded_eur']['value']
    not_overridden = []
    if a.shares:
        shares = json.loads(Path(a.shares).read_text())
        attempts = [replace(x, hotspot_share=shares[x.instance_id][x.hotspot_id]) if x.hotspot_id in shares.get(x.instance_id, {}) else x for x in attempts]
        not_overridden = sorted({x.instance_id for x in attempts if x.instance_id not in shares})
    if a.prices:
        gpu = {k: gpu[k] for k in a.prices}
    present = {x.model_key for x in attempts}
    job = partial(run, attempts, base, {k: v for k, v in SIZE.items() if k in present}, gpu, cpu, [10.0 ** k for k in a.demand_exponents], a.budgets, horizon=3, rate=a.rate, ablations=a.ablations)
    rows = job()
    for r in rows:
        r['actions'] = ' '.join(r['actions'])
    write_csv(out / 'replay_rows.csv', rows)
    write_csv(out / 'summary.csv', summarize(rows))
    write_csv(out / 'bank_attempts.csv', (asdict(x) for x in attempts))
    if a.permutations:
        seeds = [a.perm_seed + k for k in range(a.permutations)]
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            perm_rows = [r for chunk in pool.map(_run_seed, [job] * len(seeds), seeds) for r in chunk]
        write_csv(out / 'summary_permutations.csv', summarize_permutations(perm_rows))
        write_csv(out / 'opportunity_means.csv', opportunity_means(perm_rows))
    meta = {'data_kind': 'measured costs and outcomes; prices and demand are scenarios (estimated)', 'status': 'PILOT / EXPLORATORY: not preregistered, few opportunities, not for confirmatory claims', 'n_attempts': len(attempts), 'instances': sorted(base), 'models': sorted(present), 'base_time_per_call_s': base, 'gpu_prices_eur_per_h': gpu, 'cpu_price_eur_per_vcpu_h': cpu, 'discount_rate': a.rate, 'horizon_years': 3, 'strict_format': a.strict_format, 'rejected_by_review': rejected, 'reject_persistent_state': a.reject_persistent_state, 'permutations': a.permutations, 'perm_seed': a.perm_seed if a.permutations else None, 'ablations': a.ablations, 'shares_override': a.shares, 'shares_not_overridden': not_overridden, 'thresholds': a.thresholds}
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=1))
if __name__ == '__main__':
    main()
