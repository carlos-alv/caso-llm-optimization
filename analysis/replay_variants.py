import argparse
import csv
import json
import math
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pilot_analysis import SIZE, base_time_per_call
import caso.pilot_analysis as pa
from caso.bank import load_bank
from caso.decision import BetaPosterior, NormalPosterior
from caso.policies import P2FixedLarge, _stop
ORIG_BUILD = pa.build_policies
import posthoc_ablation as abl
import posthoc_policies2 as pol2
pa.build_policies = ORIG_BUILD
BASE_IDS = ('P0_STOP', 'P1_FIXED_SMALL', 'P2_FIXED_LARGE', 'P3_PERFORMANCE_FIRST', 'P4_RANDOM_ROUTER', 'P5_COST_HEURISTIC', 'P6_CASO_ADAPTIVE', 'P7_SYSLMMATIC_REPLICA', 'P8_COST_AWARE_BO_STOP', 'P6A_CASO_NO_AMDAHL')

class P2WithStop(P2FixedLarge):
    policy_id = 'P2S_LARGE_WITH_STOP'

    def decide(self, state):
        usable = [x for x in abl._caso_scored(self, state) if x[0].feasible]
        if not usable or max((x[4] for x in usable)) <= 0:
            return _stop('no action with positive expected marginal value')
        return super().decide(state)
EXTRA = {'P6N_CASO_NO_STOP': abl.P6NoStop, 'P3S_PERF_WITH_STOP': abl.P3WithStop, 'P6C_CASO_CONFIRMED': pol2.P6Confirmed, 'P5F_CASCADE_FULL': pol2.P5Full, 'P2S_LARGE_WITH_STOP': P2WithStop}

def make_policies(cost_model, priors, names):
    base = ORIG_BUILD(cost_model, priors, ablations='P6A_CASO_NO_AMDAHL' in names)
    return [p for p in base if p.policy_id in names] + [cls(cost_model, priors) for n, cls in EXTRA.items() if n in names]

def run_seed(attempts, base_s, size_class, gpu_price, cpu_price, instances, demands, budgets, names, order_seed):
    pool = [a for a in attempts if a.split in ('dev', 'calibration')]
    priors = pa.priors_from(pool, size_class)
    prices = pa.Prices(gpu_price, cpu_price)
    cost_model = pa.CostModel(priors, prices)
    acc = {}
    for iid in instances:
        bank = pa.bank_for(attempts, iid, prices, order_seed)
        c0 = base_s[iid] / 3600 * cpu_price
        for demand in demands:
            multiplier = demand * pa.annuity(0.0, 3)
            for budget in budgets:
                state = pa.OpportunityState(iid, c0, multiplier, pa.actions_for(attempts, iid), (), budget)
                for policy in make_policies(cost_model, priors, names):
                    r = pa.replay(policy, state, bank)
                    over = r.total_cost_eur - budget
                    acc[iid, policy.policy_id, demand, budget] = (r.npv_eur, r.total_cost_eur, r.value_eur > 0, bool(r.censored), r.steps, over > 1e-12, over)
    return acc

def parse_demands(spec):
    out = []
    for part in spec:
        if part.startswith('grid:'):
            rng, _, step = part[5:].partition(':')
            k1, k2 = (int(x) for x in rng.split('-'))
            out += [10.0 ** (3 + k / 8) for k in range(k1, k2 + 1, int(step or 1))]
        else:
            out.append(float(part))
    return sorted(set(out))

def write(path, total, n):
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['task_id', 'policy_id', 'demand_per_year', 'budget_eur', 'n_permutations', 'mean_npv_eur', 'mean_cost_eur', 'adoption_rate', 'censored_rate', 'mean_steps', 'share_over_budget', 'max_overshoot_eur'])
        for (t, p, d, b), v in sorted(total.items()):
            w.writerow([t, p, repr(d), b, n, v[0] / n, v[1] / n, v[2] / n, v[3] / n, v[4] / n, v[5] / n, v[6]])

def main():
    ap = argparse.ArgumentParser()
    for k in ('candidates', 'evaluations'):
        ap.add_argument(f'--{k}', nargs='+', required=True)
    for k in ('reports', 'scenarios', 'shares', 'thresholds', 'out'):
        ap.add_argument(f'--{k}', required=True)
    ap.add_argument('--variant', default='primary')
    ap.add_argument('--price', default='median')
    ap.add_argument('--cpu-factor', type=float, default=1.0, help='multiply the CPU price of the scenario (sensitivity; 1.0 = protocol price)')
    ap.add_argument('--policies', nargs='+', required=True)
    ap.add_argument('--instances', nargs='*', default=[], help='held-out instances (default: all)')
    ap.add_argument('--demands', nargs='+', required=True)
    ap.add_argument('--budgets', nargs='+', type=float, required=True)
    ap.add_argument('--permutations', type=int, default=200)
    ap.add_argument('--perm-seed', type=int, default=20261027)
    ap.add_argument('--workers', type=int, default=14)
    a = ap.parse_args()
    unknown = set(a.policies) - set(BASE_IDS) - set(EXTRA)
    assert not unknown, unknown
    thresholds = json.loads(Path(a.thresholds).read_text())
    rejected, meta = ([], {'variant': a.variant})
    if a.variant.startswith('review:'):
        rev = json.loads(Path(a.variant[7:]).read_text())
        rejected = [x['candidate_id'] for x in rev['reviews'] if x['verdict'] == 'reject']
        meta.update(review_variant=rev['variant'], rejected=len(rejected))
    attempts = load_bank(map(Path, a.candidates), map(Path, a.evaluations), reject_persistent_state=True, instance_thresholds=thresholds, rejected_by_review=rejected)
    holdout = sorted({x.instance_id for x in attempts if x.split == 'holdout'})
    if a.variant.startswith('threshold:'):
        T = float(a.variant[10:])
        pooled = min(thresholds.values())
        stricter = {**thresholds, **{i: max(thresholds.get(i, pooled), T) for i in holdout}}
        attempts = load_bank(map(Path, a.candidates), map(Path, a.evaluations), reject_persistent_state=True, instance_thresholds=stricter)
        meta.update(threshold=T, pooled_tau=pooled)
    pool = [x for x in attempts if x.split in ('dev', 'calibration')]
    logs = [math.log(x.speedup) for x in pool if x.improvement_confirmed and x.speedup and (x.speedup > 0)]
    mu, sd = (statistics.mean(logs), statistics.stdev(logs))
    for m in SIZE:
        own = [x for x in pool if x.model_key == m]
        nconf = sum((x.improvement_confirmed for x in own))
        pol2.CONF[m] = (BetaPosterior(1 + nconf, 1 + len(own) - nconf), NormalPosterior(mu, sd ** 2, sd))
    reports = {p.stem: json.loads(p.read_text()) for p in Path(a.reports).glob('*.json') if not p.name.startswith('_')}
    base_s = {iid: base_time_per_call(reports[iid]) for iid in {x.instance_id for x in attempts}}
    scen = json.loads(Path(a.scenarios).read_text())
    gpu = {s['scenario_id'].rsplit('-', 1)[-1]: s['rates']['gpu_hour_fully_loaded_eur']['value'] for s in scen}
    cpu = scen[0]['rates']['cpu_core_hour_fully_loaded_eur']['value'] * a.cpu_factor
    shares = json.loads(Path(a.shares).read_text())
    attempts = [replace(x, hotspot_share=shares[x.instance_id][x.hotspot_id]) if x.hotspot_id in shares.get(x.instance_id, {}) else x for x in attempts]
    present = {x.model_key for x in attempts}
    instances = a.instances or holdout
    assert set(instances) <= set(holdout), set(instances) - set(holdout)
    demands = parse_demands(a.demands)
    meta.update(price=a.price, gpu_eur_per_h=gpu[a.price], cpu_factor=a.cpu_factor, cpu_eur_per_vcpu_h=cpu, instances=instances, demands=demands, budgets=a.budgets, policies=a.policies, permutations=a.permutations, perm_seed=a.perm_seed, confirmed_held_out=sum((x.improvement_confirmed for x in attempts if x.split == 'holdout')))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    job = partial(run_seed, attempts, base_s, {k: v for k, v in SIZE.items() if k in present}, gpu[a.price], cpu, instances, demands, a.budgets, set(a.policies))
    seeds = [a.perm_seed + k for k in range(a.permutations)]
    total, n = ({}, 0)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for acc in ex.map(job, seeds):
            n += 1
            for key, (npv, cost, adopted, cens, steps, over_flag, over) in acc.items():
                t = total.setdefault(key, [0.0, 0.0, 0, 0, 0, 0, -math.inf])
                t[0] += npv
                t[1] += cost
                t[2] += adopted
                t[3] += cens
                t[4] += steps
                t[5] += over_flag
                t[6] = max(t[6], over)
            if n % 20 == 0 or n == len(seeds):
                write(out / 'opportunity_means.csv', total, n)
                print(f'{n} permutations', flush=True)
    print('done', len(total))
if __name__ == '__main__':
    main()
