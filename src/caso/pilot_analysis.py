from __future__ import annotations
import statistics
from dataclasses import asdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence
from .bank import BankAttempt, actions_for, bank_for, priors_from
from .decision import CostModel, OpportunityState, Prices, annuity
from .policies import P0Stop, P1FixedSmall, P2FixedLarge, P3PerformanceFirst, P4RandomRouter, P5CostHeuristic, P6CasoNoAmdahl, P6CasoVOI, P7SysLLMaticAdapted, P8CostAwareBOStop
from .replay import replay, verified_saving_per_exec

def build_policies(cost_model: CostModel, priors, seed: int=20261025, ablations: bool=False):
    policies = [P0Stop(cost_model, priors), P1FixedSmall(cost_model, priors), P2FixedLarge(cost_model, priors), P3PerformanceFirst(cost_model, priors), P4RandomRouter(cost_model, priors, seed=seed), P5CostHeuristic(cost_model, priors), P6CasoVOI(cost_model, priors), P7SysLLMaticAdapted(cost_model, priors), P8CostAwareBOStop(cost_model, priors)]
    return policies + ([P6CasoNoAmdahl(cost_model, priors)] if ablations else [])

def oracle_npv(bank, c0: float, multiplier: float) -> float:
    best = 0.0
    for observations in bank.values():
        for o in observations:
            best = max(best, verified_saving_per_exec(c0, o) * multiplier - o.cost_eur)
    return best

def run(attempts: Sequence[BankAttempt], base_time_per_call_s: Mapping[str, float], size_class: Mapping[str, str], gpu_prices: Mapping[str, float], cpu_price: float, demands: Iterable[float], budgets: Iterable[float], *, horizon: int=3, rate: float=0.0, order_seed: Optional[int]=None, ablations: bool=False) -> List[Dict[str, object]]:
    rows = []
    instances = sorted({a.instance_id for a in attempts})
    dev = [a for a in attempts if a.split == 'dev']
    for iid in instances:
        split = next((a.split for a in attempts if a.instance_id == iid))
        prior_pool = [a for a in attempts if a.split in ('dev', 'calibration')] if split == 'holdout' else [a for a in dev if a.instance_id != iid]
        priors = priors_from(prior_pool, size_class)
        for price_label, gpu_price in gpu_prices.items():
            prices = Prices(gpu_price, cpu_price)
            bank = bank_for(attempts, iid, prices, order_seed)
            c0 = base_time_per_call_s[iid] / 3600 * cpu_price
            cost_model = CostModel(priors, prices)
            for demand in demands:
                multiplier = demand * annuity(rate, horizon)
                bound = oracle_npv(bank, c0, multiplier)
                for budget in budgets:
                    state = OpportunityState(iid, c0, multiplier, actions_for(attempts, iid), (), budget)
                    for policy in build_policies(cost_model, priors, ablations=ablations):
                        r = replay(policy, state, bank)
                        rows.append({**asdict(r), 'split': split, 'price': price_label, 'demand_per_year': demand, 'budget_eur': budget, 'oracle_npv_eur': bound, 'c0_eur': c0, 'n_prior_attempts': len(prior_pool), 'order_seed': order_seed})
    return rows

def summarize(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    keyed: Dict[tuple, Dict[str, Mapping]] = {}
    for r in rows:
        keyed.setdefault((r['split'], r['price'], r['demand_per_year'], r['budget_eur'], r['task_id']), {})[r['policy_id']] = r
    groups: Dict[tuple, List[Dict[str, Mapping]]] = {}
    for (split, price, demand, budget, _task), by_policy in keyed.items():
        groups.setdefault((split, price, demand, budget), []).append(by_policy)
    out = []
    for (split, price, demand, budget), tasks in sorted(groups.items()):
        for policy in sorted(tasks[0]):
            npv = [t[policy]['npv_eur'] for t in tasks]
            row = {'split': split, 'price': price, 'demand_per_year': demand, 'budget_eur': budget, 'policy_id': policy, 'n_tasks': len(tasks), 'mean_npv_eur': statistics.mean(npv), 'mean_cost_eur': statistics.mean((t[policy]['total_cost_eur'] for t in tasks)), 'stop_without_attempt': sum((t[policy]['steps'] == 0 for t in tasks)), 'censored': sum((bool(t[policy]['censored']) for t in tasks)), 'mean_oracle_npv_eur': statistics.mean((t[policy]['oracle_npv_eur'] for t in tasks))}
            for ref in ('P5_COST_HEURISTIC', 'P8_COST_AWARE_BO_STOP', 'P0_STOP'):
                row[f'mean_diff_vs_{ref}'] = statistics.mean((t[policy]['npv_eur'] - t[ref]['npv_eur'] for t in tasks))
            out.append(row)
    return out

def _quantile(xs: Sequence[float], q: float) -> float:
    xs = sorted(xs)
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)

def summarize_permutations(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    by_seed: Dict[object, List[Mapping[str, object]]] = {}
    for r in rows:
        by_seed.setdefault(r['order_seed'], []).append(r)
    cells: Dict[tuple, List[Mapping[str, object]]] = {}
    for seed_rows in by_seed.values():
        for s in summarize(seed_rows):
            cells.setdefault((s['split'], s['price'], s['demand_per_year'], s['budget_eur'], s['policy_id']), []).append(s)
    out = []
    for (split, price, demand, budget, policy), ss in sorted(cells.items()):
        npv = [s['mean_npv_eur'] for s in ss]
        row = {'split': split, 'price': price, 'demand_per_year': demand, 'budget_eur': budget, 'policy_id': policy, 'n_permutations': len(ss), 'n_tasks': ss[0]['n_tasks'], 'mean_npv_eur': statistics.mean(npv), 'q05_npv_eur': _quantile(npv, 0.05), 'q95_npv_eur': _quantile(npv, 0.95), 'mean_oracle_npv_eur': statistics.mean((s['mean_oracle_npv_eur'] for s in ss)), 'mean_cost_eur': statistics.mean((s['mean_cost_eur'] for s in ss))}
        for ref in ('P5_COST_HEURISTIC', 'P8_COST_AWARE_BO_STOP', 'P0_STOP'):
            diffs = [s[f'mean_diff_vs_{ref}'] for s in ss]
            row[f'mean_diff_vs_{ref}'] = statistics.mean(diffs)
            row[f'share_perm_ge_{ref}'] = sum((d >= -1e-12 for d in diffs)) / len(diffs)
        out.append(row)
    return out

def opportunity_means(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    cells: Dict[tuple, List[float]] = {}
    for r in rows:
        key = (r['split'], r['price'], r['demand_per_year'], r['budget_eur'], r['task_id'], r['policy_id'])
        cells.setdefault(key, []).append(float(r['npv_eur']))
    return [{'split': k[0], 'price': k[1], 'demand_per_year': k[2], 'budget_eur': k[3], 'task_id': k[4], 'policy_id': k[5], 'n_permutations': len(v), 'mean_npv_eur': statistics.mean(v)} for k, v in sorted(cells.items())]
