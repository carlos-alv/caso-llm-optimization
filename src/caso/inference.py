from __future__ import annotations
import random
import statistics
from typing import Dict, List, Mapping, Sequence

def _pooled_mean(groups: Sequence[Sequence[float]]) -> float:
    values = [x for g in groups for x in g]
    return statistics.mean(values)

def hierarchical_bootstrap(diffs: Mapping[str, Sequence[float]], *, reps: int=5000, seed: int=20261027) -> Dict[str, object]:
    repos = sorted((r for r, d in diffs.items() if d))
    if not repos:
        raise ValueError('no differences')
    rng = random.Random(seed)
    estimates = []
    for _ in range(reps):
        groups = []
        for repo in rng.choices(repos, k=len(repos)):
            d = diffs[repo]
            groups.append(rng.choices(d, k=len(d)))
        estimates.append(_pooled_mean(groups))
    estimates.sort()
    below = sum((e <= 0 for e in estimates))
    above = sum((e >= 0 for e in estimates))
    p_greater = (1 + below) / (reps + 1)
    p_less = (1 + above) / (reps + 1)
    return {'estimate': _pooled_mean([diffs[r] for r in repos]), 'ci95': [estimates[int(0.025 * reps)], estimates[int(0.975 * reps) - 1]], 'p_one_sided_greater': p_greater, 'p_two_sided': min(1.0, 2 * min(p_greater, p_less)), 'reps': reps, 'n_repos': len(repos), 'n_opportunities': sum((len(diffs[r]) for r in repos))}

def leave_one_repo_out(diffs: Mapping[str, Sequence[float]]) -> Dict[str, float]:
    repos = sorted((r for r, d in diffs.items() if d))
    return {r: _pooled_mean([diffs[o] for o in repos if o != r]) for r in repos if len(repos) > 1}

def holm(pvalues: Mapping[str, float], alpha: float=0.05) -> Dict[str, bool]:
    order = sorted(pvalues, key=lambda k: pvalues[k])
    m, out, still = (len(order), {}, True)
    for i, name in enumerate(order):
        still = still and pvalues[name] <= alpha / (m - i)
        out[name] = still
    return out
