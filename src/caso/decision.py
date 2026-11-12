from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

@dataclass(frozen=True)
class Action:
    action_id: str
    model_key: str
    hotspot_id: str
    hotspot_share: float
    max_tokens: int

@dataclass(frozen=True)
class Observation:
    action_id: str
    model_key: str
    status: str
    valid: bool
    speedup: Optional[float]
    improvement_confirmed: bool
    cost_eur: float
    gpu_s: float
    cpu_core_s: float

@dataclass(frozen=True)
class OpportunityState:
    task_id: str
    base_cost_per_exec_eur: float
    value_multiplier: float
    actions: Tuple[Action, ...]
    history: Tuple[Observation, ...]
    remaining_budget_eur: float
    best_saving_per_exec_eur: float = 0.0

    @property
    def step(self) -> int:
        return len(self.history)

    @property
    def best_value_eur(self) -> float:
        return self.best_saving_per_exec_eur * self.value_multiplier

@dataclass
class BetaPosterior:
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def updated(self, success: bool) -> 'BetaPosterior':
        return BetaPosterior(self.alpha + success, self.beta + (not success))

@dataclass
class NormalPosterior:
    mean: float
    var: float
    obs_sd: float

    def updated(self, x: float) -> 'NormalPosterior':
        precision = 1.0 / self.var + 1.0 / self.obs_sd ** 2
        mean = (self.mean / self.var + x / self.obs_sd ** 2) / precision
        return NormalPosterior(mean, 1.0 / precision, self.obs_sd)

    @property
    def predictive_sd(self) -> float:
        return math.sqrt(self.var + self.obs_sd ** 2)

@dataclass
class ModelPrior:
    validity: BetaPosterior
    log_speedup: NormalPosterior
    mean_gpu_s: float
    mean_cpu_core_s: float
    size_class: str

@dataclass(frozen=True)
class Prices:
    gpu_hour_eur: float
    cpu_core_hour_eur: float

class CostModel:

    def __init__(self, priors: Mapping[str, ModelPrior], prices: Prices) -> None:
        self.priors = dict(priors)
        self.prices = prices

    def expected_cost_eur(self, action: Action, history: Sequence[Observation]) -> float:
        prior = self.priors[action.model_key]
        own = [o for o in history if o.model_key == action.model_key]
        gpu = (prior.mean_gpu_s * 2 + sum((o.gpu_s for o in own))) / (2 + len(own))
        cpu = (prior.mean_cpu_core_s * 2 + sum((o.cpu_core_s for o in own))) / (2 + len(own))
        return gpu / 3600 * self.prices.gpu_hour_eur + cpu / 3600 * self.prices.cpu_core_hour_eur

def posterior_for(prior: ModelPrior, history: Sequence[Observation], model_key: str) -> Tuple[BetaPosterior, NormalPosterior]:
    validity, log_speedup = (prior.validity, prior.log_speedup)
    for o in history:
        if o.model_key != model_key:
            continue
        validity = validity.updated(o.valid)
        if o.valid and o.speedup and (o.speedup > 0):
            log_speedup = log_speedup.updated(math.log(o.speedup))
    return (validity, log_speedup)

def expected_improvement_eur(state: OpportunityState, p_valid: float, log_speedup: NormalPosterior, *, amdahl_share: Optional[float], n_samples: int=4000, seed: int=0) -> Tuple[float, float]:
    rng = random.Random(seed)
    cap = math.inf if amdahl_share is None or amdahl_share >= 1 else 1 / (1 - amdahl_share)
    c0, best = (state.base_cost_per_exec_eur, state.best_value_eur)
    gains = []
    for _ in range(n_samples):
        speedup = min(math.exp(rng.gauss(log_speedup.mean, log_speedup.predictive_sd)), cap)
        value = c0 * (1 - 1 / speedup) * state.value_multiplier
        gains.append(max(0.0, value - best))
    mean = sum(gains) / n_samples
    var = sum(((g - mean) ** 2 for g in gains)) / max(1, n_samples - 1)
    return (p_valid * mean, math.sqrt(max(0.0, p_valid * (var + mean ** 2) - (p_valid * mean) ** 2)))

def annuity(rate: float, years: int) -> float:
    if rate < 0 or years < 1:
        raise ValueError('rate >= 0 and years >= 1 required')
    return sum((1 / (1 + rate) ** t for t in range(1, years + 1)))
