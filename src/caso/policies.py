from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence
from .decision import Action, CostModel, ModelPrior, OpportunityState, expected_improvement_eur, posterior_for

@dataclass(frozen=True)
class Candidate:
    action: Action
    feasible: bool
    p_valid: Optional[float] = None
    expected_gain_eur: Optional[float] = None
    expected_cost_eur: Optional[float] = None
    score: Optional[float] = None

@dataclass(frozen=True)
class PolicyDecision:
    action: Optional[Action]
    reason: str
    candidates: List[Candidate] = field(default_factory=list)
    expected_incremental_value_eur: float = 0.0

    @property
    def is_stop(self) -> bool:
        return self.action is None

class Policy:
    policy_id = 'ABSTRACT'
    revision = '0.1'

    def __init__(self, cost_model: CostModel, priors: Mapping[str, ModelPrior]) -> None:
        self.cost_model = cost_model
        self.priors = dict(priors)

    def _feasible(self, state: OpportunityState) -> List[Candidate]:
        out = []
        for action in state.actions:
            cost = self.cost_model.expected_cost_eur(action, state.history)
            out.append(Candidate(action, cost <= state.remaining_budget_eur, expected_cost_eur=cost))
        return out

    def decide(self, state: OpportunityState) -> PolicyDecision:
        raise NotImplementedError

def _stop(reason: str, candidates: Sequence[Candidate]=()) -> PolicyDecision:
    return PolicyDecision(None, reason, list(candidates))

class P0Stop(Policy):
    policy_id = 'P0_STOP'

    def decide(self, state):
        return _stop('never optimize (economic reference)')

class _FixedModel(Policy):
    size_class = ''

    def decide(self, state):
        cands = self._feasible(state)
        usable = [c for c in cands if c.feasible and c.action.model_key in self.models()]
        if not usable:
            return _stop('budget exhausted for the fixed model', cands)
        best = max(usable, key=lambda c: c.action.hotspot_share)
        return PolicyDecision(best.action, f'fixed {self.size_class} model, retry until budget', cands)

    def models(self):
        return {k for k, p in self.priors.items() if p.size_class == self.size_class}

class P1FixedSmall(_FixedModel):
    policy_id, size_class = ('P1_FIXED_SMALL', 'small')

class P2FixedLarge(_FixedModel):
    policy_id, size_class = ('P2_FIXED_LARGE', 'large')

class P3PerformanceFirst(Policy):
    policy_id = 'P3_PERFORMANCE_FIRST'

    def decide(self, state):
        cands = self._feasible(state)
        scored = []
        for c in cands:
            validity, log_speedup = posterior_for(self.priors[c.action.model_key], state.history, c.action.model_key)
            scored.append(Candidate(c.action, c.feasible, validity.mean, None, c.expected_cost_eur, validity.mean * log_speedup.mean))
        usable = [c for c in scored if c.feasible]
        if not usable:
            return _stop('budget exhausted', scored)
        best = max(usable, key=lambda c: (c.score, c.action.hotspot_share))
        return PolicyDecision(best.action, 'highest expected speedup', scored)

class P4RandomRouter(Policy):
    policy_id = 'P4_RANDOM_ROUTER'

    def __init__(self, cost_model, priors, seed: int) -> None:
        super().__init__(cost_model, priors)
        self.seed = seed

    def decide(self, state):
        cands = self._feasible(state)
        usable = [c for c in cands if c.feasible]
        if not usable:
            return _stop('budget exhausted', cands)
        rng = random.Random(f'{self.seed}:{state.task_id}:{state.step}')
        return PolicyDecision(rng.choice(usable).action, 'uniform random feasible action', cands)

class P5CostHeuristic(Policy):
    policy_id = 'P5_COST_HEURISTIC'

    def __init__(self, cost_model, priors, failures_before_escalation: int=2, max_attempts: int=6) -> None:
        super().__init__(cost_model, priors)
        self.k = failures_before_escalation
        self.max_attempts = max_attempts

    def decide(self, state):
        cands = self._feasible(state)
        if any((o.improvement_confirmed for o in state.history)):
            return _stop('satisficed: confirmed improvement found', cands)
        if state.step >= self.max_attempts:
            return _stop('cascade exhausted', cands)
        usable = sorted((c for c in cands if c.feasible), key=lambda c: c.expected_cost_eur)
        if not usable:
            return _stop('budget exhausted', cands)
        failures: Dict[str, int] = {}
        for o in state.history:
            failures[o.model_key] = failures.get(o.model_key, 0) + (not o.improvement_confirmed)
        for c in usable:
            if failures.get(c.action.model_key, 0) < self.k:
                same_model = [u for u in usable if u.action.model_key == c.action.model_key]
                best = max(same_model, key=lambda u: u.action.hotspot_share)
                return PolicyDecision(best.action, 'cheapest model not yet exhausted', cands)
        return _stop('every model failed k times', cands)

class P6CasoVOI(Policy):
    policy_id = 'P6_CASO_ADAPTIVE'
    use_amdahl = True

    def __init__(self, cost_model, priors, risk_lambda: float=0.0, mc_samples: int=4000) -> None:
        super().__init__(cost_model, priors)
        self.risk_lambda = risk_lambda
        self.mc_samples = mc_samples

    def decide(self, state):
        scored = []
        for c in self._feasible(state):
            validity, log_speedup = posterior_for(self.priors[c.action.model_key], state.history, c.action.model_key)
            gain, sd = expected_improvement_eur(state, validity.mean, log_speedup, amdahl_share=c.action.hotspot_share if self.use_amdahl else None, n_samples=self.mc_samples, seed=state.step)
            margin = gain - c.expected_cost_eur - self.risk_lambda * sd
            scored.append(Candidate(c.action, c.feasible, validity.mean, gain, c.expected_cost_eur, margin))
        usable = [c for c in scored if c.feasible]
        if not usable:
            return _stop('no action within remaining budget', scored)
        best = max(usable, key=lambda c: c.score)
        if best.score <= 0:
            return _stop('no action with positive expected marginal value', scored)
        return PolicyDecision(best.action, 'max expected marginal value', scored, best.score)

class P6CasoNoAmdahl(P6CasoVOI):
    policy_id = 'P6A_CASO_NO_AMDAHL'
    use_amdahl = False

class P7SysLLMaticAdapted(Policy):
    policy_id = 'P7_SYSLMMATIC_REPLICA'

    def __init__(self, cost_model, priors, steps: int=2) -> None:
        super().__init__(cost_model, priors)
        self.steps = steps

    def decide(self, state):
        cands = self._feasible(state)
        if state.step >= self.steps:
            return _stop(f'{self.steps} self-optimization steps done', cands)
        large = {k for k, p in self.priors.items() if p.size_class == 'large'}
        usable = [c for c in cands if c.feasible and c.action.model_key in large]
        if not usable:
            return _stop('budget exhausted', cands)
        best = max(usable, key=lambda c: (c.action.hotspot_share, -c.expected_cost_eur))
        return PolicyDecision(best.action, 'self-optimization step', cands)

class P8CostAwareBOStop(Policy):
    policy_id = 'P8_COST_AWARE_BO_STOP'

    def __init__(self, cost_model, priors, mc_samples: int=4000) -> None:
        super().__init__(cost_model, priors)
        self.mc_samples = mc_samples

    def decide(self, state):
        scored = []
        for c in self._feasible(state):
            validity, log_speedup = posterior_for(self.priors[c.action.model_key], state.history, c.action.model_key)
            gain, _ = expected_improvement_eur(state, validity.mean, log_speedup, amdahl_share=None, n_samples=self.mc_samples, seed=state.step)
            ratio = gain / c.expected_cost_eur if c.expected_cost_eur > 0 else float('inf')
            scored.append(Candidate(c.action, c.feasible, validity.mean, gain, c.expected_cost_eur, ratio))
        usable = [c for c in scored if c.feasible]
        if not usable:
            return _stop('no action within remaining budget', scored)
        if all((c.expected_gain_eur <= c.expected_cost_eur for c in usable)):
            return _stop('Xie-style rule: E[gain] <= E[cost] for every action', scored)
        best = max(usable, key=lambda c: c.score)
        return PolicyDecision(best.action, 'max E[gain]/E[cost]', scored, best.expected_gain_eur - best.expected_cost_eur)
