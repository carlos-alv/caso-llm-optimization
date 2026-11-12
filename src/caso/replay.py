from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence
from .decision import Observation, OpportunityState
from .policies import Policy

@dataclass(frozen=True)
class ReplayResult:
    policy_id: str
    task_id: str
    steps: int
    stop_reason: str
    censored: bool
    total_cost_eur: float
    best_saving_per_exec_eur: float
    value_eur: float
    npv_eur: float
    actions: List[str] = field(default_factory=list)
    censor_events: int = 0

def verified_saving_per_exec(base_cost_per_exec_eur: float, attempt: Observation) -> float:
    if not attempt.improvement_confirmed or not attempt.speedup or attempt.speedup <= 1:
        return 0.0
    return base_cost_per_exec_eur * (1 - 1 / attempt.speedup)

def replay(policy: Policy, initial: OpportunityState, bank: Mapping[str, Sequence[Observation]], *, max_steps: int=50) -> ReplayResult:
    used: Dict[str, int] = {}
    history: List[Observation] = []
    state = initial
    cost, best = (0.0, initial.best_saving_per_exec_eur)
    stop_reason, actions, censor_events = ('max steps reached', [], 0)
    available = tuple(initial.actions)
    steps_taken = 0
    while steps_taken < max_steps:
        decision = policy.decide(state)
        if decision.is_stop:
            stop_reason = decision.reason
            break
        action_id = decision.action.action_id
        k = used.get(action_id, 0)
        attempts = bank.get(action_id, ())
        if k >= len(attempts):
            censor_events += 1
            available = tuple((a for a in available if a.action_id != action_id))
            state = OpportunityState(initial.task_id, initial.base_cost_per_exec_eur, initial.value_multiplier, available, tuple(history), state.remaining_budget_eur, best)
            if not available:
                stop_reason = 'bank exhausted for every action'
                break
            continue
        steps_taken += 1
        attempt = attempts[k]
        used[action_id] = k + 1
        actions.append(action_id)
        history.append(attempt)
        cost += attempt.cost_eur
        best = max(best, verified_saving_per_exec(initial.base_cost_per_exec_eur, attempt))
        state = OpportunityState(initial.task_id, initial.base_cost_per_exec_eur, initial.value_multiplier, available, tuple(history), max(0.0, initial.remaining_budget_eur - cost), best)
    value = best * initial.value_multiplier
    return ReplayResult(policy.policy_id, initial.task_id, len(history), stop_reason, censor_events > 0, cost, best, value, value - cost, actions, censor_events)
