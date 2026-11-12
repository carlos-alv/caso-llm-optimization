from __future__ import annotations
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from .decision import Action, BetaPosterior, ModelPrior, NormalPosterior, Observation, Prices
from .patch_review import persistent_state_flags

@dataclass(frozen=True)
class BankAttempt:
    candidate_id: str
    instance_id: str
    split: str
    model_key: str
    hotspot_id: str
    hotspot_share: float
    seed: int
    status: str
    valid: bool
    speedup: Optional[float]
    improvement_confirmed: bool
    gpu_s: float
    cpu_core_s: float
    tokens_out: Optional[int]
    gpu_energy_wh: Optional[float]

    def action_id(self) -> str:
        return f'{self.model_key}:{self.hotspot_id}'

    def observation(self, prices: Prices) -> Observation:
        cost = self.gpu_s / 3600 * prices.gpu_hour_eur + self.cpu_core_s / 3600 * prices.cpu_core_hour_eur
        return Observation(self.action_id(), self.model_key, self.status, self.valid, self.speedup, self.improvement_confirmed, cost, self.gpu_s, self.cpu_core_s)

def _stage_cpu(ev: Mapping) -> float:
    total = 0.0
    for key in ('apply', 'tests'):
        if key in ev:
            total += ev[key]['cpu_core_s']
    for b in ev.get('bench', []):
        total += b['cpu_core_s']
    re = ev.get('remeasure')
    if re:
        total += re['apply']['cpu_core_s'] + sum((b['cpu_core_s'] for b in re['bench']))
    return total

def load_bank(candidate_files: Iterable[Path], outcome_files: Iterable[Path], *, strict_format: bool=False, rejected_by_review: Iterable[str]=(), reject_persistent_state: bool=False, instance_thresholds: Optional[Mapping[str, float]]=None) -> List[BankAttempt]:
    rejected = set(rejected_by_review)
    outcomes = {}
    for f in outcome_files:
        for line in Path(f).read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                outcomes[o['candidate_id']] = o
    cands = [json.loads(l) for f in candidate_files for l in Path(f).read_text().splitlines() if l.strip()]
    per_instance = {}
    for c in cands:
        per_instance[c['instance_id']] = per_instance.get(c['instance_id'], 0) + 1
    attempts = []
    for c in cands:
        o = outcomes.get(c['candidate_id'])
        if o is None:
            continue
        gen = c['generation']
        extract_share = o['extract_stage_shared']['cpu_core_s'] / per_instance[c['instance_id']]
        cpu = extract_share + _stage_cpu(o)
        status = gen['status'] if gen['status'] != 'ok' else o['status']
        if gen['status'] != 'ok':
            status = 'generation_failed' if gen['status'] == 'error' else 'timeout'
        energy = c['energy_record']['energy_wh'].get('value')
        sp = o.get('speedup_final') or (o.get('speedup') or {}).get('speedup')
        valid, confirmed = (bool(o.get('valid')), bool(o.get('improvement_confirmed')))
        if strict_format and gen['status'] == 'ok' and (c.get('format_mode') == 'function_rewrite'):
            status, valid, sp, confirmed, cpu = ('patch_invalid', False, None, False, extract_share)
        if c['candidate_id'] in rejected and valid:
            status, valid, sp, confirmed = ('rejected_by_review', False, None, False)
        if instance_thresholds and confirmed and (c['instance_id'] in instance_thresholds):
            th = instance_thresholds[c['instance_id']]
            stage2 = ((o.get('remeasure') or {}).get('speedup') or {}).get('speedup')
            if o['speedup']['speedup'] < th or (stage2 is not None and stage2 < th):
                confirmed = False
        if reject_persistent_state and valid and persistent_state_flags(c.get('diff') or ''):
            status, valid, sp, confirmed = ('rejected_persistent_state', False, None, False)
        attempts.append(BankAttempt(c['candidate_id'], c['instance_id'], c['split'], c['model_key'], c['hotspot']['hotspot_id'], float(c['hotspot']['share']), int(c['seed']), status, valid, sp, confirmed, float(gen['latency_s']), cpu, gen.get('tokens_out'), energy))
    return attempts

def priors_from(attempts: Sequence[BankAttempt], size_class: Mapping[str, str], *, prior_validity=(1.0, 1.0), log_speedup_prior=(0.0, 0.25), obs_sd: float=0.3) -> Dict[str, ModelPrior]:
    out = {}
    for model, cls in size_class.items():
        own = [a for a in attempts if a.model_key == model]
        valid = [a for a in own if a.valid]
        logs = [math.log(a.speedup) for a in valid if a.speedup and a.speedup > 0]
        mu0, var0 = log_speedup_prior
        posterior = NormalPosterior(mu0, var0, obs_sd)
        for x in logs:
            posterior = posterior.updated(x)
        out[model] = ModelPrior(BetaPosterior(prior_validity[0] + len(valid), prior_validity[1] + len(own) - len(valid)), posterior, statistics.mean([a.gpu_s for a in own]) if own else 60.0, statistics.mean([a.cpu_core_s for a in own]) if own else 600.0, cls)
    return out

def actions_for(attempts: Sequence[BankAttempt], instance_id: str, max_tokens: int=4096) -> Tuple[Action, ...]:
    seen = {}
    for a in attempts:
        if a.instance_id == instance_id:
            seen[a.action_id()] = Action(a.action_id(), a.model_key, a.hotspot_id, a.hotspot_share, max_tokens)
    return tuple(sorted(seen.values(), key=lambda x: x.action_id))

def bank_for(attempts: Sequence[BankAttempt], instance_id: str, prices: Prices, order_seed: Optional[int]=None) -> Dict[str, List[Observation]]:
    bank: Dict[str, List[Observation]] = {}
    for a in sorted((a for a in attempts if a.instance_id == instance_id), key=lambda a: a.seed):
        bank.setdefault(a.action_id(), []).append(a.observation(prices))
    if order_seed is not None:
        for action_id, observations in bank.items():
            random.Random(f'{order_seed}:{instance_id}:{action_id}').shuffle(observations)
    return bank
