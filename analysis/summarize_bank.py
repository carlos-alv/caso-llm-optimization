import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

def _f(x):
    return None if x in ('', 'None', None) else float(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', required=True, help='bank_attempts.csv')
    ap.add_argument('--scenarios', required=True)
    ap.add_argument('--scenario-id', default='pilot-fullyloaded-gpu-median')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    scen = {s['scenario_id']: s for s in json.loads(Path(a.scenarios).read_text())}[a.scenario_id]
    gpu_h = scen['rates']['gpu_hour_fully_loaded_eur']['value']
    cpu_h = scen['rates']['cpu_core_hour_fully_loaded_eur']['value']
    rows = list(csv.DictReader(open(a.bank)))
    by = defaultdict(list)
    for r in rows:
        by[r['model_key'], r['split']].append(r)
        by[r['model_key'], 'all'].append(r)
    out = []
    for (model, split), rs in sorted(by.items()):
        valid = [r for r in rs if r['valid'] == 'True']
        conf = [r for r in rs if r['improvement_confirmed'] == 'True']
        sp_valid = [_f(r['speedup']) for r in valid if _f(r['speedup'])]
        cost = [_f(r['gpu_s']) / 3600 * gpu_h + _f(r['cpu_core_s']) / 3600 * cpu_h for r in rs]
        energy = [_f(r['gpu_energy_wh']) for r in rs if _f(r['gpu_energy_wh']) is not None]
        inst = sorted({r['instance_id'] for r in rs})
        inst_conf = sorted({r['instance_id'] for r in conf})
        status = defaultdict(int)
        for r in rs:
            status[r['status']] += 1
        out.append({'model': model, 'split': split, 'n_attempts': len(rs), 'n_instances': len(inst), 'generation_failed': status['generation_failed'] + status['timeout'], 'patch_invalid': status['patch_invalid'], 'rejected_by_gate': status['rejected_by_gate'], 'valid': len(valid), 'confirmed': len(conf), 'valid_but_slower': sum((1 for s in sp_valid if s < 1.0)), 'instances_with_confirmed': len(inst_conf), 'median_speedup_valid': statistics.median(sp_valid) if sp_valid else None, 'max_speedup_confirmed': max((_f(r['speedup']) for r in conf), default=None), 'median_gpu_s': statistics.median((_f(r['gpu_s']) for r in rs)), 'median_cpu_core_s': statistics.median((_f(r['cpu_core_s']) for r in rs)), 'median_gpu_energy_wh': statistics.median(energy) if energy else None, 'n_energy_missing': len(rs) - len(energy), 'total_cost_eur': sum(cost), 'cost_per_confirmed_eur': sum(cost) / len(conf) if conf else None})
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        if r['split'] == 'all':
            print(json.dumps(r))
if __name__ == '__main__':
    main()
