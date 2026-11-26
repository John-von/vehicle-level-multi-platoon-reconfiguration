"""Exact joint assignment and interleaving by dynamic programming without truncation.

For every seed the three objectives are solved without beam truncation, and the
result is compared with the plan of the beam search of width 1600 stored in
results/planning/plans, which gives the optimality gap on the same scenarios.
Usage: python plan_exact.py --seeds 0 99
"""
from pathlib import Path
import json, argparse
from time import perf_counter

from scenario import Config, make_scenario, jsonable
from sequence import optimize_joint
from plan_methods import metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'results'/'planning'


def main(first, last, methods):
    cfg = Config(); rows = []
    pdir = OUT/'plans_exact'; pdir.mkdir(parents=True, exist_ok=True)
    for seed in range(first, last+1):
        sc = make_scenario(seed, cfg)
        for method in methods:
            f = pdir/f's{seed}_{method}.json'
            if f.exists():                      # reuse a finished exact plan
                r = json.loads(f.read_text(encoding='utf-8'))['assignment']
            else:
                t0 = perf_counter(); r = optimize_joint(sc, method, cfg, beam_width=10**9); r['seconds'] = perf_counter()-t0
                assert not r['pruning_used']
                r['method'] = method; r['solver'] = 'exact joint dynamic programming (no pruning)'; r['optimal'] = True
            row = metrics(sc, r, cfg); row.update(seed=seed, n=24, transitions=r['transitions'])
            beam_file = OUT/'plans'/f's{seed}_{method}.json'
            if beam_file.exists():
                b = json.loads(beam_file.read_text(encoding='utf-8'))['assignment']
                row['beam_objective'] = b['optimized_objective']
                row['beam_gap'] = (b['optimized_objective']-r['optimized_objective'])/abs(r['optimized_objective'])
                row['same_plan'] = (b['order'] == r['order'] and b['destination'] == r['destination'])
            rows.append(row)
            if not f.exists():
                f.write_text(json.dumps(dict(scenario=jsonable(sc), assignment=jsonable(r))), encoding='utf-8')
        print('seed', seed, 'done', flush=True)
    (OUT/f'exact_main_{first}-{last}.json').write_text(json.dumps(jsonable(rows), indent=1), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--seeds', type=int, nargs=2, default=[0, 99])
    p.add_argument('--methods', nargs='+', default=['motion_only', 'motion_speed', 'combined'])
    a = p.parse_args(); main(a.seeds[0], a.seeds[1], a.methods)
