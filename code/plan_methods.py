"""Planning-layer study.

Methods (all share the same scenario generator, slot geometry and cost model):
  source_blocks  fragment/block-based regrouping (A + first half of B -> lane 0)
  mda            minimum-displacement lane assignment (Hungarian, capacity 12/12)
                 with position-sorted interleaving (nearest-slot insertion)
  motion_only    joint assignment-ordering search, motion surrogate only
  motion_speed   joint search, motion + speed matching
  combined       joint search, motion + speed + communication adjacency (proposed)
Exact solutions (no beam pruning) are computed for the optimality-gap study.
"""
from pathlib import Path
import json, argparse
from dataclasses import replace
from time import perf_counter
import numpy as np
from scipy.optimize import linear_sum_assignment

from scenario import Config, make_scenario, jsonable
from sequence import optimize_joint, evaluate

ROOT = Path(__file__).resolve().parent.parent
METHODS = ['source_blocks', 'mda', 'motion_only', 'motion_speed', 'combined']


def plan_source_blocks(sc, cfg):
    n = len(sc['labels']); m = cfg.vehicles_per_source
    dest = np.repeat([0, 0, 1], m); dest[m+m//2:2*m] = 1
    r = evaluate(dict(sc, route=dest), list(range(n)), cfg)
    r.update(destination=dest.tolist(), method='source_blocks', solver='fixed source-block baseline', optimal=False)
    return r


def plan_mda(sc, cfg):
    """Hungarian lane assignment on final slots, then position-sorted order."""
    n = len(sc['labels']); cap = n//2
    x0 = np.array(sc['initial'])[:, 0]; src = np.array(sc['source'])
    slots = [(d, r) for d in range(2) for r in range(cap)]
    cost = np.empty((n, n))
    for k, (d, r) in enumerate(slots):
        dx = sc['target_x'][r]-x0
        dy = (d-src)*cfg.lane_width
        # same normalisation as the motion surrogate (direct transfer)
        cost[:, k] = 12*dx**2/(cfg.stage_time+cfg.compact_time)**3+12*dy**2/cfg.crossing_time**3
    rows, cols = linear_sum_assignment(cost)
    dest = np.empty(n, dtype=int); dest[rows] = [slots[c][0] for c in cols]
    order = np.argsort(-x0, kind='stable').tolist()
    r = evaluate(dict(sc, route=dest), order, cfg)
    r.update(destination=dest.tolist(), method='mda', solver='Hungarian lane assignment + position-sorted interleaving', optimal=False)
    return r


def plan(sc, method, cfg, beam=1600):
    t0 = perf_counter()
    if method == 'source_blocks': r = plan_source_blocks(sc, cfg)
    elif method == 'mda': r = plan_mda(sc, cfg)
    else:
        r = optimize_joint(sc, 'motion_speed' if method == 'motion_speed' else method, cfg, beam_width=beam)
        r['method'] = method
    r['seconds'] = perf_counter()-t0
    return r


def metrics(sc, r, cfg):
    q = np.array(sc['quality']); links = r['links']; lq = np.array([q[a, b] for a, b in links])
    dest = np.array(r['destination']); src = np.array(sc['source'])
    w = np.array([1., cfg.weight_speed, cfg.weight_link])
    return dict(method=r['method'], objective=float(np.array(r['terms'])@w), motion=r['terms'][0],
                speed=r['terms'][1], link=r['terms'][2], mean_q=float(lq.mean()), min_q=float(lq.min()),
                log_all_links=float(np.log(lq).sum()), expected_losses=float((1-lq).sum()),
                lane_changes=int((dest != src).sum()), source_switches=r['source_switches'], seconds=r['seconds'])


def study(seeds, m=8, beam=1600, out=None):
    cfg = replace(Config(), vehicles_per_source=m)
    rows = []; plans = {}
    for seed in seeds:
        sc = make_scenario(seed, cfg)
        for method in METHODS:
            r = plan(sc, method, cfg, beam)
            row = metrics(sc, r, cfg); row.update(seed=seed, n=3*m); rows.append(row)
            plans[(seed, method)] = dict(scenario=jsonable(sc), assignment=jsonable(r))
    return rows, plans


def exact_gap(seeds, m, beam=1600):
    cfg = replace(Config(), vehicles_per_source=m); rows = []
    for seed in seeds:
        sc = make_scenario(seed, cfg)
        t0 = perf_counter(); b = optimize_joint(sc, 'combined', cfg, beam_width=beam); tb = perf_counter()-t0
        t0 = perf_counter(); e = optimize_joint(sc, 'combined', cfg, beam_width=10**9); te = perf_counter()-t0
        assert not e['pruning_used']
        rows.append(dict(seed=seed, n=3*m, beam=beam, beam_objective=b['optimized_objective'], exact_objective=e['optimized_objective'],
                         gap=(b['optimized_objective']-e['optimized_objective'])/e['optimized_objective'],
                         beam_seconds=tb, exact_seconds=te, exact_transitions=e['transitions'], beam_transitions=b['transitions']))
        print('exact', 3*m, seed, rows[-1]['gap'], round(tb, 2), round(te, 1), flush=True)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['main', 'exact', 'scale', 'beam', 'weights'], default='main')
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(100)))
    p.add_argument('--m', type=int, nargs='+', default=[8])
    p.add_argument('--out', default=str(ROOT/'results/planning'))
    a = p.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.task == 'main':
        rows, plans = study(a.seeds)
        (out/f'planning_main_{a.seeds[0]}-{a.seeds[-1]}.json').write_text(json.dumps(jsonable(rows), indent=1), encoding='utf-8')
        pdir = out/'plans'; pdir.mkdir(exist_ok=True)
        for (seed, method), pl in plans.items():
            (pdir/f's{seed}_{method}.json').write_text(json.dumps(pl), encoding='utf-8')
        for method in METHODS:
            rr = [r for r in rows if r['method'] == method]
            print(method, {k: round(float(np.mean([r[k] for r in rr])), 4) for k in ['objective', 'motion', 'mean_q', 'min_q', 'expected_losses', 'lane_changes', 'source_switches', 'seconds']})
    elif a.task == 'exact':
        rows = []
        for m in a.m: rows += exact_gap(a.seeds, m)
        tag = '_'.join(map(str, a.m))+f'_s{a.seeds[0]}-{a.seeds[-1]}'
        (out/f'exact_gap_m{tag}.json').write_text(json.dumps(jsonable(rows), indent=1), encoding='utf-8')
    elif a.task == 'scale':
        rows = []
        for m in a.m:
            r, _ = study(a.seeds, m)
            rows += r
            print('scale', 3*m, {meth: round(float(np.mean([x['seconds'] for x in r if x['method'] == meth])), 3) for meth in METHODS}, flush=True)
        (out/f"scale_m{'_'.join(map(str, a.m))}_s{a.seeds[0]}-{a.seeds[-1]}.json").write_text(json.dumps(jsonable(rows), indent=1), encoding='utf-8')
    elif a.task == 'beam':
        cfg = Config(); rows = []
        for seed in a.seeds:
            sc = make_scenario(seed, cfg)
            for w in [50, 100, 200, 400, 800, 1600, 3200, 6400]:
                t0 = perf_counter(); r = optimize_joint(sc, 'combined', cfg, beam_width=w)
                rows.append(dict(seed=seed, beam=w, objective=r['optimized_objective'], seconds=perf_counter()-t0, transitions=r['transitions']))
        (out/'beam_sensitivity.json').write_text(json.dumps(jsonable(rows), indent=1), encoding='utf-8')
    elif a.task == 'weights':
        cfg = Config(); rows = []
        for seed in a.seeds:
            sc = make_scenario(seed, cfg)
            for w in [0., .05, .1, .2, .4, .8, 1.6, 3.2]:
                r = optimize_joint(sc, 'combined', cfg, beam_width=1600, link_weight=w)
                r['method'] = 'combined'; r['seconds'] = 0.
                row = metrics(sc, r, cfg); row.update(seed=seed, weight=w); rows.append(row)
        (out/'weight_sweep.json').write_text(json.dumps(jsonable(rows), indent=1), encoding='utf-8')
