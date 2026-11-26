"""Resumable batch runner for the closed-loop experiments.

Each job writes results/<group>/<name>/summary.json; existing outputs are
skipped, so batches can be split across workers and restarted safely.
"""
import json, argparse, traceback
import numpy as np

from run_exec import run, ROOT, COMFORT
from plan_methods import plan
from dataclasses import replace
from scenario import Config, make_scenario, jsonable

OUT = ROOT/'results'


def rear_layout(sc):
    """Staging and target grids shifted forward so that the last staging position coincides with the rearmost vehicle."""
    x_rear = float(np.min(np.array(sc['initial'])[:, 0]))
    shift = x_rear-sc['stage_x'][-1]
    sc['stage_x'] = np.asarray(sc['stage_x'])+shift
    sc['target_x'] = np.asarray(sc['target_x'])+shift
    return sc


def plan_file(seed, method, m=8, layout='center'):
    folder = OUT/'planning'/('plans' if m == 8 else f'plans_m{m}'); folder.mkdir(parents=True, exist_ok=True)
    f = folder/(f's{seed}_{method}.json' if layout == 'center' else f's{seed}_{method}_{layout}.json')
    if not f.exists():
        cfg = replace(Config(), vehicles_per_source=m); sc = make_scenario(seed, cfg)
        if layout == 'rear': sc = rear_layout(sc)
        r = plan(sc, method, cfg)
        f.write_text(json.dumps(dict(scenario=jsonable(sc), assignment=jsonable(r))), encoding='utf-8')
    return json.loads(f.read_text(encoding='utf-8'))


def jobs(group):
    J = []
    pol = [('barrier', False), ('eager', False), ('settled', True)]
    if group == 'exec_adaptive':      # main policy comparison, comfort-limited timing
        for seed in range(20):
            for name, settle in pol:
                J.append(dict(group=group, name=f's{seed}_{name}', seed=seed, method='combined',
                              policy='barrier' if name == 'barrier' else 'event', settle=settle, timing='adaptive'))
    elif group == 'exec_rear':        # staging grid anchored at the rear of the group, target grid shifted with it
        for seed in range(10):
            J.append(dict(group=group, name=f's{seed}_settled', seed=seed, method='combined', policy='event',
                          settle=True, timing='adaptive', layout='rear'))
    elif group == 'exec_fixed':       # uniform reference duration for all vehicles
        for seed in range(4):
            for name, settle in pol:
                J.append(dict(group=group, name=f's{seed}_{name}', seed=seed, method='combined',
                              policy='barrier' if name == 'barrier' else 'event', settle=settle, timing='fixed'))
    elif group == 'exec_methods':     # closed-loop consequence of each planning method
        for seed in range(6):
            for method in ['source_blocks', 'mda', 'motion_speed']:
                J.append(dict(group=group, name=f's{seed}_{method}', seed=seed, method=method,
                              policy='event', settle=True, timing='adaptive'))
    elif group == 'robust':           # certainty-equivalent prediction under actuator lag and disturbance
        for seed in range(5):
            for lag, noise, margin in [(.3, .02, 0.), (.3, .02, .2), (.6, .05, 0.), (.6, .05, .2)]:
                J.append(dict(group=group, name=f's{seed}_lag{lag}_w{noise}_m{margin}', seed=seed, method='combined',
                              policy='event', settle=True, timing='adaptive', lag=lag, noise=noise, margin=margin,
                              recovery=True))
            J.append(dict(group=group, name=f's{seed}_lag0.6_w0.02_m0.2', seed=seed, method='combined',
                          policy='event', settle=True, timing='adaptive', lag=.6, noise=.02, margin=.2, recovery=True, tol=(.2, .06)))
    elif group == 'ablation':         # single-gap check instead of rank-difference reservation
        for seed in range(10):
            for name, settle in [('eager', False), ('settled', True)]:
                J.append(dict(group=group, name=f's{seed}_{name}_noreserve', seed=seed, method='combined',
                              policy='event', settle=settle, timing='adaptive', reserve=False))
    elif group == 'jsweep':           # comfort envelope sweep (time-effort front)
        for seed in range(4):
            for j in [.02, .08]:
                for name, settle in pol:
                    J.append(dict(group=group, name=f's{seed}_{name}_j{j}', seed=seed, method='combined',
                                  policy='barrier' if name == 'barrier' else 'event', settle=settle, timing='adaptive', j_ref=j))
    elif group == 'lagaware':         # lag-aware prediction with the true time constant
        for seed in range(5):
            for lag, noise in [(.3, .02), (.6, .02), (.9, .02), (.6, .05)]:
                J.append(dict(group=group, name=f's{seed}_lag{lag}_w{noise}_hat{lag}', seed=seed, method='combined',
                              policy='event', settle=True, timing='adaptive', lag=lag, noise=noise, margin=.2,
                              recovery=True, lag_model=lag, tol=(.2, .06)))
    elif group == 'lagmismatch':      # true lag 0.6 s, assumed lag differs
        for seed in range(5):
            for hat in [.3, .45, .9]:
                J.append(dict(group=group, name=f's{seed}_lag0.6_w0.02_hat{hat}', seed=seed, method='combined',
                              policy='event', settle=True, timing='adaptive', lag=.6, noise=.02, margin=.2,
                              recovery=True, lag_model=hat, tol=(.2, .06)))
    elif group == 'scale':
        for m in [4, 6, 10, 12]:
            for seed in range(2):
                J.append(dict(group=group, name=f'n{3*m}_s{seed}', seed=seed, method='combined', m=m,
                              policy='event', settle=True, timing='adaptive'))
    return J


def execute(job):
    out = OUT/job['group']/job['name']
    if (out/'summary.json').exists(): return 'skip'
    pl = plan_file(job['seed'], job['method'], job.get('m', 8), job.get('layout', 'center'))
    comfort = dict(COMFORT, j_ref=job.get('j_ref', COMFORT['j_ref']))
    s = run(pl, out, policy=job['policy'], settle=job['settle'], reserve=job.get('reserve', True), timing=job['timing'],
            comfort=comfort, lag=job.get('lag', 0.), noise=job.get('noise', 0.), margin=job.get('margin', 0.),
            noise_seed=1000+job['seed'], limit=job.get('limit', 210.), recovery=job.get('recovery', False),
            lag_model=job.get('lag_model', 0.), tol=tuple(job.get('tol', (.08, .025))))
    s2 = json.loads((out/'summary.json').read_text(encoding='utf-8')); s2['job'] = job
    (out/'summary.json').write_text(json.dumps(s2, ensure_ascii=False, indent=1), encoding='utf-8')
    return f"{job['group']}/{job['name']} {s['success']} t={s['time']:.1f} E={s['acceleration_squared_integral']:.2f} fb={s['qp_fallbacks']} wall={s['wall_seconds']:.0f}"


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--groups', nargs='+', required=True)
    p.add_argument('--worker', type=int, default=0); p.add_argument('--workers', type=int, default=1)
    p.add_argument('--max', type=int, default=10**9); p.add_argument('--reverse', action='store_true')
    p.add_argument('--filter', default='True', help="python expression on job dict j")
    a = p.parse_args()
    J = [j for g in a.groups for j in jobs(g) if eval(a.filter, {}, dict(j=j))]
    J = J[a.worker::a.workers]
    if a.reverse: J = J[::-1]
    done = 0
    for job in J:
        if done >= a.max: break
        try:
            msg = execute(job)
            if msg != 'skip': done += 1; print('DONE', msg, flush=True)
        except Exception as e:
            print('FAIL', job['group'], job['name'], repr(e), flush=True); traceback.print_exc()
