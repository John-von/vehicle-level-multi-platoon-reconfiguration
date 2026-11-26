"""Group EX12: response of the upstream traffic to a reconfiguration.

Behind the reconfiguring vehicles, every lane carries a string of human-driven
vehicles that follow the calibrated intelligent driver model (driver_model.py).
A reconfiguring vehicle reserves its source lane, its destination lane and the
lanes between them until its last lane change is complete, and its destination
lane afterwards.  The first vehicle of a string follows the rearmost vehicle that
reserves its lane, so the upstream traffic never enters the region that the
reconfiguration still needs.  The upstream vehicles do not change lanes and do
not influence the reconfiguring vehicles.  The recorded trajectories are extended
by holding the final formation for EXTEND seconds so that the disturbance can
propagate.
"""
from pathlib import Path
import json, argparse
import numpy as np

from scenario import Config
from driver_model import sample_driver, idm_acc, equilibrium_gap

ROOT = Path(__file__).resolve().parent.parent; RES = ROOT/'results'; OUT = RES/'traffic'
N_UP, DT, EXTEND = 20, .1, 60.
HARD = 3.0       # hard braking threshold (m/s^2)


def drivers_for(seed, lane):
    rng = np.random.default_rng(7000+100*seed+lane)
    return [sample_driver(rng) for _ in range(N_UP)]


def load_group(run, cfg):
    """Trajectories extended by EXTEND seconds and the lane reservation of every vehicle."""
    z = np.load(run/'trajectory.npz'); X = np.array(z['trajectory'], float)
    hold = int(round(EXTEND/cfg.dt)); tail = np.repeat(X[-1:], hold, axis=0); tail[:, :, 2:] = 0.
    X = np.concatenate([X, tail]); t = np.arange(len(X))*cfg.dt
    sm = json.loads((run/'summary.json').read_text(encoding='utf-8'))
    sc = json.loads((run/'scenario.json').read_text(encoding='utf-8'))
    src = np.array(sc['source'], int); dst = np.array(sm['assignment']['destination'], int)
    done = np.array([np.inf if c is None else float(c) for c in sm.get('crossing_done', [None]*len(src))])
    span = sm.get('lane_range', [sorted((a, b)) for a, b in zip(src, dst)])
    done = np.where((src == dst) & ~np.isfinite(done), 0., done)
    res = np.zeros((len(t), X.shape[1], 3), bool)
    for i in range(X.shape[1]):
        lo, hi = span[i]
        res[:, i, lo:hi+1] = (t < done[i])[:, None]
        res[:, i, dst[i]] = True
    return t, X, res


def simulate(t, X, R, seed, cfg, record=False):
    """Upstream strings in lanes 0..2; returns metrics and optionally trajectories in the moving frame."""
    v_c, L = cfg.cruise, cfg.length
    tf = np.arange(0., t[-1]+1e-9, DT)
    # group states on the fine grid (positions with constant-acceleration interpolation, speeds linear)
    gx = np.empty((len(tf), X.shape[1])); gv = np.empty_like(gx)
    for i in range(X.shape[1]):
        gx[:, i] = np.interp(tf, t, X[:, i, 0]); gv[:, i] = np.interp(tf, t, X[:, i, 2])
    gr = R[np.minimum(np.searchsorted(t, tf, side='right')-1, len(t)-1)]
    res = dict(lanes=[]); keep = {}
    for lane in range(3):
        drv = drivers_for(seed, lane)
        occ0 = gr[0, :, lane]
        tail0 = gx[0, occ0].min() if occ0.any() else 0.
        x = np.empty(N_UP); v = np.zeros(N_UP)
        pos = tail0
        for k in range(N_UP):
            pos -= L+equilibrium_gap(v_c, drv[k]); x[k] = pos
        xs = np.empty((len(tf), N_UP)); vs = np.empty_like(xs); amin = np.zeros(N_UP)
        min_tg = np.inf; collided = False
        for n_, tt in enumerate(tf):
            xs[n_] = x; vs[n_] = v
            # leader of the first upstream vehicle: rearmost reconfiguring vehicle ahead that reserves this lane
            ahead = gr[n_, :, lane] & (gx[n_] > x[0])
            if ahead.any():
                j = np.flatnonzero(ahead)[np.argmin(gx[n_, ahead])]
                lead = (gx[n_, j]-x[0]-L, v_c+gv[n_, j])
                if lead[0] < 0: collided = True
                if v_c+v[0] > 1.: min_tg = min(min_tg, lead[0]/(v_c+v[0]))
            else:
                lead = None
            a = np.empty(N_UP)
            a[0] = idm_acc(v_c+v[0], lead[0], lead[1], drv[0]) if lead else idm_acc(v_c+v[0], None, None, drv[0])
            for k in range(1, N_UP):
                gap = x[k-1]-x[k]-L
                a[k] = idm_acc(v_c+v[k], gap, v_c+v[k-1], drv[k])
                if gap < 0: collided = True
                if v_c+v[k] > 1.: min_tg = min(min_tg, gap/(v_c+v[k]))
            # ballistic update in the frame moving with v_c, speeds kept non-negative
            vabs = v_c+v; vnew = np.maximum(vabs+a*DT, 0.)
            x = x+.5*(vabs+vnew)*DT-v_c*DT; v = vnew-v_c; amin = np.minimum(amin, a)
        speed = v_c+vs
        delay = np.sum(np.clip(v_c-speed, 0., None)/v_c, axis=0)*DT
        drop = v_c-speed.min(axis=0)
        res['lanes'].append(dict(lane=lane, delay=float(delay.sum()), delay_per_vehicle=delay.tolist(), max_drop=float(drop.max()),
                                 drop_first=float(drop[0]), drop_last=float(drop[-1]), hard_braking=int(np.sum(amin < -HARD)),
                                 min_acc=float(amin.min()), min_time_gap=float(min_tg), collision=bool(collided)))
        if record: keep[lane] = dict(x=xs[::4], v=speed[::4])
    res['delay'] = float(sum(l['delay'] for l in res['lanes'])); res['max_drop'] = float(max(l['max_drop'] for l in res['lanes']))
    res['hard_braking'] = int(sum(l['hard_braking'] for l in res['lanes'])); res['collision'] = bool(any(l['collision'] for l in res['lanes']))
    res['min_time_gap'] = float(min(l['min_time_gap'] for l in res['lanes'])); res['min_acc'] = float(min(l['min_acc'] for l in res['lanes']))
    return res, keep


def steady(run, cfg):
    t, X, R = load_group(run, cfg); Xs = np.repeat(X[:1], len(X), axis=0); Xs[:, :, 2:] = 0.
    Rs = np.zeros_like(R); Rs[:, np.arange(X.shape[1]), np.round(X[0, :, 1]/cfg.lane_width).astype(int)] = True
    return t, Xs, Rs


def runs():
    out = []
    for s in range(20):
        for p in ('barrier', 'eager', 'settled'):
            out.append((f's{s}_{p}', RES/'exec_adaptive'/f's{s}_{p}', s))
        out.append((f's{s}_rear', RES/'exec_rear'/f's{s}_settled', s))
        out.append((f's{s}_cai', RES/'cai'/'runs'/f'n24_s{s}', s))
    for s in range(6):
        for p in ('source_blocks', 'mda', 'motion_speed'):
            out.append((f's{s}_{p}', RES/'exec_methods'/f's{s}_{p}', s))
    for n in (12, 18):
        for s in (0, 1):
            out.append((f'n{n}_s{s}_dp', RES/'scale'/f'n{n}_s{s}', s))
            out.append((f'n{n}_s{s}_cai', RES/'cai'/'runs'/f'n{n}_s{s}', s))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--record', nargs='*', default=['s0_settled', 's0_rear', 's0_barrier'])
    a = ap.parse_args(); cfg = Config(); OUT.mkdir(parents=True, exist_ok=True)
    for name, run, seed in runs():
        if not (run/'trajectory.npz').exists() or (OUT/f'{name}.json').exists(): continue
        t, X, R = load_group(run, cfg)
        res, keep = simulate(t, X, R, seed, cfg, record=name in a.record)
        res.update(name=name, seed=seed)
        (OUT/f'{name}.json').write_text(json.dumps(res), encoding='utf-8')
        if keep: np.savez_compressed(OUT/f'{name}_upstream.npz', **{f'x{l}': d['x'] for l, d in keep.items()}, **{f'v{l}': d['v'] for l, d in keep.items()})
        print(name, 'delay %.1f drop %.2f hard %d mintg %.2f coll %s' % (res['delay'], res['max_drop'], res['hard_braking'], res['min_time_gap'], res['collision']), flush=True)
    for s in range(20):
        run = RES/'exec_adaptive'/f's{s}_settled'
        t, X, R = steady(run, cfg); res, _ = simulate(t, X, R, s, cfg); res.update(name=f's{s}_steady', seed=s)
        (OUT/f's{s}_steady.json').write_text(json.dumps(res), encoding='utf-8')
        print(f's{s}_steady', 'delay %.1f drop %.2f hard %d coll %s' % (res['delay'], res['max_drop'], res['hard_braking'], res['collision']), flush=True)
