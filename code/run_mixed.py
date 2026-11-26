"""Reconfiguration behind human-driven vehicles (mixed traffic).

The connected vehicles are simulated in a frame that moves with the group
reference speed v0(t).  Human-driven vehicles (HDVs) ahead in the lanes follow
their own speed schedule, do not share plans and are only measured.  The group
adapts v0 through a common acceleration a0 computed from the measured HDV
states, so the relative motion of the connected vehicles is not changed by a0.
"""
from pathlib import Path
import json, argparse
from collections import Counter
from dataclasses import replace
from time import perf_counter
import numpy as np

from scenario import Config, make_scenario, jsonable
from dmpc import matrices, reference, initial_plan
from event_mpc import EventQP, select_neighbors, crossing_gate, interval_absolute_min
from timing import comfort_duration, reference_v
from run_exec import SMOOTH, COMFORT, order_ready
from plan_methods import plan as make_plan

ROOT = Path(__file__).resolve().parent.parent
HDV = dict(gap=35., offsets=(4., 0., 9.), brake_lane=1, onset=45., drop=3., decel=1., accel=.6,
           jerk=1., resume=75., gain=.8, sigma=0., tau=4., seed=0, buffer=2.)
FRAME = dict(kp=.09, kv=.6, ks=.25, a_max=1., j_max=1.)


def showcase_scenario(seed=0, spacing=20., offsets=(5., 0., -5.)):
    """Steady source platoons with equal spacing and zero relative speeds."""
    cfg = Config(); sc = make_scenario(seed, cfg)
    m = cfg.vehicles_per_source; n = 3*m; src = np.array(sc['source'])
    x0 = np.zeros((n, 4)); x0[:, 0] = np.tile(-spacing*np.arange(m), 3)+np.repeat(offsets, m); x0[:, 1] = src*cfg.lane_width
    center = x0[:, 0].mean()
    sc.update(initial=x0, stage_x=center+(n-1)*cfg.stage_gap/2-np.arange(n)*cfg.stage_gap,
              target_x=center+(n//2-1)*cfg.final_gap/2-np.arange(n//2)*cfg.final_gap)
    return sc, cfg


def make_showcase_plan(seed=0):
    sc, cfg = showcase_scenario(seed)
    return dict(scenario=jsonable(sc), assignment=jsonable(make_plan(sc, 'combined', cfg)))


def pair_margins(xa, ua, xb, ub, h, thresholds):
    """Smallest max_d(|relative position_d| - threshold_d) over all pairs and the hold interval."""
    p = xa[:, None, :2]-xb[None, :, :2]; v = xa[:, None, 2:]-xb[None, :, 2:]; a = ua[:, None, :]-ub[None, :, :]
    return float(np.max(interval_absolute_min(p, v, a, h)-np.asarray(thresholds), axis=2).min())


def self_margins(x, u, h, thresholds):
    n = len(x); ix, iy = np.triu_indices(n, 1)
    p = x[ix, :2]-x[iy, :2]; v = x[ix, 2:]-x[iy, 2:]; a = u[ix]-u[iy]
    return float(np.max(interval_absolute_min(p, v, a, h)-np.asarray(thresholds), axis=1).min())


def hdv_target(k, t, lane, hv, cruise):
    if lane != hv['brake_lane'] or t < hv['onset'] or t >= hv['resume']: return cruise
    return cruise-hv['drop']


def run_mixed(plan, out, hdv=None, frame=None, t_start=10., t_end=160., limit=260., comfort=COMFORT,
              overrides=SMOOTH, adapt=True, verbose=False):
    hv = dict(HDV, **(hdv or {})); fr = dict(FRAME, **(frame or {}))
    sc = json.loads(json.dumps(plan['scenario'])); assignment = plan['assignment']
    if overrides: sc['config'].update(overrides)
    cfg = Config(**sc['config'])
    ctrl = replace(cfg, ax_max=cfg.ax_max-fr['a_max'])           # bound split, common a0 takes the rest
    x = np.array(sc['initial'], float); source = np.array(sc['source']); dest = np.array(assignment['destination'])
    n = len(x); h = cfg.dt; H = cfg.horizon; A, B, E, F = matrices(cfg); vc = cfg.cruise
    rank = np.empty(n, dtype=int); target_rank = np.empty(n, dtype=int)
    for r, i in enumerate(assignment['order']): rank[i] = r
    stage_goal = np.column_stack([assignment['stage_x'], source*cfg.lane_width])
    final_goal = np.column_stack([assignment['final_x'], dest*cfg.lane_width])
    coupling = [[] for _ in range(n)]
    for chain in assignment['target_orders']:
        for k, i in enumerate(chain): target_rank[i] = k
        for a, b in zip(chain[:-1], chain[1:]): coupling[a].append(b); coupling[b].append(a)
    T_stage = comfort_duration(x[:, 0], x[:, 2], np.zeros(n), stage_goal[:, 0], comfort['a_ref'], comfort['j_ref'], h,
                               comfort['t_min'], comfort['t_max'])
    T_compact = np.full(n, np.nan)
    # human-driven vehicles, one ahead of the group in every lane
    lanes = np.arange(3); m = len(lanes)
    front = float(np.max(stage_goal[:, 0]))
    xh = np.zeros((m, 4)); xh[:, 0] = front+hv['gap']+np.asarray(hv['offsets'])[:m]; xh[:, 1] = lanes*cfg.lane_width
    xh0 = xh.copy(); ah = np.zeros(m); noise = np.zeros(m); rng = np.random.default_rng(hv['seed'])
    v0 = vc; s0 = 0.; a0 = 0.
    previous = np.zeros((n, 2)); uplan, shared = initial_plan(x, E, F, cfg)
    hold_ref = np.repeat(np.column_stack([x[:, :2], np.zeros((n, 2))])[:, None, :], H, axis=1)
    crossing_start = np.full(n, np.nan); crossing_done = np.full(n, np.nan)
    needs = source != dest; crossing_done[~needs] = -np.inf
    cross_initial = np.zeros_like(x); cross_acc = np.zeros((n, 2))
    compact_start = np.full(n, np.nan); compact_initial = np.zeros_like(x); compact_acc = np.zeros((n, 2))
    qp_cache = {}; logs = []; events = []; hist = [x.copy()]; hhist = [xh.copy()]; uhist = []; hahist = []
    fhist = [(s0, v0, a0)]; m_cc = []; m_ch = []; b_cc = []; b_ch = []; modes = []
    fallback = 0; failure = None; begun = perf_counter(); messages = 0; t_done = None
    tt = np.arange(1, H+1)*h
    dims_body = np.array([cfg.length+cfg.width*np.sin(cfg.heading_max), cfg.width+cfg.length*np.sin(cfg.heading_max)])
    rule = np.array([cfg.safety_dx, cfg.safety_dy])
    while len(uhist)*h < limit-1e-9:
        t = len(uhist)*h; tr = t-t_start
        # common acceleration of the group from the measured HDV states
        e = xh[:, 0]-xh0[:, 0]; dv = xh[:, 2]
        a_des = min(fr['ks']*(vc-v0), float(np.min(fr['kp']*e+fr['kv']*dv))) if adapt else 0.
        a0 = float(np.clip(np.clip(a_des, a0-fr['j_max']*h, a0+fr['j_max']*h), -fr['a_max'], fr['a_max']))
        # HDV accelerations from their own speed schedule (not known to the group)
        for k in range(m):
            vt = hdv_target(k, t, lanes[k], hv, vc); vg = v0+xh[k, 2]
            lim = (-hv['decel'], hv['accel'])
            want = float(np.clip(hv['gain']*(vt-vg), *lim))
            ah[k] = float(np.clip(want, ah[k]-hv['jerk']*h, ah[k]+hv['jerk']*h))
        if hv['sigma'] > 0:
            rho = np.exp(-h/hv['tau']); noise = rho*noise+hv['sigma']*np.sqrt(1-rho*rho)*rng.standard_normal(m)
        ahr = ah+noise-a0                                           # HDV acceleration relative to the moving frame
        shared_h = np.stack([np.column_stack([xh[k, 0]+xh[k, 2]*tt, np.full(H, xh[k, 1]), np.full(H, xh[k, 2]), np.zeros(H)])
                             for k in range(m)])                    # constant-speed prediction from the measurement
        if tr < 0:
            ref = hold_ref.copy()
        else:
            ref = reference_v(np.array(sc['initial'], float), stage_goal, T_stage, np.tile(tr+tt, (n, 1)))
        for i in range(n):
            if np.isfinite(crossing_start[i]):
                yg = final_goal[i:i+1].copy(); yg[0, 0] = cross_initial[i, 0]
                yr = reference(cross_initial[i:i+1], yg, cfg.crossing_time, t-crossing_start[i]+tt, cross_acc[i:i+1])[0]
                ref[i, :, 1] = yr[:, 1]; ref[i, :, 3] = yr[:, 3]
                if t-crossing_start[i] >= cfg.crossing_time and abs(x[i, 1]-final_goal[i, 1]) < .08 and abs(x[i, 3]) < .025:
                    if not np.isfinite(crossing_done[i]):
                        crossing_done[i] = t; events.append(dict(time=t, vehicle=i, type='lane_complete'))
            if np.isfinite(compact_start[i]):
                ref[i] = reference(compact_initial[i:i+1], final_goal[i:i+1], T_compact[i], t-compact_start[i]+tt, compact_acc[i:i+1])[0]
        local_xready = (abs(x[:, 0]-stage_goal[:, 0]) < .08) & (abs(x[:, 2]) < .025) & (tr >= T_stage-1e-9)
        for i in range(n):
            ready = order_ready(i, x, shared, dest, target_rank, ctrl, True)
            pending = np.flatnonzero(needs & ~np.isfinite(crossing_done))
            dependencies = []
            ilo, ihi = sorted([x[i, 0], final_goal[i, 0]])
            for j in pending:
                if j == i or not min(source[j], dest[j]) <= dest[i] <= max(source[j], dest[j]): continue
                jlo, jhi = sorted([x[j, 0], stage_goal[j, 0]])
                if max(ilo, jlo)-min(ihi, jhi) < ctrl.safety_dx+.1: dependencies.append(int(j))
            can_compact = ready and (np.isfinite(crossing_done[i]) or not needs[i]) and not dependencies and local_xready[i]
            if can_compact and not np.isfinite(compact_start[i]):
                compact_start[i] = t; compact_initial[i] = x[i]; compact_acc[i] = previous[i]
                T_compact[i] = comfort_duration(x[i:i+1, 0], x[i:i+1, 2], previous[i:i+1, 0], final_goal[i:i+1, 0],
                                                comfort['a_ref'], comfort['j_ref'], h, comfort['t_min'], comfort['t_max'])[0]
                ref[i] = reference(x[i:i+1], final_goal[i:i+1], T_compact[i], tt, previous[i:i+1])[0]
                events.append(dict(time=t, vehicle=i, type='compact_start', duration=float(T_compact[i])))
        eligible = tr >= 0
        ctrl_t = replace(ctrl, cruise=v0)
        for sweep in range(cfg.sweeps):
            order = assignment['order'] if (len(uhist)+sweep) % 2 == 0 else assignment['order'][::-1]
            X = np.vstack([x, xh]); SH = np.concatenate([shared, shared_h]); UP = np.concatenate([uplan, np.zeros((m, H, 2))])
            for i in order:
                X[:n] = x; SH[:n] = shared; UP[:n] = uplan
                launch = False
                if needs[i] and not np.isfinite(crossing_start[i]) and eligible:
                    permitted, blockers = crossing_gate(i, x, shared, source, dest, rank, ctrl_t)
                    if permitted:
                        launch = True
                        yg = final_goal[i:i+1].copy(); yg[0, 0] = x[i, 0]
                        yr = reference(x[i:i+1], yg, cfg.crossing_time, tt, previous[i:i+1])[0]
                        oldref = ref[i].copy(); ref[i, :, 1] = yr[:, 1]; ref[i, :, 3] = yr[:, 3]
                iscross = np.isfinite(crossing_start[i]) and not np.isfinite(crossing_done[i])
                fullroad = launch or iscross
                if fullroad:
                    half = (cfg.width+cfg.length*np.sin(cfg.heading_max))/2
                    ylimits = (-cfg.lane_width/2+half, 2.5*cfg.lane_width-half)
                else:
                    lane = dest[i] if np.isfinite(crossing_done[i]) else source[i]
                    ylimits = (lane*cfg.lane_width-.15, lane*cfg.lane_width+.15)
                neighbors = select_neighbors(i, X, UP, SH, ctrl_t)
                fixed_x = {j: ((1 if rank[i] < rank[j] else -1), abs(target_rank[i]-target_rank[j])*(ctrl.safety_dx+.02))
                           for j in range(n) if j != i and dest[j] == dest[i] and (np.isfinite(compact_start[i]) or np.isfinite(compact_start[j]))}
                for k in range(m):                                   # stay behind every HDV with a buffer
                    if np.min(SH[n+k, :, 0]-shared[i, :, 0]) < ctrl.safety_dx+hv['buffer']+2.2:
                        fixed_x[n+k] = (-1, ctrl.safety_dx+hv['buffer'])
                neighbors = sorted(set(neighbors) | set(fixed_x))
                key = (i, tuple(neighbors), tuple(sorted(fixed_x.items())))
                if key not in qp_cache:
                    qp_cache[key] = EventQP(i, neighbors, coupling[i], ctrl_t, E, F, fixed_x)
                qp = qp_cache[key]; qp.cfg = ctrl_t
                uu, pp, log = qp.solve(X, previous, ref, SH, UP, ylimits, fullroad)
                logs.append((log['status'], log['accepted'], log['seconds'], launch))
                if uu is not None:
                    uplan[i] = uu; shared[i] = pp; messages += len(set([j for j in neighbors if j < n]+coupling[i]))
                    if launch:
                        crossing_start[i] = t; cross_initial[i] = x[i]; cross_acc[i] = previous[i]
                        events.append(dict(time=t, vehicle=i, type='lane_start'))
                else:
                    fallback += 1
                    if launch: ref[i] = oldref
        u = uplan[:, 0].copy()
        uh = np.column_stack([ahr, np.zeros(m)])
        cc = self_margins(x, u, h, rule); ch = pair_margins(x, u, xh, uh, h, rule)
        bc = self_margins(x, u, h, dims_body); bh = pair_margins(x, u, xh, uh, h, dims_body)
        m_cc.append(cc); m_ch.append(ch); b_cc.append(bc); b_ch.append(bh)
        if min(bc, bh) < 0 and failure is None:
            failure = f'body separation lost at {t:.1f} s'
        uhist.append(u.copy()); hahist.append(np.r_[ah+noise, a0])
        x = (A@x.T).T+(B@u.T).T; xh = (A@xh.T).T+(B@uh.T).T; hist.append(x.copy()); hhist.append(xh.copy())
        s0 += h*(v0-vc)+.5*h*h*a0; v0 += h*a0; fhist.append((s0, v0, a0))
        mode = np.zeros(n, dtype=int); mode[np.isfinite(crossing_start)] = 1; mode[np.isfinite(crossing_done) & needs] = 2; mode[np.isfinite(compact_start)] = 3
        modes.append(mode); previous = u
        uplan = np.concatenate([uplan[:, 1:], np.zeros((n, 1, 2))], axis=1)
        shared = np.stack([(E@xx+F@uu_.reshape(-1)).reshape(H, 4) for xx, uu_ in zip(x, uplan)])
        elapsed_ok = np.isfinite(compact_start) & ((t+h-np.nan_to_num(compact_start)) >= np.nan_to_num(T_compact)-1e-9)
        done = (np.max(abs(x[:, :2]-final_goal), axis=1) < .08) & (np.max(abs(x[:, 2:]), axis=1) < .025) & elapsed_ok
        if np.all(done) and t_done is None:
            t_done = t+h; events.append(dict(time=t_done, vehicle=-1, type='complete'))
        if failure is not None: break
        if t_done is not None and t+h >= max(t_end, t_done)-1e-9: break
        if verbose and int(t/h) % 50 == 0:
            print('t', round(t, 1), 'v0', round(v0, 2), 'a0', round(a0, 2), 'cross', int(np.isfinite(crossing_start).sum()),
                  'compact', int(np.isfinite(compact_start).sum()), 'fb', fallback, 'mcc', round(min(m_cc), 3), 'mch', round(min(m_ch), 2), flush=True)
    if failure is None and t_done is None: failure = 'time limit before all vehicles completed'
    traj = np.array(hist); u = np.array(uhist); ti = np.array([r[2] for r in logs]); fh = np.array(fhist)
    du = np.diff(np.concatenate([np.zeros((1, n, 2)), u]), axis=0)/h
    a_total = u[:, :, 0]+fh[:-1, 2][:, None]
    summary = dict(seed=sc['seed'], hdv=hv, frame=fr, adapt=adapt, t_start=t_start, success=failure is None, failure=failure,
                   completion=None if t_done is None else t_done-t_start, t_done=t_done, time=len(u)*h,
                   T_stage=T_stage.tolist(), T_compact=T_compact.tolist(), crossing_start=crossing_start.tolist(),
                   crossing_done=np.where(np.isfinite(crossing_done), crossing_done, np.nan).tolist(),
                   compact_start=compact_start.tolist(), min_center_margin_cav=float(min(m_cc)), min_center_margin_hdv=float(min(m_ch)),
                   min_body_margin_cav=float(min(b_cc)), min_body_margin_hdv=float(min(b_ch)),
                   min_gap_hdv=float(min(m_ch))+cfg.safety_dx, min_v0=float(fh[:, 1].min()), max_abs_a0=float(abs(fh[:, 2]).max()),
                   max_ax_total=float(abs(a_total).max()), max_ax_relative=float(abs(u[:, :, 0]).max()), max_ay=float(abs(u[:, :, 1]).max()),
                   max_jx_relative=float(abs(du[:, :, 0]).max()), acceleration_squared_integral=float(np.sum(u*u)*h),
                   total_acceleration_squared_integral=float((np.sum(a_total**2)+np.sum(u[:, :, 1]**2))*h),
                   qp_attempts=len(logs), qp_fallbacks=fallback, qp_status_counts=dict(Counter(r[0] for r in logs)),
                   qp_seconds_median=float(np.median(ti)), qp_seconds_p95=float(np.percentile(ti, 95)),
                   trajectory_messages=messages, wall_seconds=perf_counter()-begun, events=events,
                   assignment=assignment, config=sc['config'], comfort=comfort)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for name, data in [('summary', summary), ('scenario', sc)]:
        (out/(name+'.json')).write_text(json.dumps(jsonable(data), ensure_ascii=False, indent=1), encoding='utf-8')
    np.savez_compressed(out/'trajectory.npz', trajectory=traj, commands=u, hdv=np.array(hhist), hdv_acc=np.array(hahist),
                        frame=fh, final_goal=final_goal, stage_goal=stage_goal, margin_cav=np.array(m_cc), margin_hdv=np.array(m_ch),
                        body_cav=np.array(b_cc), body_hdv=np.array(b_ch), modes=np.array(modes))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', default=str(ROOT/'results'/'mixed'/'showcase'))
    p.add_argument('--seed', type=int, default=0); p.add_argument('--hdv', default='{}'); p.add_argument('--no-adapt', action='store_true')
    p.add_argument('--t-end', type=float, default=160.)
    a = p.parse_args()
    pf = ROOT/'results'/'mixed'/'plans'/f'showcase_s{a.seed}.json'; pf.parent.mkdir(parents=True, exist_ok=True)
    if not pf.exists(): pf.write_text(json.dumps(make_showcase_plan(a.seed)), encoding='utf-8')
    s = run_mixed(json.loads(pf.read_text(encoding='utf-8')), a.out, hdv=dict(json.loads(a.hdv), seed=a.seed),
                  adapt=not a.no_adapt, t_end=a.t_end, verbose=True)
    print({k: s[k] for k in ['success', 'failure', 'completion', 'min_center_margin_cav', 'min_center_margin_hdv', 'min_gap_hdv',
                             'min_v0', 'max_abs_a0', 'max_ax_total', 'qp_fallbacks', 'wall_seconds']})
