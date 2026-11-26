"""Reconfiguration with human-driven vehicles (HDVs) inside the source platoons.

HDVs keep their lane and follow the intelligent driver model with parameters
drawn from a naturalistic-driving calibration (driver_model.py), which the
connected vehicles do not know.  The connected vehicles predict each HDV from
its measured state, as converging to the shared motion of its predecessor when
that predecessor is a connected vehicle and at constant speed otherwise, keep a
buffer to it, do not start a lane change inside its zone unless a vehicle that
stays in that lane lies between them, and shift goals that are defined relative to
an HDV with the measured displacement of that HDV.  Behind an HDV the members of
a target platoon start their compaction in their order in that platoon.
"""
from pathlib import Path
import json
from collections import Counter
from time import perf_counter
import numpy as np

from scenario import Config, jsonable
from dmpc import matrices, reference, initial_plan
from event_mpc import EventQP, select_neighbors, crossing_gate
from timing import comfort_duration, reference_v
from run_exec import SMOOTH, COMFORT, order_ready
from run_mixed import pair_margins, self_margins
from plan_mixed import ZONE
from driver_model import idm_acc

ROOT = Path(__file__).resolve().parent.parent
TOL = (.2, .06)     # settling and completion tolerances (position, relative speed)
HUMAN = dict(sigma=.1, tau=4.)          # random acceleration of the human drivers (simulation only)
BUFFER, CONVERGE = 2., 2.                # controller: buffer to an HDV and time constant of its predicted speed


def predict_hdv(x, xh, shared, tt, h, dy):
    """Predicted HDV states over the horizon.

    An HDV whose predecessor is a connected vehicle is predicted to converge to the
    shared motion of that predecessor with the time constant CONVERGE, and an HDV
    behind another HDV or on a free road is predicted at constant speed."""
    m, H = len(xh), len(tt); out = np.zeros((m, H, 4))
    for k in range(m):
        lead, best = None, np.inf
        for j in range(len(x)):
            if abs(x[j, 1]-xh[k, 1]) < dy and xh[k, 0] < x[j, 0] < best: lead, best = j, x[j, 0]
        for q in range(m):
            if q != k and abs(xh[q, 1]-xh[k, 1]) < dy and xh[k, 0] < xh[q, 0] < best: lead, best = None, xh[q, 0]
        if lead is None:
            v = np.full(H, xh[k, 2])
        else:
            v = shared[lead, :, 2]+(xh[k, 2]-x[lead, 2])*np.exp(-tt/CONVERGE)
        vv = np.concatenate([[xh[k, 2]], v])
        out[k, :, 0] = xh[k, 0]+np.cumsum(.5*(vv[:-1]+vv[1:])*h); out[k, :, 1] = xh[k, 1]; out[k, :, 2] = v
    return out


def run_inner(plan, out, limit=210., comfort=COMFORT, overrides=SMOOTH, aware=True, seed=0, verbose=False):
    p = dict(HUMAN)
    sc = json.loads(json.dumps(plan['scenario'])); assignment = plan['assignment']
    if overrides: sc['config'].update(overrides)
    cfg = Config(**sc['config'])
    X0 = np.array(sc['initial'], float); N = len(X0); H = sorted(sc['hdv']); drv = {int(k): v for k, v in sc.get('hdv_driver', {}).items()}
    cav = [i for i in range(N) if i not in H]; loc = {i: k for k, i in enumerate(cav)}; n = len(cav); m = len(H)
    h = cfg.dt; Hh = cfg.horizon; A, B, E, F = matrices(cfg); w = cfg.lane_width
    src_all = np.array(sc['source']); dest_all = np.array(assignment['destination'])
    x = X0[cav].copy(); xh = X0[H].copy(); source = src_all[cav]; dest = dest_all[cav].copy()
    full = assignment.get('target_orders_full', assignment['target_orders'])
    rank = np.empty(n, dtype=int)
    for r, i in enumerate([i for i in assignment['order'] if i in loc]): rank[loc[i]] = r
    target_rank = np.zeros(n, dtype=int); coupling = [[] for _ in range(n)]; anchor_f = np.full(n, -1); pred_c = np.full(n, -1)
    for chain in full:
        last_h = -1
        for k_, i in enumerate(chain):
            if i in loc:
                target_rank[loc[i]] = k_
                if aware: anchor_f[loc[i]] = last_h
                if k_ > 0 and chain[k_-1] in loc: pred_c[loc[i]] = loc[chain[k_-1]]
            elif i in H: last_h = H.index(i)
        for a_, b_ in zip(chain[:-1], chain[1:]):
            if a_ in loc and b_ in loc: coupling[loc[a_]].append(loc[b_]); coupling[loc[b_]].append(loc[a_])
    stage_goal0 = np.column_stack([np.array(assignment['stage_x'])[cav], source*w])
    final_goal0 = np.column_stack([np.array(assignment['final_x'])[cav], dest*w])
    T_stage = comfort_duration(x[:, 0], x[:, 2], np.zeros(n), stage_goal0[:, 0], comfort['a_ref'], comfort['j_ref'], h,
                               comfort['t_min'], comfort['t_max'])
    T_compact = np.full(n, np.nan)
    ctrl = cfg
    previous = np.zeros((n, 2)); uplan, shared = initial_plan(x, E, F, cfg)
    crossing_start = np.full(n, np.nan); crossing_done = np.full(n, np.nan)
    needs = source != dest; crossing_done[~needs] = -np.inf
    cross_initial = np.zeros_like(x); cross_acc = np.zeros((n, 2))
    compact_start = np.full(n, np.nan); compact_initial = np.zeros_like(x); compact_acc = np.zeros((n, 2)); compact_shift = np.zeros(n)
    rng = np.random.default_rng(500+seed); noise = np.zeros(m); ah = np.zeros(m)
    qp_cache = {}; logs = []; events = []; hist = [x.copy()]; hhist = [xh.copy()]; uhist = []; hahist = []
    m_cc = []; m_ch = []; b_cc = []; b_ch = []; fallback = 0; failure = None; begun = perf_counter(); t_done = None
    tt = np.arange(1, Hh+1)*h; rule = np.array([cfg.safety_dx, cfg.safety_dy])
    dims_body = np.array([cfg.length+cfg.width*np.sin(cfg.heading_max), cfg.width+cfg.length*np.sin(cfg.heading_max)])
    hmin_gap = np.full(m, np.inf); hmin_acc = np.zeros(m)
    lane_h = np.array([int(round(X0[k, 1]/w)) for k in H], dtype=int)
    while len(uhist)*h < limit-1e-9:
        t = len(uhist)*h
        disp = xh[:, 0]-X0[H, 0]
        shift_f = np.array([disp[a] if a >= 0 else 0. for a in anchor_f])
        # staging goals behind an HDV are pushed back only as far as the HDV zone requires
        lane_now = np.where(needs & np.isfinite(crossing_done), dest, source); stays = ~needs | np.isfinite(crossing_done)
        anchor_s = np.full(n, -1); goal_s = stage_goal0[:, 0].copy()
        if aware and m:
            waiting = ~np.isfinite(compact_start)
            for k in range(m):
                behind = [i for i in range(n) if waiting[i] and lane_now[i] == lane_h[k] and x[i, 0] < xh[k, 0]
                          and not any(lane_h[q] == lane_h[k] and x[i, 0] < xh[q, 0] < xh[k, 0] for q in range(m))]
                bound = xh[k, 0]-ZONE['rear']
                for i in sorted(behind, key=lambda i: -stage_goal0[i, 0]):
                    if goal_s[i] > bound: goal_s[i] = bound; anchor_s[i] = k
                    bound = goal_s[i]-cfg.stage_gap
        shift_s = goal_s-stage_goal0[:, 0]
        stage_goal = stage_goal0.copy(); stage_goal[:, 0] += shift_s
        final_goal = final_goal0.copy(); final_goal[:, 0] += shift_f
        # human drivers (calibrated IDM) on the measured states, acceleration held over the sampling period
        allx = np.vstack([x, xh])
        for k in range(m):
            same = np.abs(allx[:, 1]-xh[k, 1]) < cfg.safety_dy
            same[n+k] = False
            ahead = same & (allx[:, 0] > xh[k, 0])
            lead = None
            if ahead.any():
                j = np.flatnonzero(ahead)[np.argmin(allx[ahead, 0])]
                lead = (allx[j, 0]-xh[k, 0]-cfg.length, cfg.cruise+allx[j, 2])
                hmin_gap[k] = min(hmin_gap[k], (allx[j, 0]-xh[k, 0]-cfg.length)/max(cfg.cruise+xh[k, 2], 1.))
            gap_l, v_l = lead if lead is not None else (None, None)
            ah[k] = idm_acc(cfg.cruise+xh[k, 2], gap_l, v_l, drv[H[k]])
        rho = np.exp(-h/p['tau']); noise = rho*noise+p['sigma']*np.sqrt(1-rho*rho)*rng.standard_normal(m)
        ahk = ah+noise; hmin_acc = np.minimum(hmin_acc, ahk)
        shared_h = predict_hdv(x, xh, shared, tt, h, cfg.safety_dy) if m else np.zeros((0, Hh, 4))
        ref = reference_v(X0[cav], stage_goal0, T_stage, np.tile(t+tt, (n, 1)))
        ref[:, :, 0] += shift_s[:, None]
        for i in range(n):
            if np.isfinite(crossing_start[i]):
                yg = final_goal[i:i+1].copy(); yg[0, 0] = cross_initial[i, 0]
                yr = reference(cross_initial[i:i+1], yg, cfg.crossing_time, t-crossing_start[i]+tt, cross_acc[i:i+1])[0]
                ref[i, :, 1] = yr[:, 1]; ref[i, :, 3] = yr[:, 3]
                if t-crossing_start[i] >= cfg.crossing_time and abs(x[i, 1]-final_goal[i, 1]) < .08 and abs(x[i, 3]) < .025:
                    if not np.isfinite(crossing_done[i]):
                        crossing_done[i] = t; events.append(dict(time=t, vehicle=int(cav[i]), type='lane_complete'))
            if np.isfinite(compact_start[i]):
                ref[i] = reference(compact_initial[i:i+1], final_goal0[i:i+1]+[[compact_shift[i], 0]], T_compact[i],
                                   t-compact_start[i]+tt, compact_acc[i:i+1])[0]
                ref[i, :, 0] += shift_f[i]-compact_shift[i]
                if anchor_f[i] >= 0: ref[i, :, 0] += xh[anchor_f[i], 2]*tt; ref[i, :, 2] += xh[anchor_f[i], 2]
        moving = anchor_s >= 0
        v_anchor = np.array([xh[a, 2] if mv else 0. for a, mv in zip(anchor_s, moving)])
        local_xready = (abs(x[:, 0]-stage_goal[:, 0]) < TOL[0]) & (abs(x[:, 2]-v_anchor) < TOL[1]) & (t >= T_stage-1e-9)
        for i in range(n):
            ready = order_ready(i, x, shared, dest, target_rank, ctrl, True)
            pending = np.flatnonzero(needs & ~np.isfinite(crossing_done)); dependencies = []
            ilo, ihi = sorted([x[i, 0], final_goal[i, 0]])
            for j in pending:
                if j == i or not min(source[j], dest[j]) <= dest[i] <= max(source[j], dest[j]): continue
                jlo, jhi = sorted([x[j, 0], stage_goal[j, 0]])
                if max(ilo, jlo)-min(ihi, jhi) < ctrl.safety_dx+.1: dependencies.append(int(j))
            # behind an HDV the members compact in their order in the target platoon
            in_turn = anchor_f[i] < 0 or pred_c[i] < 0 or np.isfinite(compact_start[pred_c[i]])
            can_compact = ready and (np.isfinite(crossing_done[i]) or not needs[i]) and not dependencies and local_xready[i] and in_turn
            if can_compact and not np.isfinite(compact_start[i]):
                compact_start[i] = t; compact_initial[i] = x[i]; compact_acc[i] = previous[i]; compact_shift[i] = shift_f[i]
                T_compact[i] = comfort_duration(x[i:i+1, 0], x[i:i+1, 2], previous[i:i+1, 0], final_goal[i:i+1, 0],
                                                comfort['a_ref'], comfort['j_ref'], h, comfort['t_min'], comfort['t_max'])[0]
                ref[i] = reference(x[i:i+1], final_goal[i:i+1], T_compact[i], tt, previous[i:i+1])[0]
                events.append(dict(time=t, vehicle=int(cav[i]), type='compact_start', duration=float(T_compact[i])))
        for sweep in range(cfg.sweeps):
            order = [loc[i] for i in assignment['order'] if i in loc]
            if (len(uhist)+sweep) % 2: order = order[::-1]
            for i in order:
                X = np.vstack([x, xh]); SH = np.concatenate([shared, shared_h]); UP = np.concatenate([uplan, np.zeros((m, Hh, 2))])
                launch = False
                if needs[i] and not np.isfinite(crossing_start[i]):
                    permitted, blockers = crossing_gate(i, x, shared, source, dest, rank, ctrl)
                    if permitted and aware:
                        lo, hi = sorted((source[i], dest[i]))
                        for k in range(m):
                            lane_k = int(round(xh[k, 1]/w))
                            if lane_k == source[i] or not lo <= lane_k <= hi: continue
                            # a vehicle that stays in the lane between the two shields the HDV
                            a_, b_ = sorted((x[i, 0], xh[k, 0]))
                            shield = [x[j, 0] for j in range(n) if j != i and stays[j] and lane_now[j] == lane_k]
                            shield += [xh[q, 0] for q in range(m) if q != k and lane_h[q] == lane_k]
                            if any(a_ < z < b_ for z in shield): continue
                            for dt_ in (0., cfg.crossing_time):
                                rel = (x[i, 0]+x[i, 2]*dt_)-(xh[k, 0]+xh[k, 2]*dt_)
                                if -ZONE['rear'] < rel < ZONE['front']: permitted = False
                    if permitted:
                        launch = True
                        yg = final_goal[i:i+1].copy(); yg[0, 0] = x[i, 0]
                        yr = reference(x[i:i+1], yg, cfg.crossing_time, tt, previous[i:i+1])[0]
                        oldref = ref[i].copy(); ref[i, :, 1] = yr[:, 1]; ref[i, :, 3] = yr[:, 3]
                iscross = np.isfinite(crossing_start[i]) and not np.isfinite(crossing_done[i])
                fullroad = launch or iscross
                if fullroad:
                    half = (cfg.width+cfg.length*np.sin(cfg.heading_max))/2
                    ylimits = (-w/2+half, 2.5*w-half)
                    lanes_i = set(range(min(source[i], dest[i]), max(source[i], dest[i])+1))
                else:
                    lane = dest[i] if np.isfinite(crossing_done[i]) else source[i]
                    ylimits = (lane*w-.15, lane*w+.15); lanes_i = {int(lane)}
                neighbors = select_neighbors(i, X, UP, SH, ctrl)
                fixed_x = {j: ((1 if rank[i] < rank[j] else -1), abs(target_rank[i]-target_rank[j])*(ctrl.safety_dx+.02))
                           for j in range(n) if j != i and dest[j] == dest[i] and (np.isfinite(compact_start[i]) or np.isfinite(compact_start[j]))}
                for k in range(m):
                    lane_k = int(round(xh[k, 1]/w))
                    if lane_k in lanes_i and np.min(np.abs(SH[n+k, :, 0]-shared[i, :, 0])) < ctrl.safety_dx+BUFFER+2.2:
                        fixed_x[n+k] = ((1 if x[i, 0] > xh[k, 0] else -1), ctrl.safety_dx+BUFFER)
                neighbors = sorted(set(neighbors) | set(fixed_x))
                key = (i, tuple(neighbors), tuple(sorted(fixed_x.items())))
                if key not in qp_cache: qp_cache[key] = EventQP(i, neighbors, coupling[i], ctrl, E, F, fixed_x)
                uu, pp, log = qp_cache[key].solve(X, previous, ref, SH, UP, ylimits, fullroad)
                logs.append((log['status'], log['accepted'], log['seconds'], launch))
                if uu is not None:
                    uplan[i] = uu; shared[i] = pp
                    if launch:
                        crossing_start[i] = t; cross_initial[i] = x[i]; cross_acc[i] = previous[i]
                        events.append(dict(time=t, vehicle=int(cav[i]), type='lane_start'))
                else:
                    fallback += 1
                    if launch: ref[i] = oldref
        u = uplan[:, 0].copy(); uh = np.column_stack([ahk, np.zeros(m)]) if m else np.zeros((0, 2))
        cc = self_margins(x, u, h, rule); bc = self_margins(x, u, h, dims_body)
        ch = pair_margins(x, u, xh, uh, h, rule) if m else np.inf; bh = pair_margins(x, u, xh, uh, h, dims_body) if m else np.inf
        hh_ = self_margins(xh, uh, h, dims_body) if m > 1 else np.inf
        m_cc.append(cc); m_ch.append(ch); b_cc.append(bc); b_ch.append(bh)
        if min(bc, bh, hh_) < 0 and failure is None: failure = f'body separation lost at {t:.1f} s'
        uhist.append(u.copy()); hahist.append(ahk.copy())
        x = (A@x.T).T+(B@u.T).T
        if m: xh = (A@xh.T).T+(B@uh.T).T
        hist.append(x.copy()); hhist.append(xh.copy()); previous = u
        uplan = np.concatenate([uplan[:, 1:], np.zeros((n, 1, 2))], axis=1)
        shared = np.stack([(E@xx+F@uu_.reshape(-1)).reshape(Hh, 4) for xx, uu_ in zip(x, uplan)])
        disp = xh[:, 0]-X0[H, 0]; shift_f = np.array([disp[a] if a >= 0 else 0. for a in anchor_f])
        goal_now = final_goal0[:, 0]+shift_f
        elapsed_ok = np.isfinite(compact_start) & ((t+h-np.nan_to_num(compact_start)) >= np.nan_to_num(T_compact)-1e-9)
        vrel = x[:, 2]-np.array([xh[a, 2] if a >= 0 else 0. for a in anchor_f])
        done = (np.abs(x[:, 0]-goal_now) < TOL[0]) & (np.abs(x[:, 1]-final_goal0[:, 1]) < TOL[0]) & (np.abs(vrel) < TOL[1]) & (np.abs(x[:, 3]) < TOL[1]) & elapsed_ok
        if np.all(done) and t_done is None:
            t_done = t+h; events.append(dict(time=t_done, vehicle=-1, type='complete')); break
        if failure is not None: break
        if verbose and int(round(t/h)) % 50 == 0:
            print('t', round(t, 1), 'cross', int(np.isfinite(crossing_start).sum()), 'compact', int(np.isfinite(compact_start).sum()),
                  'fb', fallback, 'mcc', round(min(m_cc), 3), 'mch', round(min(m_ch), 2), 'hacc', np.round(hmin_acc, 2), flush=True)
    if failure is None and t_done is None: failure = 'time limit before all vehicles completed'
    traj = np.array(hist); U = np.array(uhist); ti = np.array([r[2] for r in logs])
    summary = dict(seed=sc['seed'], hdv=H, aware=aware, success=failure is None, failure=failure, completion=t_done, time=len(U)*h,
                   min_center_margin_cav=float(min(m_cc)), min_center_margin_hdv=float(min(m_ch)) if m else None,
                   min_body_margin_cav=float(min(b_cc)), min_body_margin_hdv=float(min(b_ch)) if m else None,
                   hdv_min_acc=hmin_acc.tolist(), hdv_min_time_gap=hmin_gap.tolist(),
                   acceleration_squared_integral=float(np.sum(U*U)*h), qp_attempts=len(logs), qp_fallbacks=fallback,
                   qp_status_counts=dict(Counter(r[0] for r in logs)), qp_seconds_median=float(np.median(ti)),
                   crossing_start=crossing_start.tolist(), compact_start=compact_start.tolist(), events=events,
                   wall_seconds=perf_counter()-begun, config=sc['config'])
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=1), encoding='utf-8')
    np.savez_compressed(out/'trajectory.npz', trajectory=traj, commands=U, hdv=np.array(hhist), hdv_acc=np.array(hahist),
                        margin_cav=np.array(m_cc), margin_hdv=np.array(m_ch), body_cav=np.array(b_cc), body_hdv=np.array(b_ch),
                        cav=np.array(cav), hdv_index=np.array(H))
    return summary
