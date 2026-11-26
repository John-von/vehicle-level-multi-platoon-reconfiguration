"""Closed-loop simulation of the individually triggered sequential DMPC without human-driven vehicles.

The options cover the execution policy (common-phase, immediate or settled), the
rank-difference reservation or the single-gap rule, comfort-limited or uniform
reference durations, plant mismatch (discrete first-order actuator lag and bounded
disturbances), constraint tightening with soft recovery, and the lag-aware prediction.
"""
from pathlib import Path
import json, argparse, tempfile
from collections import Counter
from dataclasses import replace
from time import perf_counter
import numpy as np

from scenario import Config, jsonable
from dmpc import matrices, reference, initial_plan
from event_mpc import EventQP, select_neighbors, prediction_clearance, crossing_gate
from verify import geometric_clearance
from timing import comfort_duration, reference_v
from robust_qp import soft_solve

ROOT = Path(__file__).resolve().parent.parent
SMOOTH = dict(input_rate_weight_x=120., input_rate_weight_y=20., jx_max=.12, jy_max=.12,
              compact_time=60., horizon=25, coupling_position_weight=.02, coupling_speed_weight=.02,
              qp_solver='clarabel', qp_objective_scale=100.)
COMFORT = dict(a_ref=.3, j_ref=.04, t_min=8., t_max=90.)


def order_ready(i, states, shared, dest, target_rank, cfg, reserve=True):
    """Rank-difference reservation (reserve=True) or single-gap ablation."""
    peers = np.flatnonzero((dest == dest[i]) & (np.arange(len(states)) != i))
    signs = np.where(target_rank[i] < target_rank[peers], 1., -1.)
    gaps = np.abs(target_rank[i]-target_rank[peers]) if reserve else np.ones(len(peers))
    reserved = gaps*(cfg.safety_dx+.015)
    return bool(np.all(signs*(states[i, 0]-states[peers, 0]) >= reserved) and
                np.all(signs[:, None]*(shared[i, None, :, 0]-shared[peers, :, 0]) >= reserved[:, None]))


def run(plan, out, policy='event', settle=True, reserve=True, timing='adaptive', limit=210.,
        overrides=SMOOTH, comfort=COMFORT, lag=0., noise=0., margin=0., noise_seed=0, verbose=False,
        save_logs=False, recovery=False, lag_model=0., tol=(.08, .025)):
    sc = json.loads(json.dumps(plan['scenario'])); assignment = plan['assignment']
    if overrides: sc['config'].update(overrides)
    cfg = Config(**sc['config'])                      # nominal (audit) configuration
    ctrl = replace(cfg, safety_dx=cfg.safety_dx+margin, safety_dy=cfg.safety_dy+margin)
    x = np.array(sc['initial'], float); source = np.array(sc['source']); dest = np.array(assignment['destination'])
    n = len(x); h = cfg.dt; H = cfg.horizon; A, B, E, F = matrices(cfg)
    rank = np.empty(n, dtype=int); target_rank = np.empty(n, dtype=int)
    for r, i in enumerate(assignment['order']): rank[i] = r
    stage_goal = np.column_stack([assignment['stage_x'], source*cfg.lane_width])
    final_goal = np.column_stack([assignment['final_x'], dest*cfg.lane_width])
    coupling = [[] for _ in range(n)]
    for chain in assignment['target_orders']:
        for k, i in enumerate(chain): target_rank[i] = k
        for a, b in zip(chain[:-1], chain[1:]): coupling[a].append(b); coupling[b].append(a)
    if timing == 'adaptive':
        T_stage = comfort_duration(x[:, 0], x[:, 2], np.zeros(n), stage_goal[:, 0], comfort['a_ref'],
                                   comfort['j_ref'], h, comfort['t_min'], comfort['t_max'])
    else:
        T_stage = np.full(n, cfg.stage_time)
    T_compact = np.full(n, np.nan)
    rng = np.random.default_rng(noise_seed); alpha = 1.-np.exp(-h/lag) if lag > 0 else 1.
    applied = np.zeros((n, 2))
    lagaware = lag_model > 0
    if lagaware:
        from lag_qp import LagQP, lag_operators, applied_sequence
        alpha_hat = 1.-np.exp(-h/lag_model); LM, LC = lag_operators(H, alpha_hat); aplan = np.zeros((n, H, 2))
    previous = np.zeros((n, 2)); uplan, shared = initial_plan(x, E, F, cfg)
    assert min(prediction_clearance(i, x, uplan, shared, uplan[i], shared[i], ctrl) for i in range(n)) >= 0
    crossing_start = np.full(n, np.nan); crossing_done = np.full(n, np.nan)
    needs = source != dest; crossing_done[~needs] = 0.
    cross_initial = np.zeros_like(x); cross_acc = np.zeros((n, 2))
    compact_start = np.full(n, np.nan); compact_initial = np.zeros_like(x); compact_acc = np.zeros((n, 2))
    done = np.zeros(n, dtype=bool); barrier_released = False; barrier_compact = False
    qp_cache = {}; logs = []; events = []; hist = [x.copy()]; uhist = []; ahist = []; margins = []; modes = []
    fallback = 0; failure = None; begun = perf_counter(); messages = 0; last_print = 0.
    gate_checks = 0; gate_blocked = 0; recovery_attempts = 0; recovery_accepted = 0
    while len(uhist)*h < limit-1e-9:
        t = len(uhist)*h
        stage_ready = (np.max(abs(x[:, :2]-stage_goal), axis=1) < .08) & (np.max(abs(x[:, 2:]), axis=1) < .025)
        local_xready = (abs(x[:, 0]-stage_goal[:, 0]) < .08) & (abs(x[:, 2]) < .025) & (t >= T_stage-1e-9)
        if policy == 'barrier' and not barrier_released and np.all(stage_ready) and t >= T_stage.max()-1e-9:
            barrier_released = True
        steps = t+h*np.arange(1, H+1)
        ref = reference_v(np.array(sc['initial'], float), stage_goal, T_stage, np.tile(steps, (n, 1)))
        for i in range(n):
            if np.isfinite(crossing_start[i]):
                yg = final_goal[i:i+1].copy(); yg[0, 0] = cross_initial[i, 0]
                yr = reference(cross_initial[i:i+1], yg, cfg.crossing_time, t-crossing_start[i]+h*np.arange(1, H+1), cross_acc[i:i+1])[0]
                ref[i, :, 1] = yr[:, 1]; ref[i, :, 3] = yr[:, 3]
                if t-crossing_start[i] >= cfg.crossing_time and abs(x[i, 1]-final_goal[i, 1]) < .08 and abs(x[i, 3]) < .025:
                    if not np.isfinite(crossing_done[i]):
                        crossing_done[i] = t; events.append(dict(time=t, vehicle=i, type='lane_complete'))
            if np.isfinite(compact_start[i]):
                ref[i] = reference(compact_initial[i:i+1], final_goal[i:i+1], T_compact[i],
                                   t-compact_start[i]+h*np.arange(1, H+1), compact_acc[i:i+1])[0]
        if policy == 'barrier' and np.all(np.isfinite(crossing_done)) and np.all(local_xready): barrier_compact = True
        for i in range(n):
            ready = order_ready(i, x, shared, dest, target_rank, ctrl, reserve)
            pending = np.flatnonzero(needs & ~np.isfinite(crossing_done))
            dependencies = []
            ilo, ihi = sorted([x[i, 0], final_goal[i, 0]])
            for j in pending:
                if j == i or not min(source[j], dest[j]) <= dest[i] <= max(source[j], dest[j]): continue
                jlo, jhi = sorted([x[j, 0], stage_goal[j, 0]])
                if max(ilo, jlo)-min(ihi, jhi) < ctrl.safety_dx+.1: dependencies.append(int(j))
            can_compact = ready and np.isfinite(crossing_done[i]) and not dependencies
            if settle: can_compact = can_compact and local_xready[i]
            if policy == 'barrier': can_compact = local_xready[i] and np.isfinite(crossing_done[i]) and barrier_compact
            if can_compact and not np.isfinite(compact_start[i]):
                compact_start[i] = t; compact_initial[i] = x[i]; compact_acc[i] = previous[i]
                if timing == 'adaptive':
                    T_compact[i] = comfort_duration(x[i:i+1, 0], x[i:i+1, 2], previous[i:i+1, 0], final_goal[i:i+1, 0],
                                                    comfort['a_ref'], comfort['j_ref'], h, comfort['t_min'], comfort['t_max'])[0]
                else:
                    T_compact[i] = cfg.compact_time
                ref[i] = reference(x[i:i+1], final_goal[i:i+1], T_compact[i], h*np.arange(1, H+1), previous[i:i+1])[0]
                events.append(dict(time=t, vehicle=i, type='compact_start', duration=float(T_compact[i])))
        eligible = policy == 'event' or barrier_released
        for sweep in range(cfg.sweeps):
            order = assignment['order'] if (len(uhist)+sweep) % 2 == 0 else assignment['order'][::-1]
            for i in order:
                launch = False
                if needs[i] and not np.isfinite(crossing_start[i]) and eligible:
                    permitted, blockers = crossing_gate(i, x, shared, source, dest, rank, ctrl)
                    gate_checks += 1; gate_blocked += not permitted
                    if permitted:
                        launch = True
                        yg = final_goal[i:i+1].copy(); yg[0, 0] = x[i, 0]
                        yr = reference(x[i:i+1], yg, cfg.crossing_time, h*np.arange(1, H+1), previous[i:i+1])[0]
                        oldref = ref[i].copy(); ref[i, :, 1] = yr[:, 1]; ref[i, :, 3] = yr[:, 3]
                iscross = np.isfinite(crossing_start[i]) and not np.isfinite(crossing_done[i])
                fullroad = launch or iscross
                if fullroad:
                    half = (cfg.width+cfg.length*np.sin(cfg.heading_max))/2
                    ylimits = (-cfg.lane_width/2+half, 2.5*cfg.lane_width-half)
                else:
                    lane = dest[i] if np.isfinite(crossing_done[i]) else source[i]
                    ylimits = (lane*cfg.lane_width-.15, lane*cfg.lane_width+.15)
                neighbors = select_neighbors(i, x, uplan, shared, ctrl)
                gap = (lambda dr: abs(dr)*(ctrl.safety_dx+.02)) if reserve else (lambda dr: ctrl.safety_dx+.02)
                fixed_x = {j: ((1 if rank[i] < rank[j] else -1), gap(target_rank[i]-target_rank[j])) for j in range(n)
                           if j != i and dest[j] == dest[i] and (np.isfinite(compact_start[i]) or np.isfinite(compact_start[j]))}
                neighbors = sorted(set(neighbors) | set(fixed_x))
                key = (i, tuple(neighbors), tuple(fixed_x))
                if key not in qp_cache:
                    qp_cache[key] = LagQP(i, neighbors, coupling[i], ctrl, E, F, fixed_x, alpha_hat) if lagaware else EventQP(i, neighbors, coupling[i], ctrl, E, F, fixed_x)
                if lagaware:
                    qp_cache[key].a_prev = applied[i].copy()
                    uu, pp, acc, log = qp_cache[key].solve(x, previous, ref, shared, aplan, ylimits, fullroad)
                else:
                    uu, pp, log = qp_cache[key].solve(x, previous, ref, shared, uplan, ylimits, fullroad)
                if uu is None and recovery and not launch:
                    # soft recovery: relax only collision/corridor rows, accept only if the
                    # continuous prediction satisfies the nominal (untightened) rule
                    recovery_attempts += 1
                    if lagaware: uu, pp, acc, log = qp_cache[key].soft(x, previous, ref, shared, aplan, ylimits, fullroad, cfg)
                    else: uu, pp, log = soft_solve(qp_cache[key], x, previous, ref, shared, uplan, ylimits, fullroad, cfg)
                    recovery_accepted += uu is not None
                logs.append((log['status'], log['accepted'], log['seconds'], launch))
                if uu is not None:
                    uplan[i] = uu; shared[i] = pp; messages += len(set(neighbors+coupling[i]))
                    if lagaware: aplan[i] = acc
                    if launch:
                        crossing_start[i] = t; cross_initial[i] = x[i]; cross_acc[i] = previous[i]
                        events.append(dict(time=t, vehicle=i, type='lane_start'))
                else:
                    fallback += 1
                    if launch: ref[i] = oldref
        u = uplan[:, 0].copy()
        clearance, pair = geometric_clearance(x, u, cfg)
        if clearance < 0 and lag == 0 and noise == 0:
            failure = f'body separation failed before applying input: {clearance}, {pair}'; break
        # plant: discrete first-order actuator lag and bounded disturbance
        applied = applied+alpha*(u-applied) if lag > 0 else u.copy()
        if noise > 0:
            applied = applied+rng.uniform(-1, 1, (n, 2))*np.array([noise, .3*noise])
        margins.append(clearance); uhist.append(u.copy()); ahist.append(applied.copy())
        x = (A@x.T).T+(B@applied.T).T; hist.append(x.copy())
        mode = np.zeros(n, dtype=int); mode[np.isfinite(crossing_start)] = 1; mode[np.isfinite(crossing_done)] = 2; mode[np.isfinite(compact_start)] = 3
        modes.append(mode); previous = u
        uplan = np.concatenate([uplan[:, 1:], np.zeros((n, 1, 2))], axis=1)
        if lagaware:
            aplan = np.stack([applied_sequence(LM, LC, uplan[k], applied[k]) for k in range(n)])
            shared = np.stack([(E@xx+F@aa.reshape(-1)).reshape(H, 4) for xx, aa in zip(x, aplan)])
        else:
            shared = np.stack([(E@xx+F@uu.reshape(-1)).reshape(H, 4) for xx, uu in zip(x, uplan)])
        elapsed_ok = np.isfinite(compact_start) & ((t+h-np.nan_to_num(compact_start)) >= np.nan_to_num(T_compact)-1e-9)
        done = (np.max(abs(x[:, :2]-final_goal), axis=1) < tol[0]) & (np.max(abs(x[:, 2:]), axis=1) < tol[1]) & elapsed_ok
        if np.all(done): break
        if verbose and t-last_print >= 20:
            print('progress', round(t, 1), 'launched', int(np.isfinite(crossing_start).sum()), 'compact',
                  int(np.isfinite(compact_start).sum()), 'fallback', fallback, flush=True); last_print = t
    if failure is None and not np.all(done): failure = 'time limit before all vehicles completed'
    traj = np.array(hist); u = np.array(uhist); ap = np.array(ahist)
    du = np.diff(np.concatenate([np.zeros((1, n, 2)), u]), axis=0)/h
    ti = np.array([r[2] for r in logs]); started = crossing_start[needs]
    tv = float(np.abs(np.diff(np.concatenate([np.zeros_like(ap[:1]), ap]), axis=0)).sum())
    summary = dict(seed=sc['seed'], policy=policy, settle=settle, reserve=reserve, timing=timing, method=assignment.get('method'),
                   lag=lag, noise=noise, margin=margin, lag_model=lag_model, tol=list(tol), success=failure is None, failure=failure, time=len(u)*h,
                   T_stage=T_stage.tolist(), T_compact=T_compact.tolist(),
                   crossing_start=crossing_start.tolist(), crossing_done=crossing_done.tolist(), compact_start=compact_start.tolist(),
                   min_launch=float(np.nanmin(started)) if np.isfinite(started).any() else None,
                   max_launch=float(np.nanmax(started)) if np.isfinite(started).any() else None,
                   max_final_position_error=float(abs(x[:, :2]-final_goal).max()), max_final_velocity_error=float(abs(x[:, 2:]).max()),
                   acceleration_squared_integral=float(np.sum(ap*ap)*h), command_squared_integral=float(np.sum(u*u)*h),
                   jerk_squared_integral=float(np.sum(du*du)*h), total_variation_acceleration=tv,
                   max_ax=float(abs(ap[:, :, 0]).max()), max_ay=float(abs(ap[:, :, 1]).max()),
                   max_jx=float(abs(du[:, :, 0]).max()), max_jy=float(abs(du[:, :, 1]).max()),
                   min_speed=float((traj[:, :, 2]+cfg.cruise).min()), max_speed=float((traj[:, :, 2]+cfg.cruise).max()),
                   predicted_body_separation_bound=float(min(margins)), qp_attempts=len(logs), qp_fallbacks=fallback,
                   qp_status_counts=dict(Counter(r[0] for r in logs)), qp_seconds_median=float(np.median(ti)),
                   qp_seconds_p95=float(np.percentile(ti, 95)), qp_seconds_max=float(ti.max()),
                   gate_checks=gate_checks, gate_blocked=gate_blocked, recovery=recovery,
                   recovery_attempts=recovery_attempts, recovery_accepted=recovery_accepted,
                   trajectory_messages=messages, wall_seconds=perf_counter()-begun, events=events,
                   assignment=assignment, config=sc['config'], comfort=comfort if timing == 'adaptive' else None)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for name, data in [('summary', summary), ('scenario', sc)]:
        (out/(name+'.json')).write_text(json.dumps(jsonable(data), ensure_ascii=False, indent=1), encoding='utf-8')
    np.savez_compressed(out/'trajectory.npz', trajectory=traj, controls=ap, commands=u, final_goal=final_goal,
                        stage_goal=stage_goal, predicted_margin=np.array(margins), modes=np.array(modes))
    if save_logs:
        np.savez_compressed(out/'qp_log.npz', accepted=np.array([r[1] for r in logs]), seconds=ti)
    return summary


def selftest():
    """Reruns scenario 1 with uniform reference durations and compares with the stored summary."""
    plan = json.loads((ROOT/'results/planning/plans/s0_combined.json').read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory() as tmp:
        s = run(plan, Path(tmp), policy='event', settle=True, timing='fixed')
    ref = json.loads((ROOT/'results/exec_fixed/s0_settled/summary.json').read_text(encoding='utf-8'))
    for k in ['time', 'acceleration_squared_integral', 'jerk_squared_integral', 'qp_fallbacks', 'qp_attempts']:
        assert abs(s[k]-ref[k]) < 1e-5*max(1, abs(ref[k])), (k, s[k], ref[k])
    print('selftest passed', {k: s[k] for k in ['time', 'acceleration_squared_integral', 'qp_fallbacks']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    if a.selftest: selftest()
