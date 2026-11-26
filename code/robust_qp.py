"""Soft-constrained recovery QP used only when the hard local QP is infeasible.

Under plant mismatch (actuator lag, bounded acceleration disturbance) the
measured state can leave the tightened separation half-spaces or the narrow
lane-keeping band, which makes the hard local QP of event_mpc.EventQP primal
infeasible and freezes the vehicle on an open-loop shifted plan.  The recovery
problem keeps every input, input-rate, speed, heading and terminal constraint
hard and relaxes only (i) the collision half-spaces and (ii) the lateral
corridor by two scalar slacks with an exact L1 plus quadratic penalty.  A
recovery plan is accepted only if its continuous-time prediction satisfies the
*nominal* (untightened) separation rule against every shared trajectory, so the
nominal safety rule is never traded for feasibility.
"""
import numpy as np
from scipy import sparse
from time import perf_counter
from event_mpc import prediction_clearance

RHO1, RHO2 = 1e3, 1e5   # L1 and quadratic slack weights (objective units after scaling)


def soft_solve(qp, states, previous, ref, shared, controls, ylimits, lateral_intent, nominal_cfg):
    import clarabel
    i = qp.i; cfg = qp.cfg; H = cfg.horizon; F = qp.F
    base = qp.E@states[i]; b = base.reshape(H, 4)
    d0 = np.zeros(2*H); d0[:2] = previous[i]
    q = F.T@(qp.Q*(base-ref[i].reshape(-1)))-qp.D.T@(qp.J*d0)
    for j in qp.couple:
        q += cfg.coupling_position_weight*qp.Gx.T@(b[:, 0]-shared[j, :, 0]-(ref[i, :, 0]-ref[j, :, 0]))
        q += cfg.coupling_speed_weight*qp.Gv.T@(b[:, 2]-shared[j, :, 2]-(ref[i, :, 2]-ref[j, :, 2]))
    state_lo = np.tile([-np.inf, ylimits[0], cfg.v_min-cfg.cruise, -cfg.vy_max], H)
    state_hi = np.tile([np.inf, ylimits[1], cfg.v_max-cfg.cruise, cfg.vy_max], H)
    limit = np.tile([cfg.ax_max, cfg.ay_max], H)
    limit[-2:] = np.minimum(limit[-2:], [cfg.jx_max*cfg.dt, cfg.jy_max*cfg.dt])
    jerk = np.tile([cfg.jx_max, cfg.jy_max], H)*cfg.dt
    tan = np.tan(cfg.heading_max)
    hb = np.r_[b[:, 3]-tan*b[:, 2], -b[:, 3]-tan*b[:, 2]]
    lo = [state_lo-base, -limit, -jerk+d0, -b[-1, 2:], np.full(2*H, -np.inf), shared[i, :, 0]-b[:, 0]-2.]
    hi = [state_hi-base, limit, jerk+d0, -b[-1, 2:], tan*cfg.cruise-hb, shared[i, :, 0]-b[:, 0]+2.]
    nrow_fixed = sum(len(z) for z in lo)
    for j in qp.coll:
        delta = shared[i, :, :2]-shared[j, :, :2]
        xgood = abs(delta[:, 0]) >= cfg.safety_dx+.005
        ygood = abs(delta[:, 1]) >= cfg.safety_dy+.005
        axis = np.where(xgood & (lateral_intent or not np.all(ygood)), 0, 1)
        axis = np.where(~xgood, 1, axis); axis = np.where(~ygood, 0, axis)
        if j in qp.fixed_x:
            axis[:] = 0; delta[:, 0] = qp.fixed_x[j][0]
        x_threshold = qp.fixed_x[j][1] if j in qp.fixed_x else cfg.safety_dx+.02
        for dim, threshold in [(0, x_threshold), (1, cfg.safety_dy+.02)]:
            ll = np.full(H, -np.inf); uu = np.full(H, np.inf)
            plus = (axis == dim) & (delta[:, dim] >= 0); minus = (axis == dim) & ~plus
            ll[plus] = shared[j, plus, dim]+threshold-b[plus, dim]
            uu[minus] = shared[j, minus, dim]-threshold-b[minus, dim]
            lo.append(ll); hi.append(uu)
    l, u = np.concatenate(lo), np.concatenate(hi)
    A = qp.A
    m = len(A)
    # slack columns: 0 -> collision rows, 1 -> lateral corridor rows (state rows with index%4==1)
    slack = np.zeros((m, 2))
    slack[nrow_fixed:, 0] = 1.
    lat_rows = np.arange(0, 4*H)[np.arange(0, 4*H) % 4 == 1]
    slack[lat_rows, 1] = 1.
    equal = np.isfinite(l) & np.isfinite(u) & (abs(u-l) < 1e-10)
    lower = np.isfinite(l) & ~equal; upper = np.isfinite(u) & ~equal
    Az = np.hstack([A, np.zeros((m, 2))])
    rows = [Az[equal], np.hstack([A[upper], -slack[upper]]), np.hstack([-A[lower], -slack[lower]]),
            np.hstack([np.zeros((2, 2*H)), -np.eye(2)])]
    rhs = [u[equal], u[upper], -l[lower], np.zeros(2)]
    aa = sparse.csc_matrix(np.vstack(rows)); bb = np.concatenate(rhs)
    P = qp.P.toarray()                      # already scaled by 2/qp_objective_scale and upper-triangular
    Pz = np.zeros((2*H+2, 2*H+2)); Pz[:2*H, :2*H] = P; Pz[2*H, 2*H] = Pz[2*H+1, 2*H+1] = 2*RHO2
    qz = np.r_[2*q/cfg.qp_objective_scale, RHO1, RHO1]
    settings = clarabel.DefaultSettings(); settings.verbose = False; settings.max_iter = 100
    settings.tol_gap_abs = 1e-8; settings.tol_gap_rel = 1e-8; settings.tol_feas = 1e-8
    cones = [clarabel.ZeroConeT(int(equal.sum())), clarabel.NonnegativeConeT(int(upper.sum()+lower.sum()+2))]
    t0 = perf_counter()
    solved = clarabel.DefaultSolver(sparse.csc_matrix(np.triu(Pz)), qz, aa, bb, cones, settings).solve()
    seconds = perf_counter()-t0
    status = str(solved.status)
    record = dict(vehicle=i, status='Soft'+status, accepted=False, seconds=seconds)
    if status not in ('Solved', 'AlmostSolved'):
        return None, None, record
    z = np.array(solved.x); uvec = z[:2*H]; s = z[2*H:]
    # hard constraints (all rows except collision and lateral corridor) must hold
    ax = A@uvec
    soft_rows = (slack[:, 0] > 0) | (slack[:, 1] > 0)
    viol = np.maximum(0, np.maximum(l-ax, ax-u)); hard_violation = float(viol[~soft_rows].max(initial=0.))
    if hard_violation > 3e-4:
        record.update(hard_violation=hard_violation); return None, None, record
    cand = uvec.reshape(H, 2); pred = (base+F@uvec).reshape(H, 4)
    clearance = prediction_clearance(i, states, controls, shared, cand, pred, nominal_cfg)
    record.update(slack=s.tolist(), nominal_clearance=clearance, accepted=bool(clearance >= -3e-4))
    return (cand, pred, record) if record['accepted'] else (None, None, record)
