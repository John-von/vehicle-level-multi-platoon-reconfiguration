"""Lag-aware local QP for the individually triggered sequential DMPC.

The controller assumes a first-order actuator lag with time constant tau_hat,
    a_i(k) = a_i(k-1) + alpha_hat (u_i(k) - a_i(k-1)),  alpha_hat = 1 - exp(-h/tau_hat),
so that the applied acceleration sequence is affine in the commanded sequence,
    a = M u + c(a_i(k-1)),  M = kron(L, I_2),  L[l, m] = alpha_hat (1-alpha_hat)^(l-m).
Positions and velocities are predicted from the applied accelerations, the
commanded inputs keep their magnitude and rate limits, and the applied
acceleration at the end of the horizon is constrained to zero.  The shifted
plan with an appended zero command therefore keeps every vehicle at rest in
the cruise frame after the horizon, which preserves the recursive-feasibility
argument of the nominal design when tau_hat equals the true lag.

The constraint layout, admission logic and acceptance test are those of
event_mpc.EventQP and robust_qp.soft_solve.  The acceptance test integrates
the *applied* accelerations of every shared plan exactly over each interval.
"""
import numpy as np
from scipy import sparse
from time import perf_counter
import clarabel
from event_mpc import prediction_clearance

RHO1, RHO2 = 1e3, 1e5


def lag_operators(H, alpha):
    L = np.zeros((H, H))
    for l in range(H):
        for m in range(l+1): L[l, m] = alpha*(1-alpha)**(l-m)
    M = np.kron(L, np.eye(2))
    cc = np.repeat((1-alpha)**np.arange(1, H+1), 2)            # coefficient of a(k-1) per axis
    return M, cc


def applied_sequence(M, cc, U, a_prev):
    """Applied accelerations (H x 2) of commanded sequence U (H x 2) from a(k-1)."""
    return (M@U.reshape(-1)+cc*np.tile(a_prev, len(cc)//2)).reshape(-1, 2)


class LagQP:
    def __init__(self, i, coll, couple, cfg, E, F, fixed_x, alpha):
        self.i, self.coll, self.couple, self.cfg = i, coll, couple, cfg
        self.fixed_x = {} if fixed_x is None else fixed_x
        H = cfg.horizon; self.H = H
        self.M, self.cc = lag_operators(H, alpha)
        self.E, self.F0 = E, F
        Fl = F@self.M; self.F = Fl
        self.Gx, self.Gy, self.Gv = Fl[0::4], Fl[1::4], Fl[2::4]
        self.Q = np.tile([1.8, 18., .9, 3.], H); self.Q[-4:] *= 3
        self.R = np.tile([.25, 1.], H)
        self.J = np.tile([cfg.input_rate_weight_x, cfg.input_rate_weight_y], H)/cfg.dt**2
        self.D = np.eye(2*H)-np.eye(2*H, k=-2)
        P = Fl.T@(self.Q[:, None]*Fl)+np.diag(self.R)+self.D.T@(self.J[:, None]*self.D)
        P += len(couple)*(cfg.coupling_position_weight*self.Gx.T@self.Gx+cfg.coupling_speed_weight*self.Gv.T@self.Gv)
        self.P = sparse.csc_matrix(np.triu(2*P/cfg.qp_objective_scale))
        tan = np.tan(cfg.heading_max)
        heading = np.vstack([Fl[3::4]-tan*Fl[2::4], -Fl[3::4]-tan*Fl[2::4]])
        self.A = np.vstack([Fl, np.eye(2*H), self.D, Fl[-2:], self.M[-2:], heading, self.Gx,
                            *[G for _ in coll for G in (self.Gx, self.Gy)]])
        self.a_prev = np.zeros(2)

    def _assemble(self, states, previous, ref, shared, ylimits, lateral_intent):
        i, cfg, H = self.i, self.cfg, self.H
        cvec = self.cc*np.tile(self.a_prev, H)
        base = self.E@states[i]+self.F0@cvec; b = base.reshape(H, 4)
        d0 = np.zeros(2*H); d0[:2] = previous[i]
        q = self.F.T@(self.Q*(base-ref[i].reshape(-1)))-self.D.T@(self.J*d0)
        for j in self.couple:
            q += cfg.coupling_position_weight*self.Gx.T@(b[:, 0]-shared[j, :, 0]-(ref[i, :, 0]-ref[j, :, 0]))
            q += cfg.coupling_speed_weight*self.Gv.T@(b[:, 2]-shared[j, :, 2]-(ref[i, :, 2]-ref[j, :, 2]))
        state_lo = np.tile([-np.inf, ylimits[0], cfg.v_min-cfg.cruise, -cfg.vy_max], H)
        state_hi = np.tile([np.inf, ylimits[1], cfg.v_max-cfg.cruise, cfg.vy_max], H)
        limit = np.tile([cfg.ax_max, cfg.ay_max], H)
        limit[-2:] = np.minimum(limit[-2:], [cfg.jx_max*cfg.dt, cfg.jy_max*cfg.dt])
        jerk = np.tile([cfg.jx_max, cfg.jy_max], H)*cfg.dt
        tan = np.tan(cfg.heading_max)
        hb = np.r_[b[:, 3]-tan*b[:, 2], -b[:, 3]-tan*b[:, 2]]
        term_a = -cvec[-2:]
        lo = [state_lo-base, -limit, -jerk+d0, -b[-1, 2:], term_a, np.full(2*H, -np.inf), shared[i, :, 0]-b[:, 0]-2.]
        hi = [state_hi-base, limit, jerk+d0, -b[-1, 2:], term_a, tan*cfg.cruise-hb, shared[i, :, 0]-b[:, 0]+2.]
        nfixed = sum(len(z) for z in lo)
        for j in self.coll:
            delta = shared[i, :, :2]-shared[j, :, :2]
            xgood = abs(delta[:, 0]) >= cfg.safety_dx+.005
            ygood = abs(delta[:, 1]) >= cfg.safety_dy+.005
            axis = np.where(xgood & (lateral_intent or not np.all(ygood)), 0, 1)
            axis = np.where(~xgood, 1, axis); axis = np.where(~ygood, 0, axis)
            if j in self.fixed_x:
                axis[:] = 0; delta[:, 0] = self.fixed_x[j][0]
            x_threshold = self.fixed_x[j][1] if j in self.fixed_x else cfg.safety_dx+.02
            for dim, threshold in [(0, x_threshold), (1, cfg.safety_dy+.02)]:
                ll = np.full(H, -np.inf); uu = np.full(H, np.inf)
                plus = (axis == dim) & (delta[:, dim] >= 0); minus = (axis == dim) & ~plus
                ll[plus] = shared[j, plus, dim]+threshold-b[plus, dim]
                uu[minus] = shared[j, minus, dim]-threshold-b[minus, dim]
                lo.append(ll); hi.append(uu)
        return q, np.concatenate(lo), np.concatenate(hi), base, nfixed

    def _finish(self, uvec, base, states, controls, shared, cfg_check):
        H = self.H
        cand = uvec.reshape(H, 2); pred = (base+self.F@uvec).reshape(H, 4)
        acc = applied_sequence(self.M, self.cc, cand, self.a_prev)
        clearance = prediction_clearance(self.i, states, controls, shared, acc, pred, cfg_check)
        return cand, pred, acc, clearance

    def solve(self, states, previous, ref, shared, controls, ylimits, lateral_intent):
        cfg = self.cfg
        q, l, u, base, _ = self._assemble(states, previous, ref, shared, ylimits, lateral_intent)
        equal = np.isfinite(l) & np.isfinite(u) & (abs(u-l) < 1e-10)
        lower = np.isfinite(l) & ~equal; upper = np.isfinite(u) & ~equal
        aa = sparse.csc_matrix(np.vstack([self.A[equal], self.A[upper], -self.A[lower]]))
        bb = np.r_[u[equal], u[upper], -l[lower]]
        st = clarabel.DefaultSettings(); st.verbose = False; st.max_iter = 100
        st.tol_gap_abs = 1e-8; st.tol_gap_rel = 1e-8; st.tol_feas = 1e-8
        cones = [clarabel.ZeroConeT(int(equal.sum())), clarabel.NonnegativeConeT(int(upper.sum()+lower.sum()))]
        t0 = perf_counter()
        solved = clarabel.DefaultSolver(self.P, 2*q/cfg.qp_objective_scale, aa, bb, cones, st).solve()
        seconds = perf_counter()-t0; status = str(solved.status)
        x = np.array(solved.x); violation = float('inf'); clearance = None; ok = False; out = (None, None, None)
        if status == 'Solved' and np.isfinite(x).all():
            ax = self.A@x; violation = float(max(0, np.max(l-ax), np.max(ax-u)))
            if violation <= 3e-4:
                cand, pred, acc, clearance = self._finish(x, base, states, controls, shared, cfg)
                ok = clearance >= -3e-4
                if ok: out = (cand, pred, acc)
        rec = dict(vehicle=self.i, status=status, accepted=bool(ok), violation=violation, clearance=clearance,
                   seconds=seconds, iterations=solved.iterations, collision_neighbors=list(self.coll))
        return out+(rec,)

    def soft(self, states, previous, ref, shared, controls, ylimits, lateral_intent, nominal_cfg):
        cfg, H = self.cfg, self.H
        q, l, u, base, nfixed = self._assemble(states, previous, ref, shared, ylimits, lateral_intent)
        A = self.A; m = len(A)
        slack = np.zeros((m, 2)); slack[nfixed:, 0] = 1.
        lat_rows = np.arange(0, 4*H)[np.arange(0, 4*H) % 4 == 1]; slack[lat_rows, 1] = 1.
        equal = np.isfinite(l) & np.isfinite(u) & (abs(u-l) < 1e-10)
        lower = np.isfinite(l) & ~equal; upper = np.isfinite(u) & ~equal
        Az = np.hstack([A, np.zeros((m, 2))])
        rows = [Az[equal], np.hstack([A[upper], -slack[upper]]), np.hstack([-A[lower], -slack[lower]]),
                np.hstack([np.zeros((2, 2*H)), -np.eye(2)])]
        aa = sparse.csc_matrix(np.vstack(rows)); bb = np.concatenate([u[equal], u[upper], -l[lower], np.zeros(2)])
        Pz = np.zeros((2*H+2, 2*H+2)); Pz[:2*H, :2*H] = self.P.toarray(); Pz[2*H, 2*H] = Pz[2*H+1, 2*H+1] = 2*RHO2
        qz = np.r_[2*q/cfg.qp_objective_scale, RHO1, RHO1]
        st = clarabel.DefaultSettings(); st.verbose = False; st.max_iter = 100
        st.tol_gap_abs = 1e-8; st.tol_gap_rel = 1e-8; st.tol_feas = 1e-8
        cones = [clarabel.ZeroConeT(int(equal.sum())), clarabel.NonnegativeConeT(int(upper.sum()+lower.sum()+2))]
        t0 = perf_counter()
        solved = clarabel.DefaultSolver(sparse.csc_matrix(np.triu(Pz)), qz, aa, bb, cones, st).solve()
        rec = dict(vehicle=self.i, status='Soft'+str(solved.status), accepted=False, seconds=perf_counter()-t0)
        if str(solved.status) not in ('Solved', 'AlmostSolved'): return None, None, None, rec
        z = np.array(solved.x); uvec = z[:2*H]
        ax = A@uvec; soft_rows = (slack[:, 0] > 0) | (slack[:, 1] > 0)
        viol = np.maximum(0, np.maximum(l-ax, ax-u)); hard = float(viol[~soft_rows].max(initial=0.))
        if hard > 3e-4: rec.update(hard_violation=hard); return None, None, None, rec
        cand, pred, acc, clearance = self._finish(uvec, base, states, controls, shared, nominal_cfg)
        rec.update(slack=z[2*H:].tolist(), nominal_clearance=clearance, accepted=bool(clearance >= -3e-4))
        return (cand, pred, acc, rec) if rec['accepted'] else (None, None, None, rec)
