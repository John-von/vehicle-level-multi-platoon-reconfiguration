"""Post-reconfiguration platooning under lossy predecessor-to-follower V2V.

The new platoons produced by each planning method keep their computed
predecessor-follower edges.  Each follower runs the longitudinal channel of
the same local MPC (spacing and speed tracking, input and input-rate limits,
softened 8 m centre-gap constraint) using the predecessor's broadcast plan.
A broadcast reaches the follower with the per-cycle delivery probability
q_ji of that edge (independent Bernoulli or Gilbert-Elliott bursts with the
same stationary delivery ratio).  Without a fresh packet the follower keeps
the last received input sequence, shifted by its age and re-anchored at the
predecessor state measured by on-board ranging.  Random numbers are common
across planning methods (paired by follower position), so differences come
from the link qualities of the chosen adjacency only.
"""
from pathlib import Path
import json, argparse
import numpy as np
import osqp
from scipy import sparse
ROOT = Path(__file__).resolve().parent.parent

H, h = 20, .4
DF, DX = 14., 8.            # desired and minimum centre spacing (m)
AMAX, JMAX = 2., 1.5        # platooning comfort limits (m/s^2, m/s^3)
WX, WV, R, S, RHO = 1.8, .9, .25, .9, 1e4
TH = .6                     # constant time-headway gain on the relative speed (s)


def prediction_mats():
    Phi = np.zeros((H, 2)); Gx = np.zeros((H, H)); Gv = np.zeros((H, H))
    for l in range(1, H+1):
        Phi[l-1] = [1., l*h]
        for j in range(l):
            Gx[l-1, j] = (l-j-.5)*h*h; Gv[l-1, j] = h
    return Phi, Gx, Gv


PHI, GX, GV = prediction_mats()
DIFF = (np.eye(H)-np.eye(H, k=-1))/h


class Follower:
    """Variables z=[u(0..H-1), slack]; softened gap constraint."""
    def __init__(self, leader=False):
        self.leader = leader
        P = np.zeros((H+1, H+1))
        Pu = R*np.eye(H)+S*DIFF.T@DIFF+WV*GV.T@GV
        if not leader: Pu += WX*(GX+TH*GV).T@(GX+TH*GV)
        P[:H, :H] = Pu; P[H, H] = RHO
        A = [np.hstack([np.eye(H), np.zeros((H, 1))]), np.hstack([DIFF*h, np.zeros((H, 1))])]
        if not leader: A.append(np.hstack([GX, np.ones((H, 1))]))   # x + s >= ... handled as x - xp <= -DX + s
        A.append(np.r_[np.zeros(H), 1.][None, :])
        self.A = np.vstack(A)
        if not leader:  # gap row: x(l) - s <= xp(l) - DX  -> GX u - s <= xp - DX - base
            self.A[2*H:3*H, H] = -1.
        self.solver = osqp.OSQP()
        self.solver.setup(P=sparse.csc_matrix(2*np.triu(P)), q=np.zeros(H+1), A=sparse.csc_matrix(self.A),
                          l=-np.inf*np.ones(len(self.A)), u=np.inf*np.ones(len(self.A)), verbose=False,
                          eps_abs=1e-6, eps_rel=1e-6, max_iter=20000, polishing=False, warm_starting=True)

    def solve(self, s, uprev, xp=None, vp=None, vref=0.):
        bx = PHI@s; bv = s[1]+np.zeros(H)
        q = np.zeros(H+1)
        if self.leader:
            q[:H] = WV*GV.T@(bv-vref)
        else:
            q[:H] = WX*(GX+TH*GV).T@(bx+TH*bv-xp+DF)+WV*GV.T@(bv-vp)
        d0 = np.zeros(H); d0[0] = uprev/h
        q[:H] -= S*DIFF.T@d0
        lo = [-AMAX*np.ones(H), -JMAX*h*np.ones(H)+h*d0]
        hi = [AMAX*np.ones(H), JMAX*h*np.ones(H)+h*d0]
        if not self.leader:
            lo.append(-np.inf*np.ones(H)); hi.append(xp-DX-bx)
        lo.append([0.]); hi.append([np.inf])
        self.solver.update(q=2*q, l=np.concatenate(lo), u=np.concatenate(hi))
        r = self.solver.solve(raise_error=False)
        if r.x is None or not np.all(np.isfinite(r.x)):
            return None, None
        return r.x[:H], r.x[H]


def leader_reference(t, profile):
    t0, t1, dv = profile
    return -dv if t0 <= t < t1 else 0.


def simulate(qualities, U, steps=90, profile=(4., 14., 4.), mode='bernoulli', burst=3.):
    """One platoon; qualities[k] is the PDR of the link into follower k+1."""
    n = len(qualities)+1
    x = np.zeros((n, 2)); x[:, 0] = -DF*np.arange(n)
    ctrl = [Follower(leader=(k == 0)) for k in range(n)]
    plans = np.zeros((n, H)); uprev = np.zeros(n)
    last_plan = np.zeros((n, H)); age = np.ones(n, dtype=int)   # age of info about predecessor
    good = np.ones(n, dtype=bool)
    gaps = np.zeros((steps, n-1)); errs = np.zeros((steps, n-1)); acc = np.zeros((steps, n)); ages = np.zeros((steps, n-1))
    slack_used = 0
    for k in range(steps):
        t = k*h
        # delivery of the plans broadcast at the end of the previous cycle
        for f in range(1, n):
            qf = qualities[f-1]
            if mode == 'bernoulli':
                ok = U[k, f-1] < qf
            else:  # Gilbert-Elliott: Good->Bad with p, Bad->Good with r; stationary P(Good)=qf
                r = 1./burst; p = (1-qf)*r/max(qf, 1e-9)
                if good[f]: good[f] = U[k, f-1] >= p
                else: good[f] = U[k, f-1] < r
                ok = good[f]
            if ok: last_plan[f] = plans[f-1]; age[f] = 1
            else: age[f] += 1
        new = np.zeros((n, H))
        for f in range(n):
            if f == 0:
                vref = leader_reference(t, profile)
                u, _ = ctrl[0].solve(x[0], uprev[0], vref=vref)
            else:
                a = age[f]
                # the plan received with age a was computed a cycles ago; its first a
                # inputs are already in the past (the predecessor's measured state)
                stale = np.r_[last_plan[f][a:], np.zeros(a)] if a < H else np.zeros(H)
                sp = x[f-1]
                xp = PHI@sp+GX@stale; vp = sp[1]+GV@stale
                u, sl = ctrl[f].solve(x[f], uprev[f], xp, vp)
                if sl is not None and sl > 1e-4: slack_used += 1
            if u is None:   # keep shifted previous plan
                u = np.r_[plans[f][1:], 0.]
            new[f] = u
        plans = new
        a_now = plans[:, 0]
        acc[k] = a_now
        x = np.column_stack([x[:, 0]+h*x[:, 1]+.5*h*h*a_now, x[:, 1]+h*a_now])
        uprev = a_now
        gaps[k] = x[:-1, 0]-x[1:, 0]; errs[k] = gaps[k]-DF-TH*x[1:, 1]; ages[k] = age[1:]
    return dict(gaps=gaps, errs=errs, acc=acc, ages=ages, slack=slack_used)


def platoon_metrics(sim):
    e = sim['errs']; g = sim['gaps']
    peak = np.abs(e).max(0)
    return dict(rms_error=float(np.sqrt((e**2).mean())), max_error=float(peak.max()), p95_error=float(np.percentile(np.abs(e), 95)),
                min_gap=float(g.min()), violations=int((g < DX-1e-6).sum()),
                amplification=float(peak[-1]/max(peak[0], 1e-9)), effort=float((sim['acc']**2).sum()*h),
                mean_age=float(sim['ages'].mean()), stale_frac=float((sim['ages'] > 1).mean()))


def evaluate_plan(plan, U_by_platoon, mode='bernoulli', profile=(4., 14., 4.)):
    q = np.array(plan['scenario']['quality']); out = []
    for p, chain in enumerate(plan['assignment']['target_orders']):
        quals = np.array([q[a, b] for a, b in zip(chain[:-1], chain[1:])])
        sim = simulate(quals, U_by_platoon[p], mode=mode, profile=profile)
        m = platoon_metrics(sim); m.update(platoon=p, mean_q=float(quals.mean()), min_q=float(quals.min())); out.append(m)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(20)))
    ap.add_argument('--methods', nargs='+', default=['combined', 'motion_speed', 'mda', 'source_blocks'])
    ap.add_argument('--reps', type=int, default=10)
    ap.add_argument('--mode', default='bernoulli')
    ap.add_argument('--headway', type=float, default=None)
    ap.add_argument('--out', default=str(ROOT/'results/comm_loop'))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in a.seeds:
        rng = np.random.default_rng(5000+seed)
        U = rng.random((a.reps, 2, 90, 11))          # common random numbers, paired across methods
        for method in a.methods:
            plan = json.loads((ROOT/'results/planning/plans'/f's{seed}_{method}.json').read_text(encoding='utf-8'))
            if a.headway is not None:
                globals()['TH'] = a.headway
            for rep in range(a.reps):
                for m in evaluate_plan(plan, U[rep], a.mode):
                    m.update(seed=seed, method=method, rep=rep, mode=a.mode); rows.append(m)
        print('seed', seed, 'done', flush=True)
    tag = f"{a.mode}_th{TH}_s{a.seeds[0]}-{a.seeds[-1]}_r{a.reps}"
    (out/f'comm_{tag}.json').write_text(json.dumps(rows), encoding='utf-8')
