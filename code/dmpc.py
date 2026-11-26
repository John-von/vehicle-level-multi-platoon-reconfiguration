"""Coupled, sequential distributed receding-horizon QPs.

Each vehicle solves for its own accelerations and receives adjacent predicted
trajectories. Fixed phase-specific separating directions make every pairwise
constraint linear. Every accepted local update preserves the currently shared
joint plan. Exact-model, synchronous, reliable trajectory exchange only.
"""

import osqp
import numpy as np
from scipy import sparse
from time import perf_counter


def matrices(cfg):
    h, H = cfg.dt, cfg.horizon
    A = np.array([[1,0,h,0],[0,1,0,h],[0,0,1,0],[0,0,0,1.]])
    B = np.array([[h*h/2,0],[0,h*h/2],[h,0],[0,h]])
    F = np.zeros((4*H, 2*H))
    E = np.zeros((4*H, 4))
    for k in range(H):
        E[4*k:4*k+4] = np.linalg.matrix_power(A,k+1)
        for j in range(k+1):
            F[4*k:4*k+4,2*j:2*j+2] = np.linalg.matrix_power(A,k-j) @ B
    return A, B, E, F


def phase_neighbors(sc, assignment, phase):
    n = len(sc['labels'])
    if phase == 'stage':
        chains = [np.flatnonzero(sc['source'] == s).tolist() for s in range(3)]
    elif phase == 'cross':
        chains = [assignment['order']]
    else:
        chains = assignment['target_orders']
    neighbors = [[] for _ in range(n)]
    for chain in chains:
        for front, rear in zip(chain[:-1], chain[1:]):
            neighbors[front].append((rear, 1))
            neighbors[rear].append((front, -1))
    return neighbors, chains


def reference(start, goal, duration, times, start_acc=None):
    times = np.asarray(times)
    p = start[:, :2]
    v = start[:, 2:]
    a = np.zeros_like(v) if start_acc is None else start_acc
    d = goal - p
    T = duration
    coef = [p, v, a/2,
            10*d/T**3-6*v/T**2-1.5*a/T,
            -15*d/T**4+8*v/T**3+1.5*a/T**2,
            6*d/T**5-3*v/T**4-.5*a/T**3]
    t = np.clip(times, 0, T)[None,:,None]
    pos = sum(c[:,None,:]*t**k for k,c in enumerate(coef))
    vel = sum(k*c[:,None,:]*t**(k-1) for k,c in enumerate(coef) if k)
    return np.concatenate([pos,vel], axis=2)


class LocalQP:
    def __init__(self, i, neighbors, corridor, cfg, E, F):
        self.i, self.neighbors, self.cfg = i, neighbors, cfg
        self.E, self.F = E, F
        H = cfg.horizon
        self.Gx, self.Gv = F[0::4], F[2::4]
        self.D = np.eye(2*H)-np.eye(2*H,k=-2)
        Q = np.tile([1.8,18.,.9,3.], H)
        Q[-4:] *= 3.
        self.Q = Q
        self.R = np.tile([.25,1.],H)
        self.J = np.tile([.9,2.],H) / cfg.dt**2
        self.couple_x, self.couple_v = .4, .15
        P = F.T@(Q[:,None]*F) + np.diag(self.R) + self.D.T@(self.J[:,None]*self.D)
        P += len(neighbors) * (self.couple_x*self.Gx.T@self.Gx + self.couple_v*self.Gv.T@self.Gv)
        th = np.tan(cfg.heading_max)
        self.heading = np.vstack([F[3::4]-th*F[2::4],-F[3::4]-th*F[2::4]])
        self.parts = [F, np.eye(2*H), self.D, F[-2:], self.heading]
        for _, sign in neighbors:
            self.parts.append(sign*self.Gx)
        self.A = np.vstack(self.parts)
        lo = np.tile([-np.inf,corridor[0],cfg.v_min-cfg.cruise,-cfg.vy_max],H)
        hi = np.tile([np.inf,corridor[1],cfg.v_max-cfg.cruise,cfg.vy_max],H)
        self.state_lo, self.state_hi = lo, hi
        self.solver = osqp.OSQP()
        self.solver.setup(P=sparse.csc_matrix(np.triu(2*P)), q=np.zeros(2*H),
                          A=sparse.csc_matrix(self.A), l=np.full(len(self.A),-np.inf),
                          u=np.full(len(self.A),np.inf), verbose=False, eps_abs=2e-6,
                          eps_rel=2e-7, max_iter=5000, polishing=False,
                          adaptive_rho=True, warm_starting=True)

    def solve(self, state, previous_u, ref, shared, warm):
        cfg, H = self.cfg, self.cfg.horizon
        base = self.E@state
        b = base.reshape(H,4)
        d0 = np.zeros(2*H)
        d0[:2] = previous_u
        q = self.F.T@(self.Q*(base-ref[self.i].reshape(-1))) - self.D.T@(self.J*d0)
        for j, sign in self.neighbors:
            desired_dx = ref[self.i,:,0] - ref[j,:,0]
            desired_dv = ref[self.i,:,2] - ref[j,:,2]
            q += self.couple_x*self.Gx.T@(b[:,0]-shared[j,:,0]-desired_dx)
            q += self.couple_v*self.Gv.T@(b[:,2]-shared[j,:,2]-desired_dv)
        limit = np.tile([cfg.ax_max,cfg.ay_max],H)
        # A shifted stationary terminal tail must also respect the jerk limit.
        limit[-2:] = np.minimum(limit[-2:],[cfg.jx_max*cfg.dt,cfg.jy_max*cfg.dt])
        jerk = np.tile([cfg.jx_max,cfg.jy_max],H)*cfg.dt
        th = np.tan(cfg.heading_max)
        heading_base = np.r_[b[:,3]-th*b[:,2],-b[:,3]-th*b[:,2]]
        lo = [self.state_lo-base,-limit,-jerk+d0,-b[-1,2:],np.full(2*H,-np.inf)]
        hi = [self.state_hi-base,limit,jerk+d0,-b[-1,2:],th*cfg.cruise-heading_base]
        for j, sign in self.neighbors:
            lo.append(cfg.safety_dx+.01+sign*(shared[j,:,0]-b[:,0]))
            hi.append(np.full(H,np.inf))
        l, u = np.concatenate(lo), np.concatenate(hi)
        self.solver.update(q=2*q,l=l,u=u)
        self.solver.warm_start(x=warm.reshape(-1))
        begin = perf_counter()
        result = self.solver.solve(raise_error=False)
        seconds = perf_counter()-begin
        violation = float('inf')
        if result.x is not None:
            ax = self.A@result.x
            violation = float(max(0.,np.max(l-ax),np.max(ax-u)))
        accepted = result.info.status_val in (1,2) and violation <= 3e-4
        record = dict(vehicle=self.i,status=result.info.status,accepted=accepted,
                      violation=violation,seconds=seconds,iterations=result.info.iter)
        if accepted:
            controls = result.x.reshape(H,2)
            prediction = (base+self.F@result.x).reshape(H,4)
            return controls,prediction,record
        return None,None,record


def initial_plan(states, E, F, cfg):
    # A common stopping profile preserves straight-lane order for this scenario.
    # These plans are independently audited before the first control is applied.
    H, h = cfg.horizon, cfg.dt
    u = np.zeros((len(states),H,2))
    u[:,:,:] = -states[:,None,2:] / (H*h)
    pred = np.stack([(E@x+F@a.reshape(-1)).reshape(H,4) for x,a in zip(states,u)])
    return u,pred


def joint_gap(shared, chains):
    return min((float(np.min(shared[a,:,0]-shared[b,:,0]))
                for chain in chains for a,b in zip(chain[:-1],chain[1:])),default=float('inf'))
