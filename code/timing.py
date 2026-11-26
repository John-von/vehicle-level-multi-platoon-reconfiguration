"""Per-vehicle comfort-limited reference durations and vectorized quintics.

A rest-to-rest quintic of duration T has peak jerk 60|d|/T^3 and peak
acceleration 10|d|/(sqrt(3) T^2).  With non-zero initial velocity and
acceleration the peaks are evaluated numerically.  Each vehicle receives
the shortest duration on the control grid whose reference stays inside the
declared comfort envelope (a_ref, j_ref); the hard QP limits are unchanged.
"""
import numpy as np


def quintic_coefficients(p, v, a, goal, T):
    d = goal - p
    c3 = 10*d/T**3 - 6*v/T**2 - 1.5*a/T
    c4 = -15*d/T**4 + 8*v/T**3 + 1.5*a/T**2
    c5 = 6*d/T**5 - 3*v/T**4 - .5*a/T**3
    return c3, c4, c5


def quintic_peaks(p, v, a, goal, T, samples=161):
    """Peak |acc| and |jerk| of 1-D quintics; p,v,a,goal,T broadcastable."""
    p, v, a, goal, T = np.broadcast_arrays(*[np.asarray(z, float) for z in (p, v, a, goal, T)])
    c3, c4, c5 = quintic_coefficients(p, v, a, goal, T)
    s = np.linspace(0., 1., samples)
    t = T[..., None]*s
    acc = a[..., None] + 6*c3[..., None]*t + 12*c4[..., None]*t**2 + 20*c5[..., None]*t**3
    jerk = 6*c3[..., None] + 24*c4[..., None]*t + 60*c5[..., None]*t**2
    return np.abs(acc).max(-1), np.abs(jerk).max(-1)


def comfort_duration(p, v, a, goal, a_ref, j_ref, dt, t_min, t_max):
    """Shortest grid duration with reference peaks inside (a_ref, j_ref)."""
    p = np.atleast_1d(np.asarray(p, float))
    grid = np.arange(np.ceil(t_min/dt), np.floor(t_max/dt)+1)*dt
    pa, pj = quintic_peaks(p[:, None], np.atleast_1d(v)[:, None], np.atleast_1d(a)[:, None],
                           np.atleast_1d(goal)[:, None], grid[None, :])
    # A reference cannot start below the current acceleration magnitude, so
    # the acceleration envelope is max(a_ref, |a0|); the jerk envelope is strict.
    limit = np.maximum(a_ref, np.abs(np.atleast_1d(a)))[:, None]+1e-9
    ok = (pa <= limit) & (pj <= j_ref)
    first = np.where(ok.any(1), ok.argmax(1), len(grid)-1)
    return grid[first]


def reference_v(start, goal, durations, elapsed, start_acc=None):
    """Quintic references with one duration and one elapsed clock per vehicle.

    start (n,4), goal (n,2), durations (n,), elapsed (n,H) -> (n,H,4).
    """
    p = start[:, :2]; v = start[:, 2:]
    a = np.zeros_like(v) if start_acc is None else start_acc
    T = np.asarray(durations, float)[:, None]
    d = goal - p
    coef = [p, v, a/2,
            10*d/T**3 - 6*v/T**2 - 1.5*a/T,
            -15*d/T**4 + 8*v/T**3 + 1.5*a/T**2,
            6*d/T**5 - 3*v/T**4 - .5*a/T**3]
    t = np.clip(np.asarray(elapsed, float), 0, T)[:, :, None]
    pos = sum(c[:, None, :]*t**k for k, c in enumerate(coef))
    vel = sum(k*c[:, None, :]*t**(k-1) for k, c in enumerate(coef) if k)
    return np.concatenate([pos, vel], axis=2)
