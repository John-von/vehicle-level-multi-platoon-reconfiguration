"""Human-driver car-following model calibrated on naturalistic driving data.

Intelligent driver model (IDM) with the parameter distributions calibrated by
Zhu, Wang, Tarko and Fang (2018, Transportation Research Part C 93, Table 2) on
2100 car-following periods of 42 drivers on Shanghai urban expressways.  The
parameters of each driver are drawn independently from a normal distribution with
the reported mean and standard deviation, truncated to the reported 5th and 95th
percentiles, and the desired speed is capped at the speed limit V_LIMIT.
"""
import numpy as np

# parameter: (mean, standard deviation, 5th percentile, 95th percentile), SI units
ZHU2018 = {
    'a': (0.8418, 0.2775, 0.5024, 1.3222),
    'v0': (105.6281/3.6, 24.7375/3.6, 74.0/3.6, 150.0/3.6),
    'delta': (3.5238, 3.6240, 1.0, 10.0),
    'b': (0.8150, 0.7251, 0.1177, 2.3618),
    's0': (1.5554, 0.9633, 0.1, 3.4247),
    'T': (0.8780, 0.3343, 0.3555, 1.4319),
}
B_PHYS = 8.0          # physical braking limit (m/s^2)
V_LIMIT = 80/3.6      # speed limit (m/s)


def sample_driver(rng):
    d = {}
    for k, (mu, sd, lo, hi) in ZHU2018.items():
        while True:
            v = rng.normal(mu, sd)
            if lo <= v <= hi:
                break
        d[k] = float(v)
    d['v0'] = min(d['v0'], V_LIMIT)
    return d


def idm_acc(v, gap, v_lead, p):
    """IDM acceleration for speed v, bumper-to-bumper gap and leader speed (None on a free road)."""
    free = 1.-(max(v, 0.)/p['v0'])**p['delta']
    if v_lead is None:
        return p['a']*free
    s_star = p['s0']+max(0., v*p['T']+v*(v-v_lead)/(2.*np.sqrt(p['a']*p['b'])))
    return max(p['a']*(free-(s_star/max(gap, .1))**2), -B_PHYS)


def equilibrium_gap(v, p):
    """Bumper-to-bumper gap at which the IDM driver keeps the constant speed v."""
    return (p['s0']+v*p['T'])/np.sqrt(max(1.-(v/p['v0'])**p['delta'], 1e-9))
