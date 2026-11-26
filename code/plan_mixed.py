"""Assignment and interleaving when human-driven vehicles (HDVs) are inside the source platoons.

An HDV keeps its lane and its place in the lane order, it is not driven to a
staging position, and its current longitudinal position in the moving frame is the
reference of the layout at planning time.  A target platoon in the lane of an HDV contains the HDV
as a member whose rank equals the number of connected vehicles that must stay
ahead of it, and the other positions are measured from the HDV.  Only connected
vehicles occupy the staging grid, and no connected vehicle may be staged, cross or
be placed inside the zone [x_h - d_rear, x_h + d_front] of an HDV in the lanes it
uses.  An HDV follows its predecessor and does not open a gap for a vehicle that
changes into its lane, so the member directly ahead of an HDV in the target
platoon comes from the source lane of the HDV, and no vehicle crosses the lane of
an HDV between the HDV and that member.  The state of the HDV-free problem is
extended by the index of the next free staging slot, so that slots inside a zone
can be skipped, and by a flag that records a crossing of the middle lane after
its last staying member, and the dynamic program remains exact.
"""
from time import perf_counter
import numpy as np

from scenario import Config, make_scenario
from driver_model import sample_driver, equilibrium_gap

ZONE = dict(front=41., rear=20.)   # front: 99th percentile of the calibrated following distance at cruise speed, rear: sensor-based follower


def mixed_scenario(seed, n_hdv, cfg=Config(), rng_hdv=None):
    """Source platoons with n_hdv HDVs at random inner ranks of distinct lanes (one per lane when possible)."""
    sc = make_scenario(seed, cfg); m = cfg.vehicles_per_source; n = 3*m
    rng = np.random.default_rng(1000+seed) if rng_hdv is None else rng_hdv
    lanes = rng.permutation(3)[:n_hdv] if n_hdv <= 3 else rng.integers(0, 3, n_hdv)
    hdv = sorted(int(l*m+rng.integers(1, m-1)) for l in lanes)
    drivers = {h: sample_driver(rng) for h in hdv}
    x = np.array(sc['initial'], float); jitter = x[:, 0]-(np.tile(-20.*np.arange(m), 3)+np.repeat([5., 0., -5.], m))
    for s in range(3):
        pos = 5.-5.*s
        for k in range(m):
            i = s*m+k
            if k > 0:
                prev = i-1
                if i in hdv: gap = cfg.length+equilibrium_gap(cfg.cruise, drivers[i])
                elif prev in hdv: gap = ZONE['rear']
                else: gap = 20.
                pos -= gap
            x[i, 0] = pos+(0. if i in hdv else jitter[i])
    for h in hdv: x[h, 2] = 0.
    center = x[:, 0].mean()
    sc.update(initial=x, hdv=hdv, hdv_driver=drivers, hdv_time_gap={h: d['T'] for h, d in drivers.items()},
              stage_x=center+(n-1)*cfg.stage_gap/2-np.arange(n)*cfg.stage_gap,
              target_x=center+(n//2-1)*cfg.final_gap/2-np.arange(n//2)*cfg.final_gap)
    return sc


def motion_term(sc, i, z, zf, d, cfg):
    x, _, v, _ = sc['initial'][i]
    a, b = z-x, zf-z; T, C = cfg.stage_time, cfg.compact_time
    e = 12*a*a/T**3-12*a*v/T**2+4*v*v/T+12*b*b/C**3
    e += 12*((d-sc['source'][i])*cfg.lane_width)**2/cfg.crossing_time**3
    return e/cfg.motion_scale


def target_layout(sc, cfg, hdv, K):
    """Positions by rank in each target lane and the fixed ranks of the HDVs in it.

    The connected vehicles ahead of an HDV in its lane cannot pass it, so the rank of
    the HDV equals their number and the positions in front of it and behind it are
    measured from the HDV.  Those ahead of an HDV in the lane that is cleared are
    counted for the adjacent target lane when that lane also has an HDV."""
    x0 = np.array(sc['initial'])[:, 0]; src = np.array(sc['source']); xbar = x0.mean(); m = cfg.vehicles_per_source
    ahead = {}
    for lane in range(3):
        hl = [h for h in hdv if src[h] == lane]
        if hl:
            lead = max(hl, key=lambda h: x0[h])
            ahead[lane] = sum(1 for i in range(lane*m, (lane+1)*m) if i not in hdv and x0[i] > x0[lead])
    need = {0: ahead.get(0, 0), 1: ahead.get(1, 0)}
    if 2 in ahead and 1 in ahead: need[1] += ahead[2]
    pos, rank_of = {}, {}
    for d in (0, 1):
        hd = sorted([h for h in hdv if src[h] == d], key=lambda h: -x0[h])
        cur = xbar+(K[d]-1)/2*cfg.final_gap; p = []; hi = 0
        if hd: cur = x0[hd[0]]+ZONE['front']+(need[d]-1)*cfg.final_gap
        for r in range(K[d]):
            if hi < len(hd) and cur < x0[hd[hi]]+ZONE['front']:
                rank_of[hd[hi]] = r; p.append(float(x0[hd[hi]])); cur = x0[hd[hi]]-ZONE['rear']; hi += 1
            else:
                p.append(float(cur)); cur -= cfg.final_gap
        if hi < len(hd): raise ValueError('target platoon too short for its HDVs')
        pos[d] = p
    return pos, rank_of


def optimize_mixed(sc, cfg=Config(), method='combined', hdv=None, q_sensor=.5, spare=6):
    t0 = perf_counter()
    hdv = set(sc.get('hdv', []) if hdv is None else hdv)
    n, m = len(sc['labels']), cfg.vehicles_per_source
    src = np.array(sc['source']); x0 = np.array(sc['initial'])[:, 0]; v0 = np.array(sc['initial'])[:, 2]
    q = np.array(sc['quality']); seq = [[s*m+k for k in range(m)] for s in range(3)]
    members = n-sum(1 for h in hdv if src[h] not in (0, 1)); K = [(members+1)//2, members//2]
    n_c = n-len(hdv); xbar = x0.mean(); G = n_c+(spare if hdv else 0)
    zs = xbar+(G-1)/2*cfg.stage_gap-np.arange(G)*cfg.stage_gap
    pos, rank_of = target_layout(sc, cfg, hdv, K)
    hdv_ranks = {d: {rank_of[h] for h in hdv if src[h] == d} for d in (0, 1)}
    w = np.array([1., 0. if method == 'motion_only' else cfg.weight_speed, cfg.weight_link if method == 'combined' else 0.])
    hdv_prefix = [np.cumsum([0]+[1 if i in hdv else 0 for i in seq[s]]) for s in range(3)]
    hdv_nt_prefix = [np.cumsum([0]+[1 if (i in hdv and src[i] not in (0, 1)) else 0 for i in seq[s]]) for s in range(3)]
    lane_hdv = {s: [h for h in hdv if src[h] == s] for s in range(3)}

    def pair_cost(j, i):
        if j < 0: return 0.
        sp = ((v0[i]-v0[j])/cfg.speed_scale)**2
        if i in hdv: lk = 0.
        elif j in hdv: lk = -np.log(q_sensor)/cfg.link_scale
        else: lk = -np.log(q[j, i])/cfg.link_scale
        return w[1]*sp+w[2]*lk

    def zone_ok(z, h, ahead):
        return z >= x0[h]+ZONE['front'] if ahead else z <= x0[h]-ZONE['rear']

    PC = np.zeros((n+1, n))
    for i in range(n):
        for j in range(n):
            if j != i: PC[j+1, i] = pair_cost(j, i)
    slot_ok = {}
    for s in range(3):
        for c in range(m+1):
            released = set(seq[s][:c])
            slot_ok[s, c] = [all(zone_ok(zs[g], h, h not in released) for h in lane_hdv[s]) for g in range(G)]
    cross_cache, mt_cache = {}, {}

    def cross_ok(s, d, g, r, c1):
        """Zones of the HDVs in the lanes entered or crossed; an HDV of a crossed lane counts as ahead until it is placed."""
        passing = s == 2 and d == 0
        key = (s, d, g, r, c1 if passing else -1)
        if key not in cross_cache:
            z, zf = zs[g], pos[d][r]; ok = True
            lo, hi = sorted((s, d))
            for l in range(lo, hi+1):
                if l == s: continue
                for h in lane_hdv[l]:
                    ahead = zf > x0[h] if l == d else seq[l].index(h) >= c1
                    if not zone_ok(z, h, ahead): ok = False
            cross_cache[key] = ok
        return cross_cache[key]

    def mt(i, g, d, r):
        key = (i, g, d, r)
        if key not in mt_cache: mt_cache[key] = w[0]*motion_term(sc, i, zs[g], pos[d][r], d, cfg)
        return mt_cache[key]

    hp = [[int(v) for v in hdv_prefix[s]] for s in range(3)]; hnp = [[int(v) for v in hdv_nt_prefix[s]] for s in range(3)]
    last_h1 = max([seq[1].index(h) for h in lane_hdv[1]], default=-1)
    layers = [{(0, 0, 0, 0, -1, -1, 0, 0): (0., None, None)}]
    expanded = 0
    for t in range(n):
        nxt = {}
        for st, (cost, _, _) in layers[-1].items():
            c = st[:3]; k0 = st[3]
            hrel = hp[0][c[0]]+hp[1][c[1]]+hp[2][c[2]]; g0 = st[6]; left = n_c-(t-hrel)
            mem = t-hnp[0][c[0]]-hnp[1][c[1]]-hnp[2][c[2]]; k = (k0, mem-k0)
            for s in range(3):
                if c[s] == m: continue
                i = seq[s][c[s]]
                if i in hdv:
                    if src[i] in (0, 1):
                        d = int(src[i])
                        if k[d] != rank_of[i]: continue
                        if st[4+d] >= 0 and src[st[4+d]] != d: continue
                        if d == 1 and st[7]: continue
                        val = cost+PC[st[4+d]+1, i]
                        new = list(st); new[s] += 1; new[3] += d == 0; new[4+d] = i; step = (i, d, -1, -1)
                    else:
                        val = cost; new = list(st); new[s] += 1; step = (i, -1, -1, -1)
                    key = tuple(new); expanded += 1
                    if key not in nxt or val < nxt[key][0]-1e-12: nxt[key] = (val, st, step)
                    continue
                ok_s = slot_ok[s, c[s]]
                for g in range(g0, G-left+1):
                    if not ok_s[g]: continue
                    for d in (0, 1):
                        r = k[d]
                        if r >= K[d] or r in hdv_ranks[d] or not cross_ok(s, d, g, r, c[1]): continue
                        val = cost+mt(i, g, d, r)+PC[st[4+d]+1, i]
                        new = list(st); new[s] += 1; new[3] += d == 0; new[4+d] = i; new[6] = g+1
                        if s == 1 and d == 1: new[7] = 0
                        elif s == 2 and d == 0 and c[1] <= last_h1: new[7] = 1
                        key = tuple(new); expanded += 1
                        if key not in nxt or val < nxt[key][0]-1e-12: nxt[key] = (val, st, (i, d, g, r))
        if not nxt: raise ValueError('no feasible plan with the HDV zones')
        layers.append(nxt)
    final = [(v, st) for st, (v, _, _) in layers[-1].items() if st[3] == K[0]]
    best, st = min(final, key=lambda z: z[0])
    path = []
    for t in range(n, 0, -1):
        _, prev, step = layers[t][st]; path.append(step); st = prev
    path = path[::-1]
    dest = np.full(n, -1); stage = x0.copy(); fin = x0.copy(); order = []; chains = {0: [], 1: []}
    for i, d, g, r in path:
        if d >= 0: chains[d].append(int(i))
        if i in hdv:
            if d >= 0: dest[i] = d
            continue
        dest[i] = d; stage[i] = zs[g]; fin[i] = pos[d][r]; order.append(int(i))
    links = [(a, b) for d in (0, 1) for a, b in zip(chains[d][:-1], chains[d][1:])]
    cav_links = [(a, b) for a, b in links if a not in hdv and b not in hdv]
    return dict(method=method, solver='exact dynamic programming with HDV members', optimal=True, hdv=sorted(hdv),
                destination=dest.tolist(), order=order, target_orders_full=[chains[0], chains[1]],
                target_orders=[[i for i in chains[d] if i not in hdv] for d in (0, 1)],
                hdv_rank={int(h): int(r) for h, r in rank_of.items()}, stage_x=stage.tolist(), final_x=fin.tolist(),
                optimized_objective=float(best), transitions=expanded, capacity=K,
                mean_link_quality=float(np.mean([q[a, b] for a, b in cav_links])),
                sensor_links=int(sum(1 for a, b in links if a in hdv)), seconds=perf_counter()-t0)


def plan_hdv(sc, cfg=Config(), spares=(6, 10, 14, 18), **kw):
    """Exact plan with the smallest number of spare staging slots that admits a feasible plan."""
    t0 = perf_counter()
    for k, sp in enumerate(spares):
        try:
            r = optimize_mixed(sc, cfg, spare=sp, **kw)
        except ValueError:
            if k == len(spares)-1: raise
            continue
        r.update(spare=sp, seconds=perf_counter()-t0)
        return r


if __name__ == '__main__':
    cfg = Config()
    for nh in (1, 2, 3):
        sc = mixed_scenario(0, nh, cfg)
        r = plan_hdv(sc, cfg)
        print(nh, sc['hdv'], r['hdv_rank'], r['spare'], round(r['seconds'], 1), r['transitions'], round(r['mean_link_quality'], 3))
