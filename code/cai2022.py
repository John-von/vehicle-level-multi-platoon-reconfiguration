"""Formation control with target assignment and conflict-based path search (Cai et al., 2022).

Reimplementation of the target assignment and path planning algorithm (TAPPA) of
Cai, Xu, Chen, Wang, Li, Wang and Wu (2022, Transportation Research Part C 136,
103513) for the reconfiguration studied here.  The vehicles move on a relative
grid that travels with the cruise speed, with one lane per row and the platoon
spacing D_f between columns.  In each step of duration T_F a vehicle stays, moves
one column or moves one lane (motion mode 1).  The assignment cost is the grid
distance under this motion, the assignments are enumerated in non-decreasing cost
by the Hungarian algorithm with Murty's partitioning, and each assignment is
routed by conflict-based search with space-time A*, which breaks ties between
paths of equal length by the number of conflicts with the other vehicles.  The
vertex, swap and follow-on conflicts of motion mode 1 are resolved.  The search
stops when the best conflict-free path set costs no more than the next assignment.

The lower level follows the minimum control energy problem with interior point
constraints of the original method.  The vehicles do not start on grid points, so
a first step moves every vehicle to its start cell in its own lane.  The
longitudinal motion is the clamped cubic spline through the grid points, starting
from the measured state and ending at the cruise speed, and a lane change is a
rest-to-rest quintic within its step.  T_F is the smallest candidate for which the
trajectories respect the acceleration, jerk, lateral speed, speed and separation
limits of the proposed controller.  The trajectories are executed without
tracking error.
"""
from pathlib import Path
import json, heapq, itertools, argparse, hashlib
from time import perf_counter
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import CubicSpline

from scenario import Config, jsonable

ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/'results'/'cai'
T_F_CANDIDATES = (4., 5., 6., 8., 10., 12., 14., 16., 20.)
BIG = 1e9


# ---------------------------------------------------------------- relative grid
def grid(sc, cfg):
    """Start cells, target cells and the grid origin (cell column 0 is the front target position)."""
    x0 = np.array(sc['initial'], float); n = len(x0); src = np.array(sc['source'], int)
    xref = float(sc['target_x'][0]); d = cfg.final_gap
    col = np.rint((xref-x0[:, 0])/d).astype(int)
    for lane in range(3):
        idx = sorted(np.flatnonzero(src == lane), key=lambda i: -x0[i, 0])
        for a, b in zip(idx[:-1], idx[1:]):
            if col[b] <= col[a]: col[b] = col[a]+1
    starts = [(int(src[i]), int(col[i])) for i in range(n)]
    K = [(n+1)//2, n//2]
    goals = [(lane, c) for lane in (0, 1) for c in range(K[lane])]
    return starts, goals, xref


def grid_distance(a, b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])


# ------------------------------------------------- assignments in order of cost
def _solve(C, include, exclude):
    C2 = C.copy()
    for i, j in exclude: C2[i, j] = BIG
    for i, j in include:
        keep = C[i, j]; C2[i, :] = BIG; C2[:, j] = BIG; C2[i, j] = keep
    r, c = linear_sum_assignment(C2)
    cost = C2[r, c].sum()
    if cost >= BIG/2: return None
    a = np.empty(len(r), int); a[r] = c
    return float(cost), a


def ranked_assignments(C):
    """Murty's method: assignments of the square cost matrix C in non-decreasing cost."""
    n = len(C); tie = itertools.count(); heap = []
    first = _solve(C, [], set())
    if first is None: return
    heapq.heappush(heap, (first[0], next(tie), first[1], [], frozenset()))
    while heap:
        cost, _, a, inc, exc = heapq.heappop(heap)
        yield cost, a
        fixed = {i for i, _ in inc}; inc_now = list(inc)
        for i in range(n):
            if i in fixed: continue
            exc_new = exc | {(i, int(a[i]))}
            sol = _solve(C, inc_now, exc_new)
            if sol is not None: heapq.heappush(heap, (sol[0], next(tie), sol[1], list(inc_now), exc_new))
            inc_now.append((i, int(a[i])))


# ------------------------------------------------------- conflict-based search
MOVES = ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0))


def space_time_astar(start, goal, bounds, vcon, econ, t_max, cat=None):
    """Shortest path in (cell, time) avoiding vertex constraints {(cell, t)} and move constraints {(cell, cell, t)}.

    Among paths of equal length the one with the fewest conflicts with the other
    agents (conflict avoidance table cat) is returned."""
    (l_lo, l_hi), (c_lo, c_hi) = bounds
    last = max([t for (cell, t) in vcon if cell == goal], default=-1)
    occ, stay, leave = cat if cat is not None else ({}, {}, set())
    tie = itertools.count(); best = {(start, 0): 0}; parent = {(start, 0): None}
    heap = [(grid_distance(start, goal), 0, 0, next(tie), start)]
    while heap:
        f, c, mt, _, cell = heapq.heappop(heap); t = -mt
        if best.get((cell, t), c) < c: continue
        if cell == goal and t > last:
            path = []; key = (cell, t)
            while key is not None:
                path.append(key[0]); key = parent[key]
            return path[::-1]
        if t >= t_max: continue
        for dl, dc in MOVES:
            nxt = (cell[0]+dl, cell[1]+dc)
            if not (l_lo <= nxt[0] <= l_hi and c_lo <= nxt[1] <= c_hi): continue
            if (nxt, t+1) in vcon or (cell, nxt, t) in econ: continue
            k = c+occ.get((nxt, t+1), 0)+(t+1 >= stay.get(nxt, t_max+1))+(nxt != cell and (nxt, t) in leave)
            if k < best.get((nxt, t+1), k+1):
                best[(nxt, t+1)] = k; parent[(nxt, t+1)] = (cell, t)
                heapq.heappush(heap, (t+1+grid_distance(nxt, goal), k, -(t+1), next(tie), nxt))
    return None


def avoidance_table(paths, skip):
    occ, stay, leave = {}, {}, set()
    for a, p in enumerate(paths):
        if a == skip: continue
        for t in range(len(p)-1):
            occ[(p[t], t)] = occ.get((p[t], t), 0)+1
            if p[t+1] != p[t]: leave.add((p[t], t))
        stay[p[-1]] = min(stay.get(p[-1], len(p)-1), len(p)-1)
    return occ, stay, leave


def at(path, t):
    return path[t] if t < len(path) else path[-1]


def conflicts(paths, first_only=True):
    """Vertex, swap and follow-on conflicts of motion mode 1."""
    T = max(len(p) for p in paths); found = []
    for t in range(T-1):
        now = {}
        for a, p in enumerate(paths): now[at(p, t)] = a
        nxt = {}
        for a, p in enumerate(paths):
            c1 = at(p, t+1)
            if c1 in nxt:
                found.append(('vertex', nxt[c1], a, c1, t+1))
                if first_only: return found
            nxt[c1] = a
        for a, p in enumerate(paths):
            a0, a1 = at(p, t), at(p, t+1)
            if a0 == a1 or a1 not in now: continue
            b = now[a1]; b0, b1 = at(paths[b], t), at(paths[b], t+1)
            if b0 == b1: continue
            found.append(('move', a, (a0, a1), b, (b0, b1), t))
            if first_only: return found
    return found


def cbs(starts, goals, bounds, t_max, deadline):
    n = len(starts); tie = itertools.count()
    cons = [(frozenset(), frozenset()) for _ in range(n)]
    paths = []
    for i in range(n):
        paths.append(space_time_astar(starts[i], goals[i], bounds, set(), set(), t_max, avoidance_table(paths, -1)))
        if paths[-1] is None: return None, 0
    cost = sum(len(p)-1 for p in paths)
    heap = [(cost, len(conflicts(paths, False)), next(tie), cons, paths)]; expanded = 0
    while heap:
        if perf_counter() > deadline: return None, expanded
        cost, _, _, cons, paths = heapq.heappop(heap); expanded += 1
        c = conflicts(paths)
        if not c: return (cost, paths), expanded
        c = c[0]
        if c[0] == 'vertex':
            _, a, b, cell, t = c; branches = [(a, ('v', (cell, t))), (b, ('v', (cell, t)))]
        else:
            _, a, ma, b, mb, t = c; branches = [(a, ('e', (ma[0], ma[1], t))), (b, ('e', (mb[0], mb[1], t)))]
        for agent, (kind, item) in branches:
            v, e = cons[agent]
            v2, e2 = (v | {item}, e) if kind == 'v' else (v, e | {item})
            p = space_time_astar(starts[agent], goals[agent], bounds, v2, e2, t_max, avoidance_table(paths, agent))
            if p is None: continue
            cons2 = list(cons); cons2[agent] = (v2, e2); paths2 = list(paths); paths2[agent] = p
            cost2 = cost-(len(paths[agent])-1)+(len(p)-1)
            heapq.heappush(heap, (cost2, len(conflicts(paths2, False)), next(tie), cons2, paths2))
    return None, expanded


def tappa(starts, goals, time_limit=600.):
    """Assignments in non-decreasing cost, each routed by CBS, until the best path set costs no more than the next assignment."""
    t0 = perf_counter(); deadline = t0+time_limit
    C = np.array([[grid_distance(s, g) for g in goals] for s in starts], float)
    lanes = [c[0] for c in starts+goals]; cols = [c[1] for c in starts+goals]
    bounds = ((min(lanes), max(lanes)), (min(cols)-2, max(cols)+2))
    t_max = int(C.max())+len(starts)+20
    best = None; k = 0; nodes = 0; limited = False; costs = []
    gen = ranked_assignments(C)
    nxt = next(gen, None)
    while nxt is not None:
        cost_a, a = nxt; k += 1
        if best is not None and best[0] <= cost_a: break
        res, ex = cbs(starts, [goals[j] for j in a], bounds, t_max, deadline); nodes += ex
        if res is not None:
            costs.append((cost_a, res[0]))
            if best is None or res[0] < best[0]: best = (res[0], res[1], a.copy())
        if perf_counter() > deadline:
            limited = True; break
        nxt = next(gen, None)
    return dict(best=best, assignments=k, cbs_nodes=nodes, time_limited=limited, seconds=perf_counter()-t0, costs=costs)


# ----------------------------------------------------------------- lower level
def quintic(s):
    return 10*s**3-15*s**4+6*s**5


def trajectories(sc, cfg, paths, xref, TF, dt=.05):
    """Positions, velocities, accelerations and jerks in the moving frame on a fine time grid."""
    x0 = np.array(sc['initial'], float); n = len(x0); w = cfg.lane_width; d = cfg.final_gap
    K = max(len(p) for p in paths)-1
    tk = np.arange(K+1)*TF; t = np.arange(0., K*TF+1e-9, dt)
    X = np.zeros((len(t), n, 4)); A = np.zeros((len(t), n, 2)); J = np.zeros((len(t), n, 2))
    for i, p in enumerate(paths):
        cells = [at(p, k) for k in range(K+1)]
        xs = np.array([xref-c[1]*d for c in cells], float); xs[0] = x0[i, 0]
        sp = CubicSpline(tk, xs, bc_type=((1, x0[i, 2]), (1, 0.)))
        X[:, i, 0] = sp(t); X[:, i, 2] = sp(t, 1); A[:, i, 0] = sp(t, 2); J[:, i, 0] = sp(t, 3)
        y = np.full(len(t), cells[0][0]*w); vy = np.zeros(len(t)); ay = np.zeros(len(t)); jy = np.zeros(len(t))
        for k in range(K):
            l0, l1 = cells[k][0], cells[k+1][0]
            if l0 == l1:
                y[t >= tk[k]] = l1*w; continue
            sel = (t >= tk[k]) & (t <= tk[k+1]); s = (t[sel]-tk[k])/TF; D = (l1-l0)*w
            y[sel] = l0*w+D*quintic(s)
            vy[sel] = D*(30*s**2-60*s**3+30*s**4)/TF
            ay[sel] = D*(60*s-180*s**2+120*s**3)/TF**2
            jy[sel] = D*(60-360*s+360*s**2)/TF**3
            y[t > tk[k+1]] = l1*w
        X[:, i, 1] = y; X[:, i, 3] = vy; A[:, i, 1] = ay; J[:, i, 1] = jy
    return t, X, A, J


def center_margin(X, cfg):
    """Smallest value over time and pairs of max(|dx|-D_x, |dy|-D_y); negative values violate the separation rule."""
    dx = np.abs(X[:, :, None, 0]-X[:, None, :, 0])-cfg.safety_dx
    dy = np.abs(X[:, :, None, 1]-X[:, None, :, 1])-cfg.safety_dy
    m = np.maximum(dx, dy); n = X.shape[1]; m[:, np.arange(n), np.arange(n)] = np.inf
    return float(m.min())


def feasible(X, A, J, cfg):
    v = cfg.cruise+X[:, :, 2]
    return (np.abs(A[:, :, 0]).max() <= cfg.ax_max+1e-9 and np.abs(A[:, :, 1]).max() <= cfg.ay_max+1e-9
            and np.abs(J[:, :, 0]).max() <= cfg.jx_max+1e-9 and np.abs(J[:, :, 1]).max() <= cfg.jy_max+1e-9
            and np.abs(X[:, :, 3]).max() <= cfg.vy_max+1e-9 and v.min() >= cfg.v_min and v.max() <= cfg.v_max
            and center_margin(X, cfg) >= 0.)


def plan_cached(starts, goals, time_limit):
    """TAPPA depends only on the grid problem, so scenarios with the same start cells share one search."""
    f = OUT/'plans'/(hashlib.sha1(json.dumps([starts, goals]).encode()).hexdigest()[:12]+'.json')
    if f.exists():
        r = json.loads(f.read_text(encoding='utf-8'))
        if json.dumps(r['problem']) == json.dumps([starts, goals]):
            if r['best'] is not None: r['best'] = (r['best'][0], [[tuple(c) for c in p] for p in r['best'][1]], np.array(r['best'][2]))
            return r
    r = tappa(starts, goals, time_limit)
    keep = dict(r, problem=[starts, goals], best=None if r['best'] is None else [r['best'][0], r['best'][1], r['best'][2].tolist()])
    f.parent.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(jsonable(keep)), encoding='utf-8')
    return r


def run(sc, out, time_limit=600.):
    sc = json.loads(json.dumps(sc)); cfg = Config(**sc['config'])
    starts, goals, xref = grid(sc, cfg)
    res = plan_cached(starts, goals, time_limit)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if res['best'] is None:
        summary = dict(seed=sc['seed'], success=False, failure='no conflict-free path set within the time limit', planner=res)
        (out/'summary.json').write_text(json.dumps(jsonable(summary), indent=1), encoding='utf-8'); return summary
    cost, grid_paths, a = res['best']
    # the vehicles do not start on grid points, so every vehicle first moves to its start cell in its own lane
    paths = [[p[0]]+list(p) for p in grid_paths]
    chosen = None
    for TF in T_F_CANDIDATES:
        t, X, A, J = trajectories(sc, cfg, paths, xref, TF)
        if feasible(X, A, J, cfg): chosen = TF; break
    if chosen is None:
        TF = T_F_CANDIDATES[-1]; t, X, A, J = trajectories(sc, cfg, paths, xref, TF)
    n = len(paths); dt = t[1]-t[0]; K = max(len(p) for p in paths)-1
    dest = np.array([goals[j][0] for j in a]); col = np.array([goals[j][1] for j in a])
    q = np.array(sc['quality'])
    orders = [[int(i) for i in sorted(np.flatnonzero(dest == lane), key=lambda i: col[i])] for lane in (0, 1)]
    links = [q[x, y] for o in orders for x, y in zip(o[:-1], o[1:])]
    lanes_seq = [[at(p, k)[0] for k in range(K+1)] for p in paths]
    lane_changes = [sum(1 for u, v in zip(s[:-1], s[1:]) if u != v) for s in lanes_seq]
    done = []
    for s in lanes_seq:
        ks = [k for k in range(K) if s[k] != s[k+1]]
        done.append(float((ks[-1]+1)*TF) if ks else None)
    step = int(round(cfg.dt/dt)); Xs = X[::step]
    summary = dict(seed=sc['seed'], success=chosen is not None, failure=None if chosen is not None else 'no step duration meets the limits',
                   method='TAPPA (Cai et al., 2022)', T_F=TF, steps=K, time=float(K*TF), completion=float(K*TF),
                   acceleration_squared_integral=float(np.sum(A**2)*dt), jerk_squared_integral=float(np.sum(J**2)*dt),
                   max_ax=float(np.abs(A[:, :, 0]).max()), max_ay=float(np.abs(A[:, :, 1]).max()),
                   max_jx=float(np.abs(J[:, :, 0]).max()), max_jy=float(np.abs(J[:, :, 1]).max()),
                   min_speed=float((cfg.cruise+X[:, :, 2]).min()), max_speed=float((cfg.cruise+X[:, :, 2]).max()),
                   min_center_margin=center_margin(X, cfg), lane_changes=lane_changes, crossing_done=done,
                   lane_range=[[int(min(s)), int(max(s))] for s in lanes_seq],
                   path_cost=float(cost), assignment_cost=float(sum(grid_distance(starts[i], goals[a[i]]) for i in range(n))),
                   planner=dict(seconds=res['seconds'], assignments=res['assignments'], cbs_nodes=res['cbs_nodes'],
                                time_limited=res['time_limited']),
                   assignment=dict(destination=dest.tolist(), target_orders=orders, mean_link_quality=float(np.mean(links)),
                                   seconds=res['seconds']),
                   paths=[[list(at(p, k)) for k in range(K+1)] for p in paths])
    (out/'summary.json').write_text(json.dumps(jsonable(summary), indent=1), encoding='utf-8')
    (out/'scenario.json').write_text(json.dumps(jsonable(sc)), encoding='utf-8')
    np.savez_compressed(out/'trajectory.npz', trajectory=Xs, fine_time=t, fine_acc=A[::2], fine_jerk=J[::2])
    return summary


def cases():
    RES = ROOT/'results'
    out = [(f'n{n}_s{s}', RES/'scale'/f'n{n}_s{s}') for n in (12, 18) for s in (0, 1)]
    out += [(f'n24_s{s}', RES/'exec_adaptive'/f's{s}_settled') for s in range(20)]
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--only', nargs='*'); ap.add_argument('--limit', type=float, default=600.)
    ap.add_argument('--worker', type=int, default=0); ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    for name, src in cases()[a.worker::a.workers]:
        if a.only and name not in a.only: continue
        out = OUT/'runs'/name
        if (out/'summary.json').exists(): continue
        sc = json.loads((src/'scenario.json').read_text(encoding='utf-8'))
        s = run(sc, out, a.limit)
        if s['success']:
            print(name, 'T_F', s['T_F'], 'T', s['completion'], 'Eu %.2f' % s['acceleration_squared_integral'], 'ax %.2f ay %.2f jx %.2f jy %.2f' % (s['max_ax'], s['max_ay'], s['max_jx'], s['max_jy']),
                  'margin %.2f' % s['min_center_margin'], 'q %.3f' % s['assignment']['mean_link_quality'], 'plan %.1f s' % s['planner']['seconds'],
                  'k', s['planner']['assignments'], 'nodes', s['planner']['cbs_nodes'], 'limited', s['planner']['time_limited'], flush=True)
        else:
            print(name, 'failed', s['failure'], flush=True)
