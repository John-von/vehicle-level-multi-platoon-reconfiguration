"""Exact dynamic programming over source-order-preserving interleavings.

Both final predecessor links and intermediate staging positions are optimized.
This is an exact solution of this restricted model, not of unrestricted traffic
reconfiguration. Global assignment is centralized; execution is distributed.
"""
from functools import lru_cache
from time import perf_counter
import numpy as np
from scenario import Config


def local_terms(sc, i, g, rank, predecessor, cfg, destination=None):
    x, _, v, _ = sc['initial'][i]
    d = sc['stage_x'][g] - x
    d2 = sc['target_x'][rank] - sc['stage_x'][g]
    T, C = cfg.stage_time, cfg.compact_time
    # Unconstrained minimum integral(ax^2) for double-integrator transfers.
    # A surrogate only: bounds, jerk, coupling and quintic references change
    # actual closed-loop effort, which is measured separately.
    motion = (12*d*d/T**3 - 12*d*v/T**2 + 4*v*v/T + 12*d2*d2/C**3) / cfg.motion_scale
    destination = sc['route'][i] if destination is None else destination
    motion += 12 * ((destination - sc['source'][i])*cfg.lane_width)**2 / cfg.crossing_time**3 / cfg.motion_scale
    speed = 0. if predecessor is None else ((v - sc['initial'][predecessor, 2])/cfg.speed_scale)**2
    link = 0. if predecessor is None else -np.log(sc['quality'][predecessor, i])/cfg.link_scale
    return np.array([motion, speed, link], dtype=float)


def evaluate(sc, order, cfg=Config()):
    n = len(order)
    stage = np.empty(n)
    final = np.empty(n)
    rank = np.zeros(2, dtype=int)
    last = [None, None]
    terms = np.zeros(3)
    lanes = [[], []]
    per_vehicle = []
    for g, i in enumerate(order):
        route = int(sc['route'][i])
        z = local_terms(sc, i, g, rank[route], last[route], cfg)
        terms += z
        stage[i] = sc['stage_x'][g]
        final[i] = sc['target_x'][rank[route]]
        per_vehicle.append(dict(vehicle=int(i), stage_rank=g, target_lane=route,
                                target_rank=int(rank[route]), predecessor=last[route], terms=z.tolist()))
        rank[route] += 1
        last[route] = int(i)
        lanes[route].append(int(i))
    links = [(a, b) for lane in lanes for a, b in zip(lane[:-1], lane[1:])]
    switches = sum(sc['source'][a] != sc['source'][b] for a, b in links)
    return dict(order=list(map(int, order)), target_orders=lanes, stage_x=stage.tolist(),
                final_x=final.tolist(), terms=terms.tolist(),
                common_objective=float(terms @ [1, cfg.weight_speed, cfg.weight_link]),
                mean_link_quality=float(np.mean([sc['quality'][a,b] for a,b in links])),
                worst_link_quality=float(min(sc['quality'][a,b] for a,b in links)),
                source_switches=int(switches), links=links, assignments=per_vehicle)


def optimize(sc, method='combined', cfg=Config(), link_weight=None):
    started = perf_counter()
    n = len(sc['labels'])
    m = cfg.vehicles_per_source
    if method == 'source_blocks':
        order = list(range(n))
        result = evaluate(sc, order, cfg)
        result.update(method=method, solver='fixed baseline', states=0, optimal=False)
    elif method == 'position':
        order = np.argsort(-sc['initial'][:, 0]).tolist()
        result = evaluate(sc, order, cfg)
        result.update(method=method, solver='sorted current longitudinal position', states=0, optimal=False)
    else:
        weights = np.array([1., cfg.weight_speed if method != 'motion_only' else 0.,
                            cfg.weight_link if method == 'combined' else 0.])
        if link_weight is not None:
            weights[2] = link_weight
        route = sc['route'].reshape(3, m)
        prefixes = np.zeros((3, m + 1, 2), dtype=int)
        latest = np.full((3, m + 1, 2), -1, dtype=int)
        for s in range(3):
            for k in range(1, m + 1):
                prefixes[s,k] = prefixes[s,k-1]
                latest[s,k] = latest[s,k-1]
                dest = route[s,k-1]
                prefixes[s,k,dest] += 1
                latest[s,k,dest] = s*m+k-1
        @lru_cache(None)
        def solve(counts, last_sources):
            g = sum(counts)
            if g == n:
                return 0., ()
            best = (float('inf'), ())
            for s in range(3):
                if counts[s] == m:
                    continue
                i = s*m+counts[s]
                dest = int(sc['route'][i])
                r = sum(prefixes[t, counts[t], dest] for t in range(3))
                pred_source = last_sources[dest]
                pred = None if pred_source == -1 else int(latest[pred_source, counts[pred_source], dest])
                cost = float(local_terms(sc, i, g, r, pred, cfg) @ weights)
                nxt = list(counts)
                nxt[s] += 1
                last = list(last_sources)
                last[dest] = s
                future, tail = solve(tuple(nxt), tuple(last))
                candidate = cost + future
                if candidate < best[0] - 1e-12:
                    best = (candidate, (i,) + tail)
            return best
        value, order = solve((0,0,0), (-1,-1))
        result = evaluate(sc, order, cfg)
        result.update(method=method, solver='exact dynamic programming within declared interleavings',
                      states=solve.cache_info().currsize, optimal=True,
                      optimized_objective=value, optimization_weights=weights.tolist())
        assert abs(value - np.array(result['terms']) @ weights) < 1e-8
    result['seconds'] = perf_counter() - started
    for s in range(3):
        assert [i for i in result['order'] if sc['source'][i] == s] == list(range(s*m,(s+1)*m))
    return result


def optimize_joint(sc, method='combined', cfg=Config(), beam_width=1600, link_weight=None):
    """Joint destination + order beam dynamic program, with admissible ranking LB.

    Vehicle destination is a decision. Balanced target capacities are hard.
    Source-lane order is preserved. Pruning makes the joint solver heuristic;
    no global optimality claim is made. An exact conditional ordering pass is
    accepted only if it reduces cost with the chosen membership held fixed.
    """
    import heapq
    started = perf_counter()
    n, m = len(sc['labels']), cfg.vehicles_per_source
    capacity = n // 2
    weights = np.array([1., 0. if method == 'motion_only' else cfg.weight_speed,
                        cfg.weight_link if method == 'combined' else 0.])
    if link_weight is not None:
        weights[2] = link_weight
    motion = np.empty((n,n,capacity,2))
    for i in range(n):
        for g in range(n):
            for r in range(capacity):
                for dest in range(2):
                    motion[i,g,r,dest] = local_terms(sc,i,g,r,None,cfg,dest)[0]
    pair = weights[1]*((sc['initial'][:,None,2]-sc['initial'][None,:,2])/cfg.speed_scale)**2
    pair += weights[2]*(-np.log(sc['quality'])/cfg.link_scale)
    # suffix minimum relaxes both assignment uniqueness and source precedence.
    suffix = np.minimum.accumulate(motion[:,::-1,::-1,:],axis=1)
    suffix = np.minimum.accumulate(suffix,axis=2)[:,::-1,::-1,:]
    @lru_cache(None)
    def lower_bound(counts,r0):
        g = sum(counts)
        if g == n:
            return 0.
        r1 = g-r0
        unused = [s*m+k for s in range(3) for k in range(counts[s],m)]
        a = suffix[unused,g,min(r0,capacity-1),0] if r0 < capacity else np.full(len(unused),np.inf)
        b = suffix[unused,g,min(r1,capacity-1),1] if r1 < capacity else np.full(len(unused),np.inf)
        return float(np.minimum(a,b).sum())
    # state: consumed source counts, target-0 count, latest vehicle in each target
    beam = {(0,0,0,0,-1,-1):(0.,())}
    expanded, pruned = 0, False
    for g in range(n):
        candidates = {}
        for state,(cost,path) in beam.items():
            counts, r0 = state[:3], state[3]
            for s in range(3):
                if counts[s] == m:
                    continue
                i = s*m+counts[s]
                for dest in range(2):
                    r = r0 if dest == 0 else g-r0
                    if r == capacity:
                        continue
                    pred = state[4+dest]
                    value = cost+motion[i,g,r,dest]+(pair[pred,i] if pred >= 0 else 0.)
                    nxt = list(state)
                    nxt[s] += 1
                    nxt[3] += dest == 0
                    nxt[4+dest] = i
                    key = tuple(nxt)
                    if key not in candidates or value < candidates[key][0]-1e-12:
                        candidates[key] = (value,path+((i,dest),))
                    expanded += 1
        if len(candidates) > beam_width:
            keys = heapq.nsmallest(beam_width,candidates,
                    key=lambda k:(candidates[k][0]+lower_bound(k[:3],k[3]),k))
            beam = {k:candidates[k] for k in keys}
            pruned = True
        else:
            beam = candidates
    best_cost, path = min(beam.values(),key=lambda z:z[0])
    destination = np.zeros(n,dtype=int)
    for i,d in path:
        destination[i] = d
    routed = dict(sc,route=destination)
    result = evaluate(routed,[i for i,d in path],cfg)
    # Conditional ordering has only about 4,000 states and is solved exactly.
    conditional = optimize(routed,method,cfg,link_weight=link_weight)
    if np.array(conditional['terms'])@weights < best_cost-1e-9:
        result = conditional
    result.update(method=method,destination=destination.tolist(),
                  solver='joint beam dynamic programming + exact conditional ordering',
                  optimal=False,beam_width=beam_width,pruning_used=pruned,
                  transitions=expanded,optimization_weights=weights.tolist(),
                  optimized_objective=float(np.array(result['terms'])@weights),
                  seconds=perf_counter()-started)
    assert np.bincount(destination,minlength=2).tolist() == [capacity,capacity]
    return result
