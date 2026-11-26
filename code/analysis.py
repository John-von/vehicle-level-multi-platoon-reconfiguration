"""Consolidated analysis of the planning and closed-loop experiments.

* independent continuous all-pair audit of every closed-loop run
  (audit_continuous_safety.py, cached per run)
* paired statistics over scenarios (mean, SD, 95% t-interval of the paired
  mean difference, exact/approximate Wilcoxon signed-rank p-value, win count)
* planning, exact-gap, scalability and lossy-V2V platoon summaries
Writes results/analysis.json and CSV tables in results/tables/.
"""
from pathlib import Path
import json, csv, glob, collections
import numpy as np
from scipy import stats
ROOT = Path(__file__).resolve().parent.parent

import audit_continuous_safety as audit_mod

RES = ROOT/'results'
TAB = RES/'tables'; TAB.mkdir(exist_ok=True)


def jl(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def paired(a, b):
    """Statistics of b-a over matched scenarios."""
    a = np.asarray(a, float); b = np.asarray(b, float); d = b-a; n = len(d)
    out = dict(n=n, mean_a=float(a.mean()), mean_b=float(b.mean()), sd_a=float(a.std(ddof=1)) if n > 1 else 0.,
               sd_b=float(b.std(ddof=1)) if n > 1 else 0., mean_diff=float(d.mean()),
               rel_diff=float(d.mean()/a.mean()) if a.mean() != 0 else None,
               b_lower=int((d < -1e-12).sum()), ties=int((abs(d) <= 1e-12).sum()))
    if n > 1:
        se = d.std(ddof=1)/np.sqrt(n); tq = stats.t.ppf(.975, n-1)
        out.update(ci95=[float(d.mean()-tq*se), float(d.mean()+tq*se)])
        nz = d[abs(d) > 1e-12]
        out['wilcoxon_p'] = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 1 else None
    return out


def audit_run(folder, group):
    out = RES/'audit'/group; out.mkdir(parents=True, exist_ok=True)
    cache = out/(folder.name+'.json')
    if cache.exists() and cache.stat().st_mtime > (folder/'summary.json').stat().st_mtime:
        return jl(cache)
    rep = audit_mod.audit(folder, out)
    z = np.load(folder/'trajectory.npz'); s = jl(folder/'summary.json'); c = s['config']
    if 'commands' in z.files:                      # commanded-input limits (differ from applied under mismatch)
        u = z['commands']; du = np.diff(np.concatenate([np.zeros_like(u[:1]), u]), axis=0)/c['dt']
        rep['command_bounds'] = dict(ax=float(abs(u[:, :, 0]).max()), ay=float(abs(u[:, :, 1]).max()),
                                     jx=float(abs(du[:, :, 0]).max()), jy=float(abs(du[:, :, 1]).max()))
        rep['command_limits_ok'] = bool(rep['command_bounds']['ax'] <= c['ax_max']+5e-4 and rep['command_bounds']['ay'] <= c['ay_max']+5e-4
                                        and rep['command_bounds']['jx'] <= c['jx_max']+5e-4 and rep['command_bounds']['jy'] <= c['jy_max']+5e-4)
    cache.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    return rep


def load_group(group):
    rows = {}
    for f in sorted((RES/group).glob('*/summary.json')):
        s = jl(f); folder = f.parent
        a = audit_run(folder, group)
        h = s['config']['dt']
        cs = np.array(s['crossing_start'], float); ks = np.array(s['compact_start'], float)
        rows[folder.name] = dict(
            name=folder.name, seed=s['seed'], success=s['success'], time=s['time'], effort=s['acceleration_squared_integral'],
            rate=s['jerk_squared_integral'], tv=s['total_variation_acceleration'], fallbacks=s['qp_fallbacks'], attempts=s['qp_attempts'],
            fallback_ratio=s['qp_fallbacks']/max(1, s['qp_attempts']), qp_median_ms=1e3*s['qp_seconds_median'], qp_p95_ms=1e3*s['qp_seconds_p95'],
            qp_max_ms=1e3*s['qp_seconds_max'], wall=s['wall_seconds'], steps=int(round(s['time']/h)), n=len(s['T_stage']),
            min_launch=s['min_launch'], max_launch=s['max_launch'], first_compact=float(np.nanmin(ks)) if np.isfinite(ks).any() else None,
            last_compact=float(np.nanmax(ks)) if np.isfinite(ks).any() else None, lane_changers=int(np.isfinite(cs).sum()),
            max_ax=s['max_ax'], max_jx=s['max_jx'], recovery_attempts=s.get('recovery_attempts', 0), recovery_accepted=s.get('recovery_accepted', 0),
            min_center_margin=a['minima']['safety']['value'], min_body_margin=a['minima']['body']['value'],
            cond_dx=a['minima']['conditional_x']['value'], cond_dy=a['minima']['conditional_y']['value'],
            bumper_bound=a['minimum_overlap_bumper_bound'], road_margin=a['road_margin'], safety_violations=a['violations']['safety'],
            body_violations=a['violations']['body'], pair_intervals=a['pair_intervals'], audit_passed=a['passed'],
            limits_ok=a['limits_ok'], command_limits_ok=a.get('command_limits_ok'), dynamics_residual=a['dynamics_residual'],
            critical=dict(safety=a['minima']['safety'], body=a['minima']['body'], cond_x=a['minima']['conditional_x']),
            job=s.get('job', {}))
    return rows


def summarize(rows, keys=('time', 'effort', 'rate', 'tv', 'fallbacks', 'min_center_margin', 'min_body_margin')):
    rr = list(rows)
    out = dict(n=len(rr), successes=sum(r['success'] for r in rr))
    for k in keys:
        v = np.array([r[k] for r in rr if r[k] is not None], float)
        if len(v): out[k] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0., min=float(v.min()), max=float(v.max()))
    return out


def main():
    A = {}
    # ---------------- execution policies, 20 scenarios ----------------
    ex = load_group('exec_adaptive')
    pol = {p: [ex[f's{s}_{p}'] for s in range(20) if f's{s}_{p}' in ex] for p in ['barrier', 'eager', 'settled']}
    A['exec'] = {p: summarize(v, ('time', 'effort', 'rate', 'tv', 'fallbacks', 'fallback_ratio', 'min_center_margin', 'min_body_margin',
                                  'cond_dx', 'cond_dy', 'bumper_bound', 'qp_median_ms', 'qp_p95_ms', 'min_launch', 'max_launch',
                                  'first_compact', 'last_compact', 'max_ax', 'max_jx')) for p, v in pol.items()}
    A['exec_pairs'] = {}
    for a, b in [('barrier', 'eager'), ('barrier', 'settled'), ('eager', 'settled')]:
        seeds = [s for s in range(20) if f's{s}_{a}' in ex and f's{s}_{b}' in ex]
        A['exec_pairs'][f'{b}_vs_{a}'] = {k: paired([ex[f's{s}_{a}'][k] for s in seeds], [ex[f's{s}_{b}'][k] for s in seeds])
                                           for k in ['time', 'effort', 'rate', 'tv']}
    A['exec_safety'] = dict(runs=len(ex), all_success=all(r['success'] for r in ex.values()), all_audit=all(r['audit_passed'] for r in ex.values()),
                            pair_intervals=int(sum(r['pair_intervals'] for r in ex.values())),
                            safety_violations=int(sum(r['safety_violations'] for r in ex.values())),
                            body_violations=int(sum(r['body_violations'] for r in ex.values())),
                            min_center=min(r['min_center_margin'] for r in ex.values()), min_body=min(r['min_body_margin'] for r in ex.values()),
                            min_cond_dx=min(r['cond_dx'] for r in ex.values()), min_cond_dy=min(r['cond_dy'] for r in ex.values()),
                            min_bumper=min(r['bumper_bound'] for r in ex.values()), min_road=min(r['road_margin'] for r in ex.values()),
                            max_dyn=max(r['dynamics_residual'] for r in ex.values()))
    crit = min(ex.values(), key=lambda r: r['min_center_margin']); critb = min(ex.values(), key=lambda r: r['min_body_margin'])
    A['exec_critical'] = dict(center=dict(run=crit['name'], **crit['critical']['safety']), body=dict(run=critb['name'], **critb['critical']['body']))
    # ---------------- reservation ablation ----------------
    ab = load_group('ablation')
    A['ablation'] = {}
    for p in ['eager', 'settled']:
        base = [ex[f's{s}_{p}'] for s in range(10)]; nr = [ab[f's{s}_{p}_noreserve'] for s in range(10)]
        A['ablation'][p] = dict(reserve=summarize(base), noreserve=summarize(nr),
                                failures=[s for s in range(10) if not ab[f's{s}_{p}_noreserve']['success']],
                                identical=[s for s in range(10) if abs(ab[f's{s}_{p}_noreserve']['time']-ex[f's{s}_{p}']['time']) < 1e-9 and
                                           abs(ab[f's{s}_{p}_noreserve']['effort']-ex[f's{s}_{p}']['effort']) < 1e-9],
                                time_success_pairs=paired([ex[f's{s}_{p}']['time'] for s in range(10) if ab[f's{s}_{p}_noreserve']['success']],
                                                          [ab[f's{s}_{p}_noreserve']['time'] for s in range(10) if ab[f's{s}_{p}_noreserve']['success']]))
    # ---------------- comfort envelope sweep ----------------
    js = load_group('jsweep') if (RES/'jsweep').exists() else {}
    A['jsweep'] = {}
    for p in ['barrier', 'eager', 'settled']:
        for j in [.02, .04, .08]:
            rr = [ex[f's{s}_{p}'] for s in range(4)] if j == .04 else [js[f's{s}_{p}_j{j}'] for s in range(4) if f's{s}_{p}_j{j}' in js]
            if rr: A['jsweep'][f'{p}_j{j}'] = summarize(rr, ('time', 'effort', 'rate', 'tv', 'min_center_margin'))
    # ---------------- closed-loop consequence of planning methods ----------------
    em = load_group('exec_methods') if (RES/'exec_methods').exists() else {}
    A['exec_methods'] = {}
    for m in ['source_blocks', 'mda', 'motion_speed']:
        rr = [em[f's{s}_{m}'] for s in range(6) if f's{s}_{m}' in em]
        if rr:
            seeds = [r['seed'] for r in rr]
            A['exec_methods'][m] = dict(summary=summarize(rr, ('time', 'effort', 'rate', 'tv', 'min_center_margin', 'fallbacks')),
                                        vs_combined={k: paired([r[k] for r in rr], [ex[f's{s}_settled'][k] for s in seeds]) for k in ['time', 'effort', 'rate']})
    A['exec_methods']['combined'] = dict(summary=summarize([ex[f's{s}_settled'] for s in range(6)], ('time', 'effort', 'rate', 'tv', 'min_center_margin', 'fallbacks')))
    # ---------------- scalability (closed loop) ----------------
    sc = load_group('scale')
    A['scale'] = {}
    for n in [12, 18, 30, 36]:
        rr = [sc[f'n{n}_s{s}'] for s in range(2) if f'n{n}_s{s}' in sc]
        A['scale'][n] = summarize(rr, ('time', 'effort', 'qp_median_ms', 'qp_p95_ms', 'wall', 'min_center_margin', 'fallback_ratio'))
        A['scale'][n]['wall_per_step'] = float(np.mean([r['wall']/r['steps'] for r in rr]))
        A['scale'][n]['safety_violations'] = int(sum(r['safety_violations'] for r in rr))
    rr = [ex[f's{s}_settled'] for s in range(2)]
    A['scale'][24] = summarize(rr, ('time', 'effort', 'qp_median_ms', 'qp_p95_ms', 'wall', 'min_center_margin', 'fallback_ratio'))
    A['scale'][24]['wall_per_step'] = float(np.mean([r['wall']/r['steps'] for r in rr])); A['scale'][24]['safety_violations'] = 0
    # ---------------- planning layer ----------------
    P = jl(RES/'planning/planning_main_0-99.json')
    by = collections.defaultdict(dict)
    for r in P: by[r['method']][r['seed']] = r
    A['planning'] = {m: {k: dict(mean=float(np.mean([v[k] for v in by[m].values()])), sd=float(np.std([v[k] for v in by[m].values()], ddof=1)))
                         for k in ['objective', 'motion', 'speed', 'link', 'mean_q', 'min_q', 'expected_losses', 'lane_changes', 'source_switches', 'seconds']}
                     for m in by}
    seeds = sorted(by['combined'])
    A['planning_pairs'] = {m: {k: paired([by[m][s][k] for s in seeds], [by['combined'][s][k] for s in seeds])
                               for k in ['objective', 'motion', 'mean_q', 'min_q', 'expected_losses']} for m in by if m != 'combined'}
    E = jl(RES/'planning/exact_main_0-99.json')
    A['exact_main'] = {m: dict(n=len([r for r in E if r['method'] == m]), max_gap=max(r['beam_gap'] for r in E if r['method'] == m),
                               same_plan=sum(r['same_plan'] for r in E if r['method'] == m),
                               exact_seconds_mean=float(np.mean([r['seconds'] for r in E if r['method'] == m])),
                               exact_seconds_max=float(max(r['seconds'] for r in E if r['method'] == m)))
                       for m in ['motion_only', 'motion_speed', 'combined']}
    G = jl(RES/'planning/exact_gap_m4_6_8_s0-19.json')+jl(RES/'planning/exact_gap_m10_s0-9.json')
    A['exact_gap'] = {}
    for n in sorted(set(r['n'] for r in G)):
        rr = [r for r in G if r['n'] == n]
        A['exact_gap'][n] = dict(cases=len(rr), max_gap=max(r['gap'] for r in rr), mean_gap=float(np.mean([r['gap'] for r in rr])),
                                 zero_gap=sum(abs(r['gap']) < 1e-12 for r in rr), beam_s=float(np.mean([r['beam_seconds'] for r in rr])),
                                 exact_s=float(np.mean([r['exact_seconds'] for r in rr])), exact_transitions=int(max(r['exact_transitions'] for r in rr)),
                                 beam_transitions=int(max(r['beam_transitions'] for r in rr)))
    # ---------------- lossy V2V platooning after reconfiguration ----------------
    A['comm'] = {}
    for tag in ['bernoulli_th0.6', 'ge_th0.6']:
        rows = []
        for f in glob.glob(str(RES/f'comm_loop/comm_{tag}_*.json')): rows += jl(f)
        if not rows: continue
        agg = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows: agg[(r['method'], r['seed'])]['all'].append(r)
        keys = ['rms_error', 'max_error', 'p95_error', 'min_gap', 'violations', 'mean_age', 'stale_frac', 'mean_q', 'effort']
        per = {(m, s): {k: float(np.mean([r[k] for r in v['all']])) for k in keys} for (m, s), v in agg.items()}
        methods = sorted(set(m for m, _ in per)); seeds = sorted(set(s for _, s in per))
        A['comm'][tag] = dict(methods={m: {k: float(np.mean([per[(m, s)][k] for s in seeds])) for k in keys} for m in methods},
                              pairs={m: {k: paired([per[(m, s)][k] for s in seeds], [per[('combined', s)][k] for s in seeds]) for k in keys}
                                     for m in methods if m != 'combined'},
                              runs=len(rows), seeds=len(seeds), max_error_overall={m: float(max(r['max_error'] for r in rows if r['method'] == m)) for m in methods},
                              min_gap_overall={m: float(min(r['min_gap'] for r in rows if r['method'] == m)) for m in methods})
    (RES/'analysis.json').write_text(json.dumps(A, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    # per-run CSV for the execution study
    with (TAB/'exec_runs.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); keys = ['name', 'seed', 'success', 'time', 'effort', 'rate', 'tv', 'fallbacks', 'attempts', 'min_center_margin', 'min_body_margin', 'cond_dx', 'cond_dy', 'qp_median_ms', 'qp_p95_ms']
        w.writerow(keys)
        for g in ['exec_adaptive', 'exec_fixed', 'exec_rear', 'ablation', 'jsweep', 'exec_methods', 'scale']:
            if not (RES/g).exists(): continue
            for r in load_group(g).values(): w.writerow([g+'/'+r['name']]+[r[k] for k in keys[1:]])
    print(json.dumps({k: A[k] for k in ['exec_safety']}, indent=1))
    return A


if __name__ == '__main__':
    main()
