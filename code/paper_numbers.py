"""Write results.tex and tab_cai_rows.tex of both manuscripts (manuscript/zh, manuscript/en).

Every number quoted in the text and the data rows of the comparison table are computed
here from the result files and rounded to two decimals."""
from pathlib import Path
import json, re
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; RES = ROOT/'results'



def jl(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def f2(x): return f'{x:.2f}'
def pct(a, b): return 100.*(a-b)/b
def sci(p):
    if p >= .01: return f2(p)
    m, e = f'{p:.2e}'.split('e'); return f'\\ensuremath{{{m}\\times10^{{{int(e)}}}}}'


def main():
    T = {}; A = jl(RES/'analysis.json')
    # planning, 100 scenarios
    P = jl(RES/'planning/planning_main_0-99.json')
    by = {}
    for r in P: by.setdefault(r['method'], {})[r['seed']] = r
    for m, k in [('source_blocks', 'sb'), ('mda', 'mda'), ('motion_only', 'm'), ('motion_speed', 'ms'), ('combined', 'c')]:
        rr = list(by[m].values())
        for key, name in [('motion', 'mot'), ('mean_q', 'q'), ('min_q', 'minq'), ('expected_losses', 'loss')]:
            T[f'pl_{k}_{name}'] = f2(np.mean([r[key] for r in rr]))
    seeds = sorted(by['combined'])
    mc = np.mean([by['combined'][s]['motion'] for s in seeds]); mm = np.mean([by['motion_speed'][s]['motion'] for s in seeds])
    T['pl_mot_inc'] = f2(pct(mc, mm))
    lc = np.mean([by['combined'][s]['expected_losses'] for s in seeds]); lm = np.mean([by['motion_speed'][s]['expected_losses'] for s in seeds])
    T['pl_loss_red'] = f2(-pct(lc, lm))
    T['pl_q_win'] = str(sum(by['combined'][s]['mean_q'] > by['motion_speed'][s]['mean_q'] for s in seeds))
    T['pl_n'] = str(len(seeds))
    sb = np.mean([by['source_blocks'][s]['motion'] for s in seeds]); T['pl_sb_mot_ratio'] = f2(sb/mc)
    eg = A['exact_gap']
    T['ex_trans_24'] = f"\\ensuremath{{{eg['24']['exact_transitions']/1e6:.2f}\\times10^{{6}}}}"
    T['ex_s_24'] = f2(eg['24']['exact_s']); T['ex_s_30'] = f2(eg['30']['exact_s']); T['ex_beam_s_30'] = f2(eg['30']['beam_s'])
    T['ex_gap_30'] = f2(100*eg['30']['max_gap']); T['ex_zero_30'] = str(eg['30']['zero_gap']); T['ex_n_30'] = str(eg['30']['cases'])
    bound = 9**3*13*25**2; T['ex_bound_24'] = f"\\ensuremath{{{bound/1e6:.2f}\\times10^{{6}}}}"
    T['ex_main_same'] = str(A['exact_main']['combined']['same_plan']); T['ex_main_s'] = f2(A['exact_main']['combined']['exact_seconds_mean'])
    # execution policies, 20 scenarios
    pols = ['barrier', 'eager', 'settled']; ab = {'barrier': 'b', 'eager': 'e', 'settled': 's'}
    ex = {(s, p): jl(RES/'exec_adaptive'/f's{s}_{p}/summary.json') for s in range(20) for p in pols}
    for p in pols:
        T[f'xp_{ab[p]}_t'] = f2(np.mean([ex[(s, p)]['time'] for s in range(20)]))
        T[f'xp_{ab[p]}_eu'] = f2(np.mean([ex[(s, p)]['acceleration_squared_integral'] for s in range(20)]))
        T[f'xp_{ab[p]}_ej'] = f2(np.mean([ex[(s, p)]['jerk_squared_integral'] for s in range(20)]))
    mean = lambda p, k: np.mean([ex[(s, p)][k] for s in range(20)])
    T['xp_sb_t'] = f2(-pct(mean('settled', 'time'), mean('barrier', 'time')))
    T['xp_sb_eu'] = f2(pct(mean('settled', 'acceleration_squared_integral'), mean('barrier', 'acceleration_squared_integral')))
    T['xp_se_eu'] = f2(-pct(mean('settled', 'acceleration_squared_integral'), mean('eager', 'acceleration_squared_integral')))
    T['xp_se_ej'] = f2(-pct(mean('settled', 'jerk_squared_integral'), mean('eager', 'jerk_squared_integral')))
    T['xp_es_t'] = f2(-pct(mean('eager', 'time'), mean('settled', 'time')))
    T['xp_sb_p'] = sci(A['exec_pairs']['settled_vs_barrier']['time']['wilcoxon_p'])
    T['xp_se_p'] = sci(A['exec_pairs']['settled_vs_eager']['effort']['wilcoxon_p'])
    T['xp_e_tmax'] = f2(max(ex[(s, 'eager')]['time'] for s in range(20)))
    S = A['exec_safety']
    T['saf_pairint'] = f"\\ensuremath{{{S['pair_intervals']/1e6:.2f}\\times10^{{6}}}}"
    T['saf_mu'] = f2(S['min_center']); T['saf_beta'] = f2(S['min_body']); T['saf_dx'] = f2(S['min_cond_dx']); T['saf_dy'] = f2(S['min_cond_dy'])
    # ablation and comfort sweep
    for p in ['eager', 'settled']:
        a = A['ablation'][p]; T[f'ab_{ab[p]}_ok'] = f"{a['noreserve']['successes']}/{a['noreserve']['n']}"; T[f'ab_{ab[p]}_okn'] = str(a['noreserve']['successes']); T[f'ab_{ab[p]}_n'] = str(a['noreserve']['n'])
    for p in pols:
        for j in ['0.02', '0.04', '0.08']:
            c = A['jsweep'].get(f'{p}_j{j}')
            if c: T[f'js_{ab[p]}_t{j[2:]}'] = f2(c['time']['mean']); T[f'js_{ab[p]}_e{j[2:]}'] = f2(c['effort']['mean'])
    for p in pols:
        uu = [jl(RES/'exec_fixed'/f's{s}_{p}/summary.json') for s in range(4)]
        T[f'js_{ab[p]}_tu'] = f2(np.mean([c['time'] for c in uu])); T[f'js_{ab[p]}_eu'] = f2(np.mean([c['acceleration_squared_integral'] for c in uu]))
    T['js_s_dtu'] = f2(float(T['js_s_tu'])-float(T['js_s_t04']))
    # mixed traffic
    B = RES/'mixed/batch'
    sh = jl(B/'showcase/summary.json'); st = jl(B/'showcase_steady/summary.json')
    T['sh_T'] = f2(sh['completion']); T['sh_T_steady'] = f2(st['completion']); T['sh_tdone'] = f2(sh['t_done'])
    T['sh_a0min'] = f2(float(np.load(B/'showcase/trajectory.npz')['frame'][:, 2].min())); T['sh_v0min'] = f2(sh['min_v0']); T['sh_a0max'] = f2(sh['max_abs_a0']); T['sh_axmax'] = f2(sh['max_ax_total'])
    T['sh_gap'] = f2(sh['min_gap_hdv']); T['sh_mcc'] = f2(sh['min_center_margin_cav']); T['sh_ay'] = f2(sh['max_ay'])
    z = np.load(B/'showcase/trajectory.npz'); X = z['trajectory']; v0 = z['frame'][:, 1]
    T['sh_vmin'] = f2((X[:, :, 2]+v0[:, None]).min()); T['sh_vmax'] = f2((X[:, :, 2]+v0[:, None]).max())
    cs = np.array(sh['crossing_start'], float); T['sh_lc_first'] = f2(np.nanmin(cs)); T['sh_lc_last'] = f2(np.nanmax(cs))
    T['sh_lc_n'] = str(int(np.isfinite(cs).sum()))
    conds = ['steady', 'b1_t15', 'b1_t30', 'b1_t50', 'b2_t30', 'noadapt_t30']
    R = {c: [jl(B/f's{s}_{c}/summary.json') for s in range(5) if (B/f's{s}_{c}/summary.json').exists()] for c in conds}
    for c in conds:
        k = c.replace('_', '')
        rr = R[c]; T[f'mx_{k}_n'] = str(len(rr))
        done = [r['completion'] for r in rr if r['success']]
        T[f'mx_{k}_ok'] = str(len(done)); T[f'mx_{k}_T'] = f2(np.mean(done)) if done else '--'
        T[f'mx_{k}_gap'] = f2(min(r['min_gap_hdv'] for r in rr)) if rr else '--'
        T[f'mx_{k}_v0'] = f2(min(r['min_v0'] for r in rr)) if rr else '--'
        T[f'mx_{k}_a0'] = f2(max(r['max_abs_a0'] for r in rr)) if rr else '--'
        T[f'mx_{k}_coll'] = str(sum((r['min_body_margin_hdv'] < 0) or (r['min_body_margin_cav'] < 0) for r in rr))
    adapt = [r for c in conds[:5] for r in R[c]]
    T['mx_runs'] = str(sum(len(v) for v in R.values())+2)
    T['mx_adapt_ok'] = f"{sum(r['success'] for r in adapt)}/{len(adapt)}"
    diffs = []
    for s in range(5):
        base = [r for r in R['steady'] if r['seed'] == s]
        if not base: continue
        for c in conds[1:5]:
            for r in R[c]:
                if r['seed'] == s and r['success']: diffs.append(abs(r['completion']-base[0]['completion']))
    T['mx_dT_max'] = f2(max(diffs)) if diffs else '--'
    T['mx_mcc'] = f2(min(r['min_center_margin_cav'] for r in adapt))
    T['mx_gap_min'] = f2(min(r['min_gap_hdv'] for r in adapt))
    T['mx_axmax'] = f2(max(r['max_ax_total'] for r in adapt))
    na = R['noadapt_t30']
    T['mx_na_dt'] = f2(np.mean([float(re.findall(r'[\d.]+', r['failure'])[0])-r['hdv']['onset'] for r in na if r['failure']])) if na else '--'
    T['mx_na_t'] = f2(np.mean([float(re.findall(r'[\d.]+', r['failure'])[0]) for r in na if r['failure']])) if na else '--'
    # human-driven vehicles inside the source platoons
    I = {}
    for f in sorted((RES/'inner'/'runs').glob('*/summary.json')):
        r = jl(f); I[(r['seed'], r['n_hdv'], r['aware'])] = r
    coll = lambda r: r.get('failure') is not None and 'body' in r['failure']
    aw_all = [r for (s_, n_, a_), r in I.items() if a_ and n_ > 0]
    aw = [r for r in aw_all if not r.get('plan_failed')]; nf = [r for r in aw_all if r.get('plan_failed')]
    ag = [r for (s_, n_, a_), r in I.items() if not a_]
    T['in_runs'] = str(len(I)); T['in_aw_n'] = str(len(aw)); T['in_ag_n'] = str(len(ag))
    T['in_aw_nf'] = str(len(nf)); T['in_aw_nall'] = str(len(aw_all))
    T['in_aw_ok'] = str(sum(r['success'] for r in aw)); T['in_ag_ok'] = str(sum(r['success'] for r in ag))
    T['in_aw_coll'] = str(sum(coll(r) for r in aw)); T['in_ag_coll'] = str(sum(coll(r) for r in ag))
    viol = lambda r: (not r['success']) and not coll(r) and min(r.get('min_center_margin_hdv') or 0., r['min_center_margin_cav']) < 0
    T['in_ag_viol'] = str(sum(viol(r) for r in ag)); T['in_aw_viol'] = str(sum(viol(r) for r in aw))
    T['in_ag_stall'] = str(sum((not r['success']) and not coll(r) and not viol(r) for r in ag))
    T['in_aw_stall'] = str(sum((not r['success']) and not coll(r) and not viol(r) for r in aw))
    for n_ in range(4):
        done = [r['completion'] for (s_, k_, a_), r in I.items() if a_ and k_ == n_ and r['success']]
        T[f'in_T{n_}'] = f2(np.mean(done)) if done else '--'
        g = [r for (s_, k_, a_), r in I.items() if (not a_) and k_ == n_]
        if g: T[f'in_ag_ok{n_}'] = str(sum(r['success'] for r in g)); T[f'in_ag_coll{n_}'] = str(sum(coll(r) for r in g)); T[f'in_ag_n{n_}'] = str(len(g))
    for n_ in range(4):
        rr = [r for (s_, k_, a_), r in I.items() if a_ and k_ == n_ and r['success']]
        if rr:
            T[f'in_lc{n_}'] = f2(np.mean([np.nanmax(np.array(r['crossing_start'], float)) for r in rr]))
            T[f'in_tail{n_}'] = f2(np.mean([r['completion']-np.nanmax(np.array(r['compact_start'], float)) for r in rr]))
    gaps = []
    for (s_, k_, a_), r in I.items():
        if a_ and k_ > 0 and not r.get('plan_failed'):
            th = jl(RES/'inner'/'plans'/f's{s_}_h{k_}_aware.json')['scenario']['hdv_time_gap']
            gaps += [g-th[str(h)] for g, h in zip(r['hdv_min_time_gap'], r['hdv'])]
    if gaps: T['in_aw_tg_margin'] = f2(min(gaps)); T['in_aw_tg_below'] = str(sum(g < 0 for g in gaps))
    stalled = [(s_, k_) for (s_, k_, a_), r in I.items() if a_ and k_ > 0 and not r['success'] and not r.get('plan_failed')]
    if stalled:
        s_, k_ = stalled[0]; z = np.load(RES/'inner'/'runs'/f's{s_}_h{k_}_aware'/'trajectory.npz')
        T['in_stall_drift'] = f2(-min(z['hdv'][-1, :, 0]-z['hdv'][0, :, 0])); T['in_stall_n'] = str(k_); T['in_stall_scen'] = str(s_+1)
    T['in_ag_ok_lane2'] = str(sum(1 for (s_, k_, a_), r in I.items() if not a_ and r['success']
                                   and all(h >= 16 for h in jl(RES/'inner'/'plans'/f's{s_}_h{k_}_agnostic.json')['scenario']['hdv'])))
    if aw:
        fwd, back, end, left = [], [], [], 0
        for (s_, k_, a_), r in I.items():
            if a_ and k_ > 0 and not r.get('plan_failed'):
                zz = np.load(RES/'inner'/'runs'/f's{s_}_h{k_}_aware'/'trajectory.npz'); Hh = zz['hdv']
                sc_ = jl(RES/'inner'/'plans'/f's{s_}_h{k_}_aware.json')['scenario']; lanes = [sc_['source'][h] for h in sc_['hdv']]
                for q, lane in enumerate(lanes):
                    dd = Hh[:, q, 0]-Hh[0, q, 0]
                    if lane < 2: fwd.append(dd.max()); back.append(-dd.min()); end.append(dd[-1])
                    else: left += 1
        T['in_hdv_fwd'] = f2(max(fwd)); T['in_hdv_back'] = f2(max(back))
        T['in_hdv_end_min'] = f2(min(end)); T['in_hdv_end_max'] = f2(max(end)); T['in_hdv_end_mean'] = f2(np.mean(end))
        T['in_hdv_n'] = str(len(end)); T['in_hdv_left'] = str(left)
        T['in_aw_mch'] = f2(min(r['min_center_margin_hdv'] for r in aw)); T['in_aw_mcc'] = f2(min(r['min_center_margin_cav'] for r in aw))
        T['in_aw_brake'] = f2(-min(min(r['hdv_min_acc']) for r in aw)); T['in_ag_brake'] = f2(-min(min(r['hdv_min_acc']) for r in ag))
        T['in_aw_tg'] = f2(min(min(r['hdv_min_time_gap']) for r in aw)); T['in_ag_tg'] = f2(min(min(r['hdv_min_time_gap']) for r in ag))
        ps = [r['plan_seconds'] for r in aw if r.get('plan_seconds')]
        T['in_plan_s'] = f2(np.mean(ps)); T['in_plan_smax'] = f2(max(ps))
        tf = [float(re.findall(r'[\d.]+', r['failure'])[0]) for r in ag if coll(r)]
        T['in_ag_tfail'] = f2(np.mean(tf)) if tf else '--'
    # robustness
    RA = jl(RES/'robust_analysis.json'); C = RA['cells']; RR = RA['runs']
    def cell(ctrl, tau, w, d, hat):
        return C.get(f'{ctrl}_tau{tau}_w{w}_d{d}_hat{hat}')
    c = cell('CE', .3, .02, 0, 0); T['rb_ce03_ok'] = f"{c['safe_completed']}/{c['n']}"; T['rb_ce03_mu'] = f2(c['mu_min'])
    c = cell('CE', .3, .02, .2, 0); T['rb_ce03d_mu'] = f2(c['mu_min'])
    ce = [r for r in RR if r['ctrl'] == 'CE' and r['lag'] == .6]
    T['rb_ce06_coll'] = str(sum(r['body_viol'] > 0 for r in ce)); T['rb_ce06_n'] = str(len(ce))
    la = [r for r in RR if r['ctrl'] == 'LA' and abs(r['hat']-r['lag']) < 1e-9 and r['w'] == .02]
    T['rb_la_n'] = str(len(la)); T['rb_la_safe'] = str(sum(r['viol'] == 0 for r in la)); T['rb_la_done'] = str(sum(r['t'] is not None and r['viol'] == 0 for r in la))
    T['rb_la_mu'] = f2(min(r['mu'] for r in la))
    for tau, k in [(.3, '03'), (.6, '06'), (.9, '09')]:
        c = cell('LA', tau, .02, .2, tau); T[f'rb_la{k}_t'] = f2(c['t_mean']) if c['t_mean'] else '--'
    T['rb_nom_t'] = f2(RA['nominal_time_s0_4'])
    c = cell('LA', .6, .05, .2, .6); T['rb_la005_done'] = str(c['safe_completed']); T['rb_la005_unsafe'] = str(c['n']-c['safe']); T['rb_la005_coll'] = str(c['collisions'])
    for hat, k in [(.3, '03'), (.45, '045'), (.9, '09')]:
        rr = [r for r in RR if r['ctrl'] == 'LA' and abs(r['lag']-.6) < 1e-9 and abs(r['w']-.02) < 1e-9 and abs(r['hat']-hat) < 1e-9]
        T[f'rb_mm{k}_viol'] = str(sum(r['viol'] > 0 for r in rr)); T[f'rb_mm{k}_coll'] = str(sum(r['body_viol'] > 0 for r in rr))
        T[f'rb_mm{k}_ok'] = str(sum(r['t'] is not None and r['viol'] == 0 for r in rr)); T[f'rb_mm{k}_n'] = str(len(rr))
    T['rb_tolp'] = f2(RA['tol'][0]); T['rb_tolv'] = f2(RA['tol'][1])
    T['rb_runs'] = str(len(RR))
    # communication
    for mode, k in [('bernoulli_th0.6', 'b'), ('ge_th0.6', 'g')]:
        M = A['comm'][mode]['methods']
        for met in ['stale_frac', 'rms_error', 'p95_error']:
            T[f'cm_{k}_{met.split("_")[0]}_red'] = f2(-pct(M['combined'][met], M['motion_speed'][met]))
        T[f'cm_{k}_viol'] = str(sum(M[m]['violations'] for m in M))
    T['cm_runs'] = str(sum(A['comm'][m]['runs'] for m in A['comm']))
    # computation
    Sc = A['scale']
    for n in ['12', '24', '36']:
        T[f'sc_{n}_qp'] = f2(Sc[n]['qp_median_ms']['mean']); T[f'sc_{n}_cyc'] = f2(Sc[n]['wall_per_step'])
    T['sc_runs'] = str(sum(Sc[n]['n'] for n in Sc))
    # experiment-group sizes and closed-loop execution of different plans
    T['ex_runs'] = str(sum(eg[n]['cases'] for n in eg)+sum(A['exact_main'][m]['n'] for m in A['exact_main']))
    T['js_runs'] = str(sum(1 for g in ('jsweep', 'exec_fixed') for _ in (RES/g).glob('*/summary.json')))
    EM = A['exec_methods']
    for m, k in [('source_blocks', 'sb'), ('mda', 'mda'), ('motion_speed', 'ms'), ('combined', 'c')]:
        T[f'em_{k}_t'] = f2(EM[m]['summary']['time']['mean']); T[f'em_{k}_eu'] = f2(EM[m]['summary']['effort']['mean'])
    T['em_cms_t'] = f2(abs(pct(EM['combined']['summary']['time']['mean'], EM['motion_speed']['summary']['time']['mean'])))
    T['em_cms_eu'] = f2(abs(pct(EM['combined']['summary']['effort']['mean'], EM['motion_speed']['summary']['effort']['mean'])))
    T['em_sb_t_inc'] = f2(pct(EM['source_blocks']['summary']['time']['mean'], EM['combined']['summary']['time']['mean']))
    T['em_sb_eu_inc'] = f2(pct(EM['source_blocks']['summary']['effort']['mean'], EM['combined']['summary']['effort']['mean']))
    import platoon_comm as pc
    pc.TH = .6; Ub = np.random.default_rng(5000).random((10, 2, 90, 11))[0, 0]
    for m, k in [('motion_speed', 'ms'), ('combined', 'c')]:
        pl = jl(RES/'planning/plans'/f's0_{m}.json'); q = np.array(pl['scenario']['quality']); ch = pl['assignment']['target_orders'][0]
        quals = np.array([q[a, b] for a, b in zip(ch[:-1], ch[1:])])
        T[f'cm_brake_{k}'] = f2(np.abs(pc.simulate(quals, Ub, mode='ge')['errs']).max())
    traffic_and_comparison(T, A)
    lines = ['\\makeatletter']+[f'\\@namedef{{res@{k}}}{{{v}}}' for k, v in sorted(T.items())]+['\\makeatother']
    for lang in ('zh', 'en'):
        (ROOT/'manuscript'/lang).mkdir(parents=True, exist_ok=True)
        (ROOT/'manuscript'/lang/'results.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    for k, v in sorted(T.items()): print(k, v)


def paired(a, b):
    """Mean difference a-b with its t-based 95% half width and the two-sided Wilcoxon p-value."""
    from scipy import stats
    d = np.asarray(a, float)-np.asarray(b, float)
    hw = stats.t.ppf(.975, len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
    p = stats.wilcoxon(a, b).pvalue if np.any(d != 0) else 1.
    return d.mean(), hw, p


def traffic_and_comparison(T, A):
    from driver_model import sample_driver, equilibrium_gap
    rng = np.random.default_rng(0); L = 4.8
    d = np.array([L+equilibrium_gap(16., sample_driver(rng)) for _ in range(20000)])
    T['idm_d50'] = f2(np.percentile(d, 50)); T['idm_d99'] = f2(np.percentile(d, 99))
    # upstream traffic
    D = {f.stem: jl(f) for f in (RES/'traffic').glob('*.json')}
    tot = lambda k: D[k]['delay'] if k in D else None
    T['tr_runs'] = str(sum(1 for k in D if not k.endswith('_steady')))
    for k in ('barrier', 'eager', 'settled', 'rear', 'cai'):
        v = [tot(f's{s_}_{k}') for s_ in range(20) if tot(f's{s_}_{k}') is not None]
        if v: T[f'tr_{k}_d'] = f2(np.mean(v)); T[f'tr_{k}_n'] = str(len(v))
    for k, m in (('sb', 'source_blocks'), ('mda', 'mda'), ('ms', 'motion_speed'), ('c', 'settled')):
        v = [tot(f's{s_}_{m}') for s_ in range(6) if tot(f's{s_}_{m}') is not None]
        if v: T[f'tr_m{k}_d'] = f2(np.mean(v))
    pairs = lambda a, b: [(tot(f's{s_}_{a}'), tot(f's{s_}_{b}')) for s_ in range(20) if tot(f's{s_}_{a}') is not None and tot(f's{s_}_{b}') is not None]
    for a, b, key in (('settled', 'barrier', 'sb'), ('rear', 'settled', 'rs'), ('eager', 'settled', 'es')):
        pr = pairs(a, b)
        if len(pr) > 2:
            x, y = map(np.array, zip(*pr)); m, hw, pv = paired(x, y)
            T[f'tr_{key}_red'] = f2(-100*m/y.mean()); T[f'tr_{key}_p'] = sci(pv); T[f'tr_{key}_n'] = str(len(pr))
    rows = [D[k] for k in D if k[:1] == 's' and k.split('_', 1)[1] in ('barrier', 'eager', 'settled', 'rear')]
    if rows:
        T['tr_drop_max'] = f2(max(r['max_drop'] for r in rows)); T['tr_hard'] = str(sum(r['hard_braking'] for r in rows))
        T['tr_tg'] = f2(min(r['min_time_gap'] for r in rows)); T['tr_coll'] = str(sum(r['collision'] for r in rows))
        amp = [l['drop_last']/l['drop_first'] for r in rows for l in r['lanes'] if l['drop_first'] > .5]
        T['tr_amp_max'] = f2(max(amp)) if amp else '--'
    st = [D[f's{s_}_settled'] for s_ in range(20) if f's{s_}_settled' in D]
    if st:
        share = np.array([[l['delay'] for l in r['lanes']] for r in st]).sum(0); share = 100*share/share.sum()
        for lane in range(3): T[f'tr_share{lane}'] = f2(share[lane])
        T['tr_lane'] = str(int(np.argmax([l['delay'] for l in D['s0_settled']['lanes']]))) if 's0_settled' in D else '0'
        T['tr_drop_first'] = f2(np.mean([max(l['drop_first'] for l in r['lanes']) for r in st]))
        back = []
        for s_ in range(20):
            X = np.load(RES/'exec_adaptive'/f's{s_}_settled'/'trajectory.npz')['trajectory'][:, :, 0]
            back.append(X[0].min()-X.min(1).min())
        T['tr_back'] = f2(np.mean(back))
    # closed-loop cost of the rear-anchored layout, paired with the centred layout
    rr = [(jl(RES/'exec_rear'/f's{s_}_settled'/'summary.json'), jl(RES/'exec_adaptive'/f's{s_}_settled'/'summary.json'))
          for s_ in range(20) if (RES/'exec_rear'/f's{s_}_settled'/'summary.json').exists()]
    if rr:
        T['rear_n'] = str(len(rr)); T['rear_ok'] = str(sum(a['success'] for a, _ in rr))
        ok = [(a, b) for a, b in rr if a['success'] and b['success']]
        for key, f in (('t', 'time'), ('eu', 'acceleration_squared_integral')):
            x = np.array([a[f] for a, _ in ok]); y = np.array([b[f] for _, b in ok])
            T[f'rear_{key}'] = f2(x.mean()); T[f'rear_{key}_c'] = f2(y.mean()); T[f'rear_{key}_inc'] = f2(100*(x.mean()-y.mean())/y.mean())
        T['rear_q'] = f2(np.mean([a['assignment']['mean_link_quality'] for a, _ in rr]))
        import analysis as AN
        au = AN.load_group('exec_rear')
        T['rear_mcc'] = f2(min(r['min_center_margin'] for r in au.values())); T['rear_viol'] = str(sum(r['safety_violations'] > 0 for r in au.values()))
    # comparison with TAPPA
    eg = A['exact_gap']; runs = 0; rows_tex = {'zh': [], 'en': []}
    for n in (12, 18, 24):
        seeds = (0, 1) if n < 24 else range(20)
        ours = [jl((RES/'scale'/f'n{n}_s{s_}' if n < 24 else RES/'exec_adaptive'/f's{s_}_settled')/'summary.json') for s_ in seeds]
        cai = [jl(RES/'cai'/'runs'/f'n{n}_s{s_}'/'summary.json') for s_ in seeds if (RES/'cai'/'runs'/f'n{n}_s{s_}'/'summary.json').exists()]
        runs += len(cai); ok = [c for c in cai if c['success']]
        up_o = [tot(f'n{n}_s{s_}_dp') if n < 24 else tot(f's{s_}_settled') for s_ in seeds]
        up_c = [tot(f'n{n}_s{s_}_cai') if n < 24 else tot(f's{s_}_cai') for s_ in seeds]
        up_o = [u for u in up_o if u is not None]; up_c = [u for u in up_c if u is not None]
        T[f'ca_n{n}_ok'] = str(len(ok)); T[f'ca_n{n}_n'] = str(len(cai))
        if up_o: T[f'tr_n{n}_dp'] = f2(np.mean(up_o))
        if up_c: T[f'tr_n{n}_cai'] = f2(np.mean(up_c))
        T[f'ca_n{n}_T_dp'] = f2(np.mean([o['time'] for o in ours])); T[f'ca_n{n}_eu_dp'] = f2(np.mean([o['acceleration_squared_integral'] for o in ours]))
        T[f'ca_n{n}_q_dp'] = f2(np.mean([o['assignment']['mean_link_quality'] for o in ours])); T[f'ca_n{n}_plan_dp'] = f2(eg[str(n)]['exact_s'])
        cells_o = [str(n), '{ours}', f'{len(ours)}/{len(ours)}', T[f'ca_n{n}_T_dp'], T[f'ca_n{n}_eu_dp'], T[f'ca_n{n}_q_dp'], T[f'ca_n{n}_plan_dp'],
                   f2(np.mean(up_o)) if up_o else '--']
        if ok:
            T[f'ca_n{n}_T'] = f2(np.mean([c['completion'] for c in ok])); T[f'ca_n{n}_eu'] = f2(np.mean([c['acceleration_squared_integral'] for c in ok]))
            T[f'ca_n{n}_q'] = f2(np.mean([c['assignment']['mean_link_quality'] for c in ok]))
            T[f'ca_n{n}_tf'] = f2(np.mean([c['T_F'] for c in ok])); T[f'ca_n{n}_steps'] = str(int(np.mean([c['steps'] for c in ok])))
            T[f'ca_n{n}_jx'] = f2(max(c['max_jx'] for c in ok)); T[f'ca_n{n}_jy'] = f2(max(c['max_jy'] for c in ok))
            T[f'ca_n{n}_ax'] = f2(max(c['max_ax'] for c in ok)); T[f'ca_n{n}_mcc'] = f2(min(c['min_center_margin'] for c in ok))
        plan_c = [c['planner']['seconds'] for c in cai if 'planner' in c]
        lim = any(c['planner']['time_limited'] or not c['success'] for c in cai if 'planner' in c)
        T[f'ca_n{n}_plan'] = f2(np.mean(plan_c)) if plan_c else '--'
        cells_c = [str(n), 'TAPPA', f'{len(ok)}/{len(cai)}', T.get(f'ca_n{n}_T', '--'), T.get(f'ca_n{n}_eu', '--'), T.get(f'ca_n{n}_q', '--'),
                   (r'$\geq$600' if lim else T[f'ca_n{n}_plan']), f2(np.mean(up_c)) if up_c else '--']
        for lang, name in (('zh', '本文'), ('en', 'Proposed')):
            rows_tex[lang] += [' & '.join(cells_o).replace('{ours}', name)+r' \\', ' & '.join(cells_c)+r' \\']+([r'\midrule'] if n < 24 else [])
        nodes = [c['planner']['cbs_nodes'] for c in cai if 'planner' in c and not c['success']]
        if nodes: T[f'ca_n{n}_nodes'] = f'{np.mean(nodes)/1e6:.2f}'
    T['ca_runs'] = str(runs)
    for lang in ('zh', 'en'):
        (ROOT/'manuscript'/lang).mkdir(parents=True, exist_ok=True)
        (ROOT/'manuscript'/lang/'tab_cai_rows.tex').write_text('\n'.join(rows_tex[lang])+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
