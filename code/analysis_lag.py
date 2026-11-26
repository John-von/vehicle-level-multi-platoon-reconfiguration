"""Robustness study under actuator lag and disturbances: certainty-equivalent
(CE, group robust) versus lag-aware (LA, groups lagaware and lagmismatch).

Under persistent disturbances the strict tolerance of the nominal runs
(0.08 m, 0.025 m/s) is not meaningful, because the disturbance alone moves the
vehicles by more than that.  A run therefore counts as completed at the first
instant at which every vehicle has finished its compaction reference and lies
within TOL_P of its final slot with relative speed below TOL_V.  This post-hoc
rule is applied identically to every run of the three groups.  Safety is the
independent continuous audit of the whole recorded trajectory.
Writes results/robust_analysis.json."""
from pathlib import Path
import json
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; RES = ROOT/'results'

import analysis as AN
TOL_P, TOL_V = .2, .06


def completion(folder, s):
    z = np.load(folder/'trajectory.npz'); x = z['trajectory']; fg = z['final_goal']; h = s['config']['dt']
    ks = np.array(s['compact_start'], float); Tk = np.array(s['T_compact'], float)
    t = np.arange(len(x))*h
    within = (np.max(np.abs(x[:, :, :2]-fg[None]), axis=2) < TOL_P) & (np.max(np.abs(x[:, :, 2:]), axis=2) < TOL_V)
    elapsed = np.isfinite(ks)[None, :] & (t[:, None]-np.nan_to_num(ks)[None, :] >= np.nan_to_num(Tk)[None, :]-1e-9)
    ok = np.flatnonzero(np.all(within & elapsed, axis=1))
    return float(t[ok[0]]) if len(ok) else None


def records():
    out = []
    for group, ctrl in [('robust', 'CE'), ('lagaware', 'LA'), ('lagmismatch', 'LA')]:
        if not (RES/group).exists(): continue
        for f in sorted((RES/group).glob('*/summary.json')):
            s = AN.jl(f); a = AN.audit_run(f.parent, group)
            out.append(dict(group=group, ctrl=ctrl, name=f.parent.name, seed=s['seed'], lag=s['lag'], w=s['noise'], delta=s['margin'],
                            hat=(s.get('lag_model') or 0.) if ctrl == 'LA' else 0., t=completion(f.parent, s),
                            mu=a['minima']['safety']['value'], beta=a['minima']['body']['value'], viol=int(a['violations']['safety']),
                            body_viol=int(a['violations']['body']), rec=int(s.get('recovery_attempts', 0)), fb=int(s['qp_fallbacks']),
                            effort=s['acceleration_squared_integral'], cmd_ok=bool(a.get('command_limits_ok', True))))
    return out


def cell(rr):
    done = [r for r in rr if r['t'] is not None]
    return dict(n=len(rr), completed=len(done), safe=sum(r['viol'] == 0 for r in rr), collisions=sum(r['body_viol'] > 0 for r in rr),
                safe_completed=sum(r['t'] is not None and r['viol'] == 0 for r in rr),
                t_mean=float(np.mean([r['t'] for r in done])) if done else None, t_max=float(max(r['t'] for r in done)) if done else None,
                mu_min=float(min(r['mu'] for r in rr)), beta_min=float(min(r['beta'] for r in rr)),
                viol=int(sum(r['viol'] for r in rr)), rec=float(np.mean([r['rec'] for r in rr])), seeds=sorted(r['seed'] for r in rr))


def main():
    R = records(); C = {}
    keys = sorted(set((r['ctrl'], r['lag'], r['w'], r['delta'], r['hat']) for r in R))
    for k in keys:
        rr = [r for r in R if (r['ctrl'], r['lag'], r['w'], r['delta'], r['hat']) == k]
        C['%s_tau%g_w%g_d%g_hat%g' % k] = cell(rr)
    nom = [AN.jl(RES/'exec_adaptive'/f's{s}_settled/summary.json')['time'] for s in range(5)]
    out = dict(tol=[TOL_P, TOL_V], nominal_time_s0_4=float(np.mean(nom)), cells=C, runs=R)
    (RES/'robust_analysis.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    for k, v in C.items(): print(k, {kk: v[kk] for kk in ['n', 'completed', 'safe', 'collisions', 't_mean', 'mu_min', 'beta_min']})
    return out


if __name__ == '__main__':
    main()
