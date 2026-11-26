"""Mixed-traffic group (EX10): reconfiguration behind human-driven vehicles."""
import json, argparse

from run_mixed import run_mixed, make_showcase_plan, ROOT
from batch_exec import plan_file

OUT = ROOT/'results'/'mixed'


def jobs():
    J = [dict(name='showcase', plan='showcase', seed=0, hdv=dict(onset=28., resume=62.), t_start=10., t_end=160.),
         dict(name='showcase_steady', plan='showcase', seed=0, hdv=dict(drop=0.), t_start=10., t_end=160.)]
    for seed in range(5):
        base = dict(plan='standard', seed=seed, t_start=0., t_end=0.)
        J.append(dict(base, name=f's{seed}_steady', hdv=dict(drop=0., sigma=.1)))
        for onset in [15., 30., 50.]:
            J.append(dict(base, name=f's{seed}_b1_t{int(onset)}', hdv=dict(onset=onset, resume=onset+30., drop=3., decel=1., sigma=.1)))
        J.append(dict(base, name=f's{seed}_b2_t30', hdv=dict(onset=30., resume=60., drop=6., decel=2., jerk=2., sigma=.1)))
        J.append(dict(base, name=f's{seed}_noadapt_t30', adapt=False, limit=200.,
                      hdv=dict(onset=30., resume=60., drop=3., decel=1., sigma=.1)))
    return J


def execute(j):
    out = OUT/'batch'/j['name']
    if (out/'summary.json').exists(): return 'skip '+j['name']
    if j['plan'] == 'showcase':
        pf = OUT/'plans'/f"showcase_s{j['seed']}.json"; pf.parent.mkdir(parents=True, exist_ok=True)
        if not pf.exists(): pf.write_text(json.dumps(make_showcase_plan(j['seed'])), encoding='utf-8')
        pl = json.loads(pf.read_text(encoding='utf-8'))
    else:
        pl = plan_file(j['seed'], 'combined')
    s = run_mixed(pl, out, hdv=dict(j['hdv'], seed=100+j['seed']), adapt=j.get('adapt', True), t_start=j['t_start'],
                  t_end=j['t_end'], limit=j.get('limit', 260.))
    s2 = json.loads((out/'summary.json').read_text(encoding='utf-8')); s2['job'] = j
    (out/'summary.json').write_text(json.dumps(s2, ensure_ascii=False, indent=1), encoding='utf-8')
    return f"{j['name']} ok={s['success']} T={s['completion']} mcc={s['min_center_margin_cav']:.3f} gap={s['min_gap_hdv']:.2f} wall={s['wall_seconds']:.0f}"


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--worker', type=int, default=0); p.add_argument('--workers', type=int, default=1)
    a = p.parse_args()
    for j in jobs()[a.worker::a.workers]:
        print(execute(j), flush=True)
