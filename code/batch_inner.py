"""Group EX11: human-driven vehicles inside the source platoons."""
import json, argparse

from scenario import Config, jsonable
from plan_mixed import mixed_scenario, plan_hdv
from plan_methods import plan as make_plan
from run_inner import run_inner, ROOT

OUT = ROOT/'results'/'inner'


def jobs():
    base = [dict(seed=s, n_hdv=0, aware=True) for s in range(5)]
    return [dict(seed=s, n_hdv=k, aware=a) for k in (1, 2, 3) for s in range(5) for a in (True, False)]+base


def get_plan(seed, n_hdv, aware, cfg=Config()):
    f = OUT/'plans'/f"s{seed}_h{n_hdv}_{'aware' if aware else 'agnostic'}.json"; f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists(): return json.loads(f.read_text(encoding='utf-8'))
    sc = mixed_scenario(seed, n_hdv, cfg)
    if aware and n_hdv > 0:
        r = plan_hdv(sc, cfg)
    else:
        r = make_plan({k: v for k, v in sc.items() if k not in ('hdv', 'hdv_time_gap', 'hdv_driver')}, 'combined', cfg)
        r['target_orders_full'] = r['target_orders']
    pl = dict(scenario=jsonable(sc), assignment=jsonable(r)); f.write_text(json.dumps(pl), encoding='utf-8')
    return pl


def execute(j):
    name = f"s{j['seed']}_h{j['n_hdv']}_{'aware' if j['aware'] else 'agnostic'}"; out = OUT/'runs'/name
    if (out/'summary.json').exists(): return 'skip '+name
    try:
        pl = get_plan(j['seed'], j['n_hdv'], j['aware'])
    except ValueError:
        out.mkdir(parents=True, exist_ok=True)
        (out/'summary.json').write_text(json.dumps(dict(seed=j['seed'], n_hdv=j['n_hdv'], aware=j['aware'], success=False,
                                                        failure='no feasible plan', plan_failed=True)), encoding='utf-8')
        return name+' plan infeasible'
    s = run_inner(pl, out, aware=j['aware'], seed=j['seed'])
    s2 = json.loads((out/'summary.json').read_text(encoding='utf-8')); s2.update(n_hdv=j['n_hdv'], plan_seconds=pl['assignment'].get('seconds'))
    (out/'summary.json').write_text(json.dumps(s2, ensure_ascii=False, indent=1), encoding='utf-8')
    mch = s['min_center_margin_hdv']; hacc = min(s['hdv_min_acc']) if s['hdv_min_acc'] else 0.
    return f"{name} ok={s['success']} T={s['completion']} mcc={s['min_center_margin_cav']:.3f} mch={mch if mch is None else round(mch, 2)} hacc={hacc:.2f} {s['failure']}"


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--worker', type=int, default=0); p.add_argument('--workers', type=int, default=1)
    p.add_argument('--queue', action='store_true')
    a = p.parse_args()
    if a.queue:
        lock = OUT/'locks'; lock.mkdir(parents=True, exist_ok=True)
        for j in sorted(jobs(), key=lambda j: (not j['aware'], j['n_hdv'] == 0, -j['n_hdv'], j['seed'])):
            name = f"s{j['seed']}_h{j['n_hdv']}_{'aware' if j['aware'] else 'agnostic'}"
            try:
                (lock/name).mkdir()
            except FileExistsError:
                continue
            print(execute(j), flush=True)
    else:
        for j in jobs()[a.worker::a.workers]: print(execute(j), flush=True)
