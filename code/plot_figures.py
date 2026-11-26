"""Figures of the manuscript, in Chinese (manuscript/zh/figs) or English (manuscript/en/figs).

Usage: python plot_figures.py [zh|en] [figure names].  Text is typeset by XeLaTeX through
the PGF backend with the fonts of the manuscript, and every value is read from results/."""
from pathlib import Path
import sys, json, glob, collections
import numpy as np
import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors

ROOT = Path(__file__).resolve().parent.parent
ARGS = sys.argv[1:]
LANG = ARGS.pop(0) if ARGS and ARGS[0] in ('zh', 'en') else 'zh'
RES = ROOT/'results'; OUT = ROOT/'manuscript'/LANG/'figs'; OUT.mkdir(parents=True, exist_ok=True)

EN = {'统一阶段': 'Common-phase',
      '逐车即时': 'Immediate',
      '逐车稳定': 'Settled',
      '运动代价': 'Motion cost',
      '平均链路质量': 'Mean link quality',
      '最差链路质量': 'Worst link quality',
      '每周期期望丢包数': 'Expected losses per cycle',
      '横向位置 $y_i$ (m)': 'Lateral position $y_i$ (m)',
      '相对纵向位置 $x_i$ (m)': 'Longitudinal position $x_i$ (m)',
      '速度 (m/s)': 'Speed (m/s)',
      '纵向加速度 (m/s$^2$)': 'Longitudinal acceleration (m/s$^2$)',
      '横向加速度 $\\tilde u_{y,i}$ (m/s$^2$)': 'Lateral acceleration $\\tilde u_{y,i}$ (m/s$^2$)',
      '最小间距裕度 (m)': 'Minimum separation margin (m)',
      '时间 (s)': 'Time (s)',
      '源队列 A': 'Source platoon A',
      '源队列 B': 'Source platoon B',
      '源队列 C': 'Source platoon C',
      '全部网联车辆的范围': 'Range over all CAVs',
      '群体参考 $v_0$、$a_0$': 'Group reference $v_0$, $a_0$',
      '人工驾驶车辆': 'HDV',
      '网联车辆之间': 'Between CAVs',
      '重组开始与完成': 'Start and completion',
      '前方人工驾驶车辆变速': 'Speed change of the HDV ahead',
      '排位': 'Staging',
      '换道': 'Lane change',
      '压缩': 'Compaction',
      '完成时间 (s)': 'Completion time (s)',
      '控制能量 $E_u$ (m$^2$/s$^3$)': 'Control energy $E_u$ (m$^2$/s$^3$)',
      '输入变化量 $E_j$ (m$^2$/s$^5$)': 'Input variation $E_j$ (m$^2$/s$^5$)',
      '预留': 'Reserved',
      '单间隙': 'Single gap',
      '完成的场景数': 'Completed scenarios',
      '均匀\n时长': 'Uniform\nduration',
      '峰值纵向加速度 (m/s$^2$)': 'Peak longitudinal\nacceleration (m/s$^2$)',
      '参考时长设计（$j_{\\rm ref}$, m/s$^3$）': 'Reference duration design ($j_{\\rm ref}$, m/s$^3$)',
      '匀速': 'Steady',
      '排位中': 'In staging',
      '换道中': 'In lane change',
      '压缩中': 'In compaction',
      '强制动': 'Hard braking',
      '不调速': 'No adaptation',
      '全部\n相撞': 'All\ncollide',
      '分离阈值 $D_x$': 'Separation threshold $D_x$',
      '最小纵向距离 (m)': 'Minimum longitudinal\ndistance (m)',
      '调整群体速度': 'With adaptation',
      '不调整群体速度': 'Without adaptation',
      '同车道纵向距离 (m)': 'Same-lane longitudinal\ndistance (m)',
      '考虑': 'Aware',
      '忽略': 'Agnostic',
      '运行次数': 'Number of runs',
      '完成': 'Completed',
      '未完成': 'Not completed',
      '违反分离规则': 'Separation violated',
      '车体相交': 'Body overlap',
      '无可行计划': 'No feasible plan',
      '人工驾驶车辆数': 'Number of HDVs',
      '最大减速度 (m/s$^2$)': 'Maximum\ndeceleration (m/s$^2$)',
      '同车道最小纵向距离 (m)': 'Minimum same-lane\nlongitudinal distance (m)',
      '后端基准': 'Rear-anchored',
      '上游总延误 (车$\\cdot$s)': 'Total upstream delay (veh$\\cdot$s)',
      '车道 0': 'Lane 0',
      '车道 1': 'Lane 1',
      '车道 2': 'Lane 2',
      '群体坐标系中的位置 (m)': 'Position in the group frame (m)',
      '来车速度 (m/s)': 'Speed of upstream vehicles (m/s)',
      '占用该车道的重组车辆的后端': 'Rear end of the reconfiguring vehicles in the lane',
      '碰撞': 'coll.',
      '完成且无违规': 'Completed, no violation',
      '未完成、无违规': 'Not completed, no violation',
      '假定滞后 $\\hat\\tau$ (s)，$\\tau=0.6$ s': 'Assumed lag $\\hat\\tau$ (s), $\\tau=0.6$ s',
      '过期计划比例': 'Fraction of stale plans',
      '间距误差均方根 (m)': 'RMS spacing error (m)',
      '间距误差 95\\% 分位数 (m)': '95th-percentile spacing error (m)',
      '伯努利丢包': 'Bernoulli loss',
      'Gilbert--Elliott 丢包': 'Gilbert--Elliott loss',
      '头车减速 4 m/s': 'Leader slows by 4 m/s',
      '最大间距误差 (m)': 'Maximum spacing error (m)',
      '中位数': 'Median',
      '95\\% 分位数': '95th percentile',
      '车辆数 $N$': 'Number of vehicles $N$',
      '单次求解时间 (ms)': 'Solve time per QP (ms)',
      '采样周期 $h$': 'Sampling period $h$',
      '单周期顺序计算时间 (s)': 'Sequential time per cycle (s)',
      '停滞': 'stalled',
      '，平均 ': ', mean '}


def L(s): return s if LANG == 'zh' else EN[s]


COL, TXT = 4.6, 5.4     # figure widths (in) for the 390 pt text block
PRE = (r'\usepackage{fontspec}\setmainfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,'
       r'ItalicFont=*-italic,BoldItalicFont=*-bolditalic]\usepackage{xeCJK}'
       r'\IfFontExistsTF{NotoSerifCJK-Regular.ttc}{\setCJKmainfont{NotoSerifCJK-Regular.ttc}[FontIndex=2,Script=CJK,'
       r'Language=Chinese Simplified,BoldFont=NotoSerifCJK-Bold.ttc,BoldFeatures={FontIndex=2}]}{\setCJKmainfont{SimSun}}'
       r'\usepackage{amsmath,bm}')
plt.rcParams.update({
    'pgf.texsystem': 'xelatex', 'pgf.rcfonts': False, 'pgf.preamble': PRE, 'font.family': 'serif',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.linewidth': .55, 'lines.linewidth': .9, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'xtick.major.size': 2.4, 'ytick.major.size': 2.4, 'xtick.minor.size': 1.3,
    'ytick.minor.size': 1.3, 'xtick.major.width': .5, 'ytick.major.width': .5, 'legend.frameon': False,
    'legend.handlelength': 1.6, 'axes.grid': True, 'grid.color': '#E3E7EC', 'grid.linewidth': .45, 'axes.axisbelow': True,
    'axes.edgecolor': '#30343B', 'axes.labelcolor': '#30343B', 'text.color': '#30343B', 'xtick.color': '#30343B',
    'ytick.color': '#30343B', 'hatch.linewidth': .5})
SRC = ['#4477AA', '#228833', '#EE7733']
POL = {'barrier': '#888888', 'eager': '#CC3311', 'settled': '#0077BB'}
POLH = {'barrier': '', 'eager': '//////', 'settled': ''}
POLN = {'barrier': L('统一阶段'), 'eager': L('逐车即时'), 'settled': L('逐车稳定')}
POLS = {'barrier': '-', 'eager': '--', 'settled': '-'}
MET = ['source_blocks', 'mda', 'motion_only', 'motion_speed', 'combined']
METN = {'source_blocks': 'SB', 'mda': 'MDA', 'motion_only': 'DP-M', 'motion_speed': 'DP-MS', 'combined': 'DP-MSL'}
METC = {'source_blocks': '#BBBBBB', 'mda': '#CCBB44', 'motion_only': '#66CCEE', 'motion_speed': '#AA3377', 'combined': '#004488'}
HDVC = '#30343B'
OUTC = {'ok': '#77AADD', 'stall': '#D9DDE3', 'viol': '#DDAA33', 'coll': '#CC3311'}


def jl(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def tag(ax, s, x=.025, y=.965, ha='left'):
    ax.text(x, y, r'\textbf{'+s+'}', transform=ax.transAxes, va='top', ha=ha, fontsize=8, zorder=20,
            bbox=dict(boxstyle='square,pad=0.05', fc='white', ec='none', alpha=.85))


def save(fig, name):
    fig.savefig(OUT/f'{name}.pdf', bbox_inches='tight', pad_inches=.015, dpi=600); plt.close(fig)


def ci95(v):
    v = np.asarray(v, float); return 1.96*v.std(ddof=1)/np.sqrt(len(v)) if len(v) > 1 else 0.


def bars(ax, xs, means, errs, colors, hatches, fmt='{:.2f}', dy=None, labels=True, width=.62):
    for x, m, e, c, h in zip(xs, means, errs, colors, hatches):
        ax.bar(x, m, width, color=mcolors.to_rgba(c, .78), edgecolor=c, lw=.6, hatch=h, zorder=2)
        ax.errorbar(x, m, yerr=e, color='#30343B', lw=.6, capsize=1.8, capthick=.6, zorder=3)
        if labels: ax.text(x, m-e-.025*ax.get_ylim()[1], fmt.format(m), ha='center', va='top', fontsize=6.6, color='white', zorder=4)


# ------------------------------------------------------------------ planning statistics
def fig_plan_stats():
    P = jl(RES/'planning/planning_main_0-99.json')
    by = collections.defaultdict(list)
    for r in P: by[r['method']].append(r)
    fig, axes = plt.subplots(2, 2, figsize=(COL, 3.25), layout='constrained')
    rng = np.random.default_rng(3)
    for ax, k, lab, letter in zip(axes.flat, ['motion', 'mean_q', 'min_q', 'expected_losses'],
                                  [L('运动代价'), L('平均链路质量'), L('最差链路质量'), L('每周期期望丢包数')], 'abcd'):
        data = [[r[k] for r in by[m]] for m in MET]
        for x, d in enumerate(data):
            ax.scatter(x+rng.uniform(-.16, .16, len(d)), d, s=1.6, color='#777777', alpha=.35, lw=0, zorder=1)
        b = ax.boxplot(data, positions=list(range(5)), widths=.56, patch_artist=True, showfliers=False,
                       medianprops=dict(color='black', lw=.9), whiskerprops=dict(lw=.6), capprops=dict(lw=.6), boxprops=dict(lw=.6))
        for patch, m in zip(b['boxes'], MET):
            patch.set_facecolor(mcolors.to_rgba(METC[m], .45)); patch.set_edgecolor(METC[m])
        ax.set_xticks(range(5)); ax.set_xticklabels([METN[m].replace('DP-', 'DP-\n') for m in MET], fontsize=6.2, linespacing=.9)
        ax.set_ylabel(lab, fontsize=7.6); tag(ax, f'({letter})', x=.965 if k == 'motion' else .035, ha='right' if k == 'motion' else 'left')
        if k == 'motion': ax.set_yscale('log'); ax.set_yticks([20, 30, 40, 60]); ax.set_yticklabels(['20', '30', '40', '60']); ax.minorticks_off()
    save(fig, 'fig_plan_stats')


# ------------------------------------------------------------------ showcase run with human-driven vehicles
def change_spans(acc, h, thr=.02):
    on = np.abs(acc) > thr; spans = []; k = 0
    while k < len(on):
        if on[k]:
            j = k
            while j < len(on) and on[j]: j += 1
            spans.append((k*h, j*h)); k = j
        else: k += 1
    merged = []
    for a, b in spans:
        if merged and a-merged[-1][1] < 2.: merged[-1] = (merged[-1][0], b)
        else: merged.append((a, b))
    return merged


def fig_main():
    f = RES/'mixed/batch/showcase'; z = np.load(f/'trajectory.npz'); s = jl(f/'summary.json'); sc = jl(f/'scenario.json')
    src = np.array(sc['source']); X = z['trajectory']; U = z['commands']; Hh = z['hdv']; fr = z['frame']; ha = z['hdv_acc']
    h = sc['config']['dt']; t = np.arange(len(X))*h; tu = np.arange(len(U))*h; q = s['hdv']['brake_lane']
    v0 = fr[:, 1]; a0 = fr[:-1, 2]
    spans = change_spans(ha[:, q], h)
    fig, axes = plt.subplots(2, 3, figsize=(TXT, 3.75), layout='constrained')
    a = axes.flat
    for ax in a:
        for lo, hi in spans: ax.axvspan(lo, hi, color='#EE7733', alpha=.10, lw=0, zorder=0)
        for tv in (s['t_start'], s['t_done']): ax.axvline(tv, color='#7A808A', lw=.6, ls=(0, (3, 2)), zorder=1)
    for i in range(len(src)):
        a[0].plot(t, X[:, i, 1], color=SRC[src[i]], lw=.7)
        a[1].plot(t, X[:, i, 0], color=SRC[src[i]], lw=.7)
    v = X[:, :, 2]+v0[:, None]
    a[2].fill_between(t, v.min(1), v.max(1), color='#4477AA', alpha=.28, lw=0)
    a[2].plot(t, v0, color=HDVC, lw=.9)
    a[2].plot(t, v0+Hh[:, q, 2], color=HDVC, lw=.9, ls=(0, (4, 1.6)))
    ay = U[:, :, 1]
    a[4].fill_between(tu, ay.min(1), ay.max(1), color='#4477AA', alpha=.28, lw=0, step='post')
    ax_tot = U[:, :, 0]+a0[:, None]
    a[3].fill_between(tu, ax_tot.min(1), ax_tot.max(1), color='#4477AA', alpha=.28, lw=0, step='post')
    a[3].step(tu, a0, where='post', color=HDVC, lw=.9)
    a[5].semilogy(tu, np.maximum(z['margin_cav'], 1e-3), color='#0077BB', lw=.9)
    a[5].semilogy(tu, np.maximum(z['margin_hdv'], 1e-3), color=HDVC, lw=.9, ls=(0, (4, 1.6)))
    a[0].set_ylabel(L('横向位置 $y_i$ (m)')); a[0].set_yticks([0, 3.6, 7.2]); a[0].set_ylim(-.9, 8.1)
    a[1].set_ylabel(L('相对纵向位置 $x_i$ (m)')); a[2].set_ylabel(L('速度 (m/s)')); a[2].set_ylim(np.floor(v.min()-.2), np.ceil(v.max()+.2))
    a[3].set_ylabel(L('纵向加速度 (m/s$^2$)')); a[4].set_ylabel(L('横向加速度 $\\tilde u_{y,i}$ (m/s$^2$)'))
    a[5].set_ylabel(L('最小间距裕度 (m)')); a[5].set_ylim(.01, 2000)
    lim = 1.1*max(abs(ax_tot).max(), .1); a[3].set_ylim(-lim, lim)
    lim = 1.25*abs(U[:, :, 1]).max(); a[4].set_ylim(-lim, lim)
    for ax, l in zip(a, 'abcdef'):
        if LANG == 'en': ax.yaxis.label.set_fontsize(7.4)
        ax.set_xlabel(L('时间 (s)')); ax.set_xlim(0, t[-1]); tag(ax, f'({l})', x=.975 if l == 'a' else .025, ha='right' if l == 'a' else 'left')
    hs = [Line2D([0], [0], color=c, lw=1.2) for c in SRC]+[Patch(fc=mcolors.to_rgba('#4477AA', .28), ec='none'),
          Line2D([0], [0], color=HDVC, lw=.9), Line2D([0], [0], color=HDVC, lw=.9, ls=(0, (4, 1.6))), Line2D([0], [0], color='#0077BB', lw=.9),
          Line2D([0], [0], color='#7A808A', lw=.6, ls=(0, (3, 2))), Patch(fc=mcolors.to_rgba('#EE7733', .18), ec='none')]
    fig.legend(hs, [L('源队列 A'), L('源队列 B'), L('源队列 C'), L('全部网联车辆的范围'), L('群体参考 $v_0$、$a_0$'), L('人工驾驶车辆'),
                    L('网联车辆之间'), L('重组开始与完成'), L('前方人工驾驶车辆变速')], loc='outside upper center', ncol=5, fontsize=6.4,
               columnspacing=.9, handlelength=1.6, handletextpad=.4)
    save(fig, 'fig_main')


# ------------------------------------------------------------------ execution policies
def fig_policies(seed=11):
    pols = ['barrier', 'eager', 'settled']
    ex = {(s_, p): jl(RES/'exec_adaptive'/f's{s_}_{p}/summary.json') for s_ in range(20) for p in pols}
    fig = plt.figure(figsize=(TXT, 3.55), layout='constrained')
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1])
    axes = [fig.add_subplot(gs[0, :])]+[fig.add_subplot(gs[1, k]) for k in range(3)]
    ax = axes[0]; ax.grid(axis='y', visible=False)
    names = [L('排位'), L('换道'), L('压缩')]; alphas = [.25, .55, .95]
    for g, p in enumerate(pols):
        r = ex[(seed, p)]; cs = np.array(r['crossing_start'], float); cd = np.array(r['crossing_done'], float)
        ks = np.array(r['compact_start'], float)
        leave = np.where(np.isfinite(cs), cs, ks)
        spans = [(0., np.nanmax(leave)), (np.nanmin(cs), np.nanmax(cd[np.isfinite(cs)])), (np.nanmin(ks), r['time'])]
        for k, (lo, hi) in enumerate(spans):
            y = g*3.6+k
            ax.barh(y, hi-lo, left=lo, height=.72, color=mcolors.to_rgba(POL[p], alphas[k]), edgecolor=POL[p], lw=.5, hatch=POLH[p])
        ax.text(r['time']+1.5, g*3.6+2, f"{r['time']:.2f}", va='center', ha='left', fontsize=6.4)
    ax.set_yticks([g*3.6+k for g in range(3) for k in range(3)]); ax.set_yticklabels(names*3, fontsize=6.4)
    ax.invert_yaxis(); ax.set_xlim(0, 140); ax.set_xlabel(L('时间 (s)')); ax.tick_params(axis='y', length=0)
    for g, p in enumerate(pols):
        ax.text(-.085 if LANG == 'zh' else -.118, 1-(g*3.6+1+.5)/(2*3.6+3), POLN[p], transform=ax.transAxes, rotation=90, va='center', ha='center',
                fontsize=7 if LANG == 'zh' else 6.2, color=POL[p])
    ax.set_ylim(2*3.6+2.6, -.6); tag(ax, '(a)', x=.975, y=.975, ha='right')
    for ax, k, lab, letter, fmt in zip(axes[1:], ['time', 'acceleration_squared_integral', 'jerk_squared_integral'],
                                       [L('完成时间 (s)'), L(r'控制能量 $E_u$ (m$^2$/s$^3$)'), L(r'输入变化量 $E_j$ (m$^2$/s$^5$)')], 'bcd',
                                       ['{:.2f}', '{:.2f}', '{:.2f}']):
        vals = np.array([[ex[(s_, p)][k] for p in pols] for s_ in range(20)])
        top = (vals.mean(0)+np.array([ci95(vals[:, j]) for j in range(3)])).max()
        ax.set_ylim(0, top*1.16)
        bars(ax, range(3), vals.mean(0), [ci95(vals[:, j]) for j in range(3)], [POL[p] for p in pols], [POLH[p] for p in pols], fmt=fmt)
        ax.set_xticks(range(3)); ax.set_xticklabels([POLN[p][:2]+'\n'+POLN[p][2:] if LANG == 'zh' else POLN[p].replace('-', '-\n') for p in pols], fontsize=6.8 if LANG == 'zh' else 6.2, linespacing=.95); ax.set_ylabel(lab); ax.set_xlim(-.6, 2.6)
        ax.grid(axis='x', visible=False); tag(ax, f'({letter})')
    save(fig, 'fig_policies')


# ------------------------------------------------------------------ mechanisms
def fig_mechanisms():
    fig, axes = plt.subplots(2, 2, figsize=(COL, 3.3), layout='constrained')
    ax = axes[0, 0]
    xs = []
    for g, p in enumerate(['eager', 'settled']):
        for b_, res in enumerate([True, False]):
            okc = 0
            for s_ in range(10):
                f = RES/('exec_adaptive' if res else 'ablation')/(f's{s_}_{p}' if res else f's{s_}_{p}_noreserve')/'summary.json'
                okc += jl(f)['success']
            x = g*2.9+b_*1.25; xs.append(x)
            ax.bar(x, okc, .95, color=mcolors.to_rgba(POL[p], .78 if res else .2), edgecolor=POL[p], lw=.6, hatch='' if res else '////', zorder=2)
            if 10-okc:
                ax.bar(x, 10-okc, .95, bottom=okc, color='#4D4D4D', edgecolor='#4D4D4D', lw=.6, zorder=2)
                ax.text(x, okc+(10-okc)/2, L('停滞')+f' {10-okc}', ha='center', va='center', fontsize=6.2, color='white', zorder=4)
            ax.text(x, 10.25, f'{okc}/10', ha='center', va='bottom', fontsize=6.6)
    ax.set_xticks(xs); ax.set_xticklabels([L('预留'), L('单间隙'), L('预留'), L('单间隙')], fontsize=6.6 if LANG == 'zh' else 6.2)
    ax.set_ylim(0, 13.4); ax.set_xlim(-.75, 2.9+1.25+.75); ax.set_ylabel(L('完成的场景数')); ax.grid(axis='x', visible=False)
    for g, p in enumerate(['eager', 'settled']): ax.text(g*2.9+.625, -2.5, POLN[p], ha='center', va='top', fontsize=7, color=POL[p])
    tag(ax, '(a)')
    pts = collections.defaultdict(list)
    for p in ['barrier', 'eager', 'settled']:
        for s_ in range(4):
            pts[(p, .04)].append(jl(RES/'exec_adaptive'/f's{s_}_{p}/summary.json'))
            for j in [.02, .08]:
                f = RES/'jsweep'/f's{s_}_{p}_j{j}/summary.json'
                if f.exists(): pts[(p, j)].append(jl(f))
    for p in ['barrier', 'eager', 'settled']:
        pts[(p, 'u')] = [jl(RES/'exec_fixed'/f's{s_}_{p}/summary.json') for s_ in range(4)]
    keys = ['u', .02, .04, .08]; xl = [L('均匀\n时长'), '$0.02$', '$0.04$', '$0.08$']
    mk = {'barrier': 'D', 'eager': 'o', 'settled': 's'}
    for ax, k, lab, letter in zip([axes[0, 1], axes[1, 0], axes[1, 1]], ['time', 'acceleration_squared_integral', 'max_ax'],
                                  [L('完成时间 (s)'), r'$E_u$ (m$^2$/s$^3$)', L('峰值纵向加速度 (m/s$^2$)')], 'bcd'):
        for p in ['eager', 'settled']:
            xs_ = [x for x, kk in enumerate(keys) if pts.get((p, kk))]
            ys = [np.mean([r[k] for r in pts[(p, keys[x])]]) for x in xs_]
            ax.plot(xs_, ys, color=POL[p], ls=POLS[p], marker=mk[p], ms=3.2, mfc='white' if p == 'eager' else POL[p], lw=.9, label=POLN[p])
        ax.set_xticks(range(4)); ax.set_xticklabels(xl, fontsize=6.6, linespacing=.95); ax.set_xlim(-.35, 3.35)
        ax.set_xlabel(L(r'参考时长设计（$j_{\rm ref}$, m/s$^3$）'), fontsize=7.2); ax.set_ylabel(lab)
        tag(ax, f'({letter})', x=.975 if letter == 'b' else .025, ha='right' if letter == 'b' else 'left')
        ax.margins(y=.14)
        if letter == 'b': ax.legend(loc='lower left', fontsize=6.4)
        if letter == 'd': ax.set_ylim(0, None)
    save(fig, 'fig_mechanisms')


# ------------------------------------------------------------------ mixed traffic
MIXC = [('steady', L('匀速')), ('b1_t15', L('排位中')), ('b1_t30', L('换道中')), ('b1_t50', L('压缩中')), ('b2_t30', L('强制动')), ('noadapt_t30', L('不调速'))]


def fig_mixed():
    B = RES/'mixed/batch'; rows = {}
    for key, _ in MIXC:
        rows[key] = [jl(B/f's{s_}_{key}/summary.json') for s_ in range(5) if (B/f's{s_}_{key}/summary.json').exists()]
    fig, axes = plt.subplots(2, 2, figsize=(COL, 3.3), layout='constrained')
    hats = ['', '', '', '', '//////', '']
    ax = axes[0, 0]
    for x, (key, nm) in enumerate(MIXC):
        rr = rows[key]; done = [r['completion'] for r in rr if r['success']]
        if done:
            m = np.mean(done); e = ci95(done)
            ax.bar(x, m, .66, color=mcolors.to_rgba('#0077BB', .7), edgecolor='#0077BB', lw=.6, hatch=hats[x], zorder=2)
            ax.errorbar(x, m, yerr=e, color='#30343B', lw=.6, capsize=1.8, zorder=3)
            ax.text(x, m-e-4, f'{m:.2f}', ha='center', va='top', fontsize=6.2, rotation=90, color='white')
        else:
            ax.text(x, 4, L('全部\n相撞'), ha='center', va='bottom', fontsize=6.2, color=OUTC['coll'], linespacing=.95)
    ax.set_ylim(0, 150); ax.set_ylabel(L('完成时间 (s)')); ax.set_xticks(range(6)); ax.set_xticklabels([n for _, n in MIXC], fontsize=6.3, rotation=35)
    ax.grid(axis='x', visible=False); tag(ax, '(a)')
    ax = axes[0, 1]
    for x, (key, nm) in enumerate(MIXC):
        g = [r['min_gap_hdv'] for r in rows[key]]
        c = OUTC['coll'] if key.startswith('noadapt') else '#30343B'
        ax.bar(x, np.min(g), .66, color=mcolors.to_rgba(c, .55), edgecolor=c, lw=.6, hatch=hats[x], zorder=2)
        if key.startswith('noadapt'): ax.text(x, max(np.min(g), 8.)+1.2, f'{np.min(g):.2f}', ha='center', va='bottom', fontsize=6.2, rotation=90, color=OUTC['coll'])
        else: ax.text(x, np.min(g)-1.5, f'{np.min(g):.2f}', ha='center', va='top', fontsize=6.2, rotation=90, color='white')
    ax.axhline(8., color=OUTC['coll'], lw=.7, ls=(0, (3, 2)), label=L('分离阈值 $D_x$'))
    ax.set_ylim(0, 48); ax.set_ylabel(L('最小纵向距离 (m)'))
    ax.set_xticks(range(6)); ax.set_xticklabels([n for _, n in MIXC], fontsize=6.3, rotation=35); ax.grid(axis='x', visible=False)
    ax.legend(loc='upper right', fontsize=6.4); tag(ax, '(b)')
    runs = [('s0_b1_t30', '#0077BB', '-', L('调整群体速度')), ('s0_noadapt_t30', OUTC['coll'], (0, (4, 1.6)), L('不调整群体速度'))]
    for key, c, ls, nm in runs:
        z = np.load(B/key/'trajectory.npz'); s = jl(B/key/'summary.json'); h = s['config']['dt']; fr = z['frame']; Hh = z['hdv']; q = s['hdv']['brake_lane']
        t = np.arange(len(fr))*h
        if key == runs[0][0]:
            axes[1, 0].plot(t, fr[:, 1]+Hh[:, q, 2], color=HDVC, lw=.9, ls=(0, (1, 1.2)), label=L('人工驾驶车辆'))
        axes[1, 0].plot(t, fr[:, 1], color=c, ls=ls, lw=.9, label=nm)
        X = z['trajectory']; lane = np.abs(X[:, :, 1]-Hh[:, q:q+1, 1]) < 2.8
        dx = np.where(lane, Hh[:, q:q+1, 0]-X[:, :, 0], np.inf).min(1)
        axes[1, 1].plot(t, dx, color=c, ls=ls, lw=.9, label=nm)
    axes[1, 0].set_ylabel(L('速度 (m/s)')); axes[1, 0].set_ylim(9.6, 17.6); axes[1, 0].legend(loc='lower right', fontsize=6.2)
    axes[1, 1].axhline(8., color=OUTC['coll'], lw=.7, ls=(0, (3, 2))); axes[1, 1].set_ylim(0, 160)
    axes[1, 1].set_ylabel(L('同车道纵向距离 (m)')); axes[1, 1].legend(loc='upper center', bbox_to_anchor=(.46, 1.), fontsize=6.2, labelspacing=.3, borderaxespad=.3)
    for ax, l in zip(axes[1], 'cd'): ax.set_xlabel(L('时间 (s)')); ax.set_xlim(0, 105); tag(ax, f'({l})')
    save(fig, 'fig_mixed')


# ------------------------------------------------------------------ HDVs inside the platoons
AWC, AGC = '#0077BB', '#CC3311'


def inner_runs():
    I = {}
    for f in sorted((RES/'inner'/'runs').glob('*/summary.json')):
        r = jl(f); I[(r['seed'], r['n_hdv'], r['aware'])] = r
    return I


def hdv_distance(run):
    z = np.load(run/'trajectory.npz'); X = z['trajectory']; Hh = z['hdv']
    lane = np.abs(X[:, :, None, 1]-Hh[:, None, :, 1]) < 2.8
    d = np.where(lane, np.abs(X[:, :, None, 0]-Hh[:, None, :, 0]), np.inf).min(axis=(1, 2))
    return np.arange(len(d))*.4, d


def fig_inner(example=(0, 2)):
    I = inner_runs(); coll = lambda r: r.get('failure') is not None and 'body' in r['failure']
    viol = lambda r: not r['success'] and not coll(r) and min(r.get('min_center_margin_hdv') or 0., r['min_center_margin_cav']) < 0
    fig, axes = plt.subplots(2, 2, figsize=(COL, 3.15), layout='constrained')
    ax = axes[0, 0]; xt, xl = [], []
    for g, n_ in enumerate((1, 2, 3)):
        for b, a_ in enumerate((True, False)):
            cell = [r for (s_, k_, aa), r in I.items() if k_ == n_ and aa == a_]
            x = g*2.6+b*1.1; ok = sum(r['success'] for r in cell); co = sum(coll(r) for r in cell)
            vi = sum(viol(r) for r in cell)
            nf = sum(bool(r.get('plan_failed')) for r in cell); st = len(cell)-ok-co-vi-nf; base = 0
            for v, k in zip([ok, st, vi, co], ['ok', 'stall', 'viol', 'coll']):
                if v: ax.bar(x, v, .9, bottom=base, color=OUTC[k], edgecolor='white', lw=.5, zorder=2); base += v
            if nf: ax.bar(x, nf, .9, bottom=base, color='white', edgecolor='#8A8F98', hatch='//////', lw=.6, zorder=2); base += nf
            xt.append(x); xl.append(L('考虑') if a_ else ('' if LANG == 'zh' else '\n')+L('忽略'))
        ax.text(g*2.6+.55, 5.18, (f'{n_} 辆' if LANG == 'zh' else f'{n_} HDV'+'s'*(n_ > 1)), ha='center', va='bottom', fontsize=6.8)
        if g: ax.axvline(g*2.6-.75, color='#C9CED6', lw=.5, zorder=1)
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=6.4, linespacing=.9); ax.set_ylim(0, 6.6); ax.set_yticks(range(6)); ax.set_xlim(-1.35, 2*2.6+1.85)
    ax.set_ylabel(L('运行次数')); ax.grid(axis='x', visible=False); tag(ax, '(a)', y=1.0)
    hs = [Patch(fc=OUTC[k], ec='white') for k in ['ok', 'stall', 'viol', 'coll']]; names = [L('完成'), L('未完成'), L('违反分离规则'), L('车体相交')]
    if any(r.get('plan_failed') for r in I.values()):
        hs.append(Patch(fc='white', ec='#8A8F98', hatch='//////', lw=.6)); names.append(L('无可行计划'))
    fig.legend(hs, names, loc='outside upper center', ncol=len(names), fontsize=6.6)
    ax = axes[0, 1]
    for n_ in range(4):
        done = [r['completion'] for (s_, k_, aa), r in I.items() if aa and k_ == n_ and r['success']]
        if not done: continue
        m = np.mean(done); e = ci95(done)
        ax.bar(n_, m, .62, color=mcolors.to_rgba(AWC, .7), edgecolor=AWC, lw=.6, zorder=2)
        ax.errorbar(n_, m, yerr=e, color='#30343B', lw=.6, capsize=1.8, capthick=.6, zorder=3)
        ax.scatter(n_+np.linspace(-.18, .18, len(done)), done, s=4, color='#30343B', zorder=4, lw=0)
        ax.text(n_, 6, f'{m:.2f}', ha='center', va='bottom', fontsize=6.2, color='white', rotation=90, zorder=5)
    ax.set_xticks(range(4)); ax.set_xlabel(L('人工驾驶车辆数')); ax.set_ylabel(L('完成时间 (s)')); ax.set_ylim(0, 200)
    ax.grid(axis='x', visible=False); tag(ax, '(b)', y=1.0)
    ax = axes[1, 0]
    for n_ in (1, 2, 3):
        for a_, c, mk, dx in [(True, AWC, 'o', -.14), (False, AGC, '^', .14)]:
            v = [-min(r['hdv_min_acc']) for (s_, k_, aa), r in I.items() if aa == a_ and k_ == n_ and 'hdv_min_acc' in r]
            ax.scatter(np.full(len(v), n_+dx), v, s=9, marker=mk, facecolor='white' if not a_ else c, edgecolor=c, lw=.6, zorder=3,
                       label=(L('考虑') if a_ else L('忽略')) if n_ == 1 else None)
    ax.set_xticks((1, 2, 3)); ax.set_xlim(.5, 3.5); ax.set_xlabel(L('人工驾驶车辆数')); ax.set_ylabel(L('最大减速度 (m/s$^2$)'))
    ax.set_ylim(0, 11.2); ax.set_yticks([0, 2, 4, 6, 8])
    ax.legend(loc='upper right', fontsize=6.2, handletextpad=.3, borderaxespad=.3, ncol=2, columnspacing=.8); ax.grid(axis='x', visible=False); tag(ax, '(c)', y=1.0)
    ax = axes[1, 1]; s_, n_ = example
    for a_, c, ls, nm in [(True, AWC, '-', L('考虑')), (False, AGC, (0, (4, 1.6)), L('忽略'))]:
        run = RES/'inner'/'runs'/f"s{s_}_h{n_}_{'aware' if a_ else 'agnostic'}"
        if not (run/'trajectory.npz').exists(): continue
        t, d = hdv_distance(run); ax.plot(t, d, color=c, ls=ls, lw=.9, label=nm)
        r = I[(s_, n_, a_)]
        if coll(r): ax.plot(t[-1], d[-1], marker='x', ms=4, color=c, mew=.9)
    ax.axhline(8., color='#30343B', lw=.6, ls=(0, (3, 2)), label='$D_x$')
    ax.set_xlabel(L('时间 (s)')); ax.set_ylabel(L('同车道最小纵向距离 (m)')); ax.set_xlim(0, 120); ax.set_ylim(0, 72)
    ax.legend(loc='upper right', fontsize=6.2, ncol=3, handlelength=1.4, columnspacing=.8); tag(ax, '(d)', y=1.0)
    save(fig, 'fig_inner')


# ------------------------------------------------------------------ upstream traffic
TRC = {'barrier': POL['barrier'], 'eager': POL['eager'], 'settled': POL['settled'], 'rear': '#009988'}
TRN = {'barrier': L('统一阶段'), 'eager': L('逐车即时'), 'settled': L('逐车稳定'), 'rear': L('后端基准')}
LANEC = ['#4477AA', '#228833', '#EE7733']


def traffic_rows():
    T = RES/'traffic'; out = {}
    for f in T.glob('*.json'): out[f.stem] = jl(f)
    return out


def lane_stack(ax, x, rows, color, hatch=''):
    per = np.array([[l['delay'] for l in r['lanes']] for r in rows]); tot = per.sum(1)
    base = 0.
    for lane in range(3):
        m = per[:, lane].mean()
        ax.bar(x, m, .66, bottom=base, color=mcolors.to_rgba(LANEC[lane], .30+.25*lane), edgecolor=color, lw=.6, hatch=hatch, zorder=2)
        base += m
    e = ci95(tot); ax.errorbar(x, tot.mean(), yerr=e, color='#30343B', lw=.6, capsize=1.8, capthick=.6, zorder=3)
    ax.text(x, tot.mean()+e+.02*ax.get_ylim()[1], f'{tot.mean():.2f}', ha='center', va='bottom', fontsize=6.4)
    return tot


def group_rear(run, cfg, lane):
    import traffic_impact as ti
    t, X, R = ti.load_group(run, cfg)
    rear = np.full(len(t), np.nan)
    for k in range(len(t)):
        occ = R[k, :, lane]
        if occ.any(): rear[k] = X[k, occ, 0].min()
    return t, rear


def fig_traffic():
    from scenario import Config
    D = traffic_rows()
    fig = plt.figure(figsize=(TXT, 3.95), layout='constrained')
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.08])
    ax = fig.add_subplot(gs[0, 0])
    keys = ['barrier', 'eager', 'settled', 'rear']
    top = max(np.mean([D[f's{s_}_{k}']['delay'] for s_ in range(20) if f's{s_}_{k}' in D]) for k in keys if f's0_{k}' in D)
    ax.set_ylim(0, top*1.45)
    for x, k in enumerate(keys):
        rows = [D[f's{s_}_{k}'] for s_ in range(20) if f's{s_}_{k}' in D]
        if rows: lane_stack(ax, x, rows, TRC[k], POLH.get(k, ''))
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([TRN[k] if LANG == 'zh' else TRN[k].replace('-', '-\n') for k in keys], fontsize=6.8 if LANG == 'zh' else 6.4, linespacing=.95)
    ax.set_ylabel(L('上游总延误 (车$\\cdot$s)')); ax.grid(axis='x', visible=False); tag(ax, '(a)')
    hs = [Patch(fc=mcolors.to_rgba(LANEC[l], .30+.25*l), ec='#30343B', lw=.5) for l in range(3)]
    ax.legend(hs, [L('车道 0'), L('车道 1'), L('车道 2')], loc='upper right', fontsize=6.2, ncol=3, handlelength=1.1, columnspacing=.8, borderaxespad=.3)
    ax = fig.add_subplot(gs[0, 1])
    mk = [('source_blocks', 'SB'), ('mda', 'MDA'), ('motion_speed', 'DP-MS'), ('settled', 'DP-MSL')]
    top = max(np.mean([D[f's{s_}_{k}']['delay'] for s_ in range(6) if f's{s_}_{k}' in D]) for k, _ in mk)
    ax.set_ylim(0, top*1.45)
    for x, (k, nm) in enumerate(mk):
        rows = [D[f's{s_}_{k}'] for s_ in range(6) if f's{s_}_{k}' in D]
        c = METC['combined' if k == 'settled' else k]
        if rows: lane_stack(ax, x, rows, c)
    ax.set_xticks(range(4)); ax.set_xticklabels([nm for _, nm in mk], fontsize=6.8)
    ax.set_ylabel(L('上游总延误 (车$\\cdot$s)')); ax.grid(axis='x', visible=False); tag(ax, '(b)')
    # time-space diagrams of the lane with the largest delay in scenario 1
    lane = int(np.argmax([l['delay'] for l in D['s0_settled']['lanes']]))
    cfg = Config(); norm = mcolors.Normalize(12.5, 16.5)
    from matplotlib.collections import LineCollection
    cmap = matplotlib.colormaps['RdYlBu']
    for col, (k, run, nm) in enumerate([('s0_settled', RES/'exec_adaptive'/'s0_settled', L('逐车稳定')), ('s0_rear', RES/'exec_rear'/'s0_settled', L('后端基准'))]):
        ax = fig.add_subplot(gs[1, col])
        f = RES/'traffic'/f'{k}_upstream.npz'
        if not f.exists(): continue
        z = np.load(f); x = z[f'x{lane}']; v = z[f'v{lane}']; tt = np.arange(len(x))*.4
        for j in range(x.shape[1]):
            pts = np.column_stack([tt, x[:, j]]).reshape(-1, 1, 2); seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(seg, cmap=cmap, norm=norm, lw=.8, rasterized=True); lc.set_array(v[:-1, j]); ax.add_collection(lc)
        t_g, rear = group_rear(run, cfg, lane)
        ax.plot(t_g, rear, color='#30343B', lw=1.0)
        ax.set_xlim(0, tt[-1]); ax.set_ylim(np.nanmin(x)-10, np.nanmax(rear)+70)
        ax.set_xlabel(L('时间 (s)')); ax.set_ylabel(L('群体坐标系中的位置 (m)')); tag(ax, f'({"cd"[col]}) {nm}')
        ax.grid(False)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=fig.axes[2:], location='right', shrink=.9, pad=.01, aspect=22)
    cb.set_label(L('来车速度 (m/s)')); cb.outline.set_linewidth(.4)
    fig.legend([Line2D([0], [0], color='#30343B', lw=1.0)], [L('占用该车道的重组车辆的后端')], loc='outside lower center', fontsize=6.4)
    save(fig, 'fig_traffic')
    return lane


# ------------------------------------------------------------------ robustness
def outcome(cell):
    ok = sum(r['t'] is not None and r['viol'] == 0 for r in cell); coll = sum(r['body_viol'] > 0 for r in cell)
    viol = sum(r['viol'] > 0 and r['body_viol'] == 0 for r in cell); stall = len(cell)-ok-coll-viol
    return ok, stall, viol, coll


def stack(ax, x, cell, width=.7):
    ok, stall, viol, coll = outcome(cell); b = 0
    for n, k in zip([ok, stall, viol, coll], ['ok', 'stall', 'viol', 'coll']):
        if n: ax.bar(x, n, width, bottom=b, color=OUTC[k], edgecolor='white', lw=.5, zorder=2); b += n
    mu = min(r['mu'] for r in cell)
    ax.text(x, len(cell)+.15, L('碰撞') if coll else f'{mu:.2f}', ha='center', va='bottom', fontsize=6.2, color=OUTC['coll'] if coll else '#30343B', rotation=0)


def fig_robust():
    R = jl(RES/'robust_analysis.json')['runs']
    conds = [(.3, .02), (.6, .02), (.9, .02), (.6, .05)]
    preds = [('CE0', lambda r: r['ctrl'] == 'CE' and r['delta'] == 0.), ('CE', lambda r: r['ctrl'] == 'CE' and r['delta'] == .2),
             ('LA', lambda r: r['ctrl'] == 'LA' and abs(r['hat']-r['lag']) < 1e-9)]
    fig = plt.figure(figsize=(TXT, 2.3), layout='constrained')
    gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1])
    ax = fig.add_subplot(gs[0]); xt, xl = [], []; x = 0.
    for g, (lag, w) in enumerate(conds):
        x0 = x
        for b_, (nm, f) in enumerate(preds):
            cell = [r for r in R if f(r) and abs(r['lag']-lag) < 1e-9 and abs(r['w']-w) < 1e-9]
            if cell: stack(ax, x, cell); xt.append(x); xl.append({'CE0': 'CE\n$\\delta{=}0$', 'CE': 'CE', 'LA': 'LA'}[nm]); x += 1.
        ax.text((x0+x-1)/2, 6.05, f'$\\tau{{=}}{lag}$ s\n$w{{=}}{w}$ m/s$^2$', ha='center', va='bottom', fontsize=6.6, linespacing=1.)
        if g < len(conds)-1: ax.axvline(x, color='#C9CED6', lw=.5, zorder=1)
        x += 1.
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=6.3, linespacing=.9); ax.set_ylim(0, 7.4); ax.set_yticks(range(6))
    ax.set_ylabel(L('运行次数')); ax.grid(axis='x', visible=False); ax.set_xlim(-.7, x-1.3)
    hs = [Patch(fc=OUTC[k], ec='white') for k in ['ok', 'stall', 'viol', 'coll']]
    fig.legend(hs, [L('完成且无违规'), L('未完成、无违规'), L('违反分离规则'), L('车体相交')], loc='outside upper center', ncol=4, fontsize=6.6)
    tag(ax, '(a)', x=.006, y=.985)
    ax = fig.add_subplot(gs[1])
    for k, hat in enumerate([.3, .45, .6, .9]):
        cell = [r for r in R if r['ctrl'] == 'LA' and abs(r['lag']-.6) < 1e-9 and abs(r['w']-.02) < 1e-9 and abs(r['hat']-hat) < 1e-9]
        if cell: stack(ax, k, cell)
    ax.set_xticks(range(4)); ax.set_xticklabels(['0.3', '0.45', '0.6', '0.9']); ax.set_xlabel(L(r'假定滞后 $\hat\tau$ (s)，$\tau=0.6$ s'))
    ax.set_ylim(0, 7.4); ax.set_yticks(range(6)); ax.set_ylabel(L('运行次数')); ax.grid(axis='x', visible=False); tag(ax, '(b)', x=.03, y=.985)
    save(fig, 'fig_robust')


# ------------------------------------------------------------------ communication
def fig_comm():
    import platoon_comm as pc
    rows = []
    for f in glob.glob(str(RES/'comm_loop/comm_*_th0.6_*.json')): rows += jl(f)
    fig = plt.figure(figsize=(TXT, 3.7), layout='constrained')
    gs = fig.add_gridspec(2, 2)
    meths = ['source_blocks', 'mda', 'motion_speed', 'combined']
    for k_, (k, lab, letter) in enumerate(zip(['stale_frac', 'rms_error', 'p95_error'], [L('过期计划比例'), L('间距误差均方根 (m)'), L('间距误差 95\\% 分位数 (m)')], 'abc')):
        ax = fig.add_subplot(gs[k_//2, k_ % 2])
        for off, mode, hatch in [(-.19, 'bernoulli', None), (.19, 'ge', '//////')]:
            per = collections.defaultdict(list)
            for r in rows:
                if r['mode'] == mode: per[(r['method'], r['seed'])].append(r[k])
            data = [[np.mean(per[(m, s_)]) for s_ in range(20)] for m in meths]
            b = ax.boxplot(data, positions=[x+off for x in range(4)], widths=.32, patch_artist=True, showfliers=False,
                           medianprops=dict(color='black', lw=.9), whiskerprops=dict(lw=.6), capprops=dict(lw=.6), boxprops=dict(lw=.6))
            for patch, m in zip(b['boxes'], meths):
                patch.set_facecolor(mcolors.to_rgba(METC[m], .55 if mode == 'bernoulli' else .12)); patch.set_edgecolor(METC[m])
                if hatch: patch.set_hatch(hatch)
        ax.set_xticks(range(4)); ax.set_xticklabels([METN[m].replace('DP-', 'DP-\n') for m in meths], fontsize=6.2, linespacing=.9)
        ax.set_ylabel(lab, fontsize=7.4); tag(ax, f'({letter})', x=.96, ha='right')
        if k_ == 0:
            hs = [Rectangle((0, 0), 1, 1, fc=mcolors.to_rgba('#777777', .5), ec='#777777', lw=.5),
                  Rectangle((0, 0), 1, 1, fc=mcolors.to_rgba('#777777', .12), ec='#777777', hatch='//////', lw=.5)]
            ax.legend(hs, [L('伯努利丢包'), L('Gilbert--Elliott 丢包')], loc='lower left', fontsize=6.2)
    ax = fig.add_subplot(gs[1, 1])
    pc.TH = .6
    U = np.random.default_rng(5000).random((10, 2, 90, 11))[0, 0]
    for m in ['motion_speed', 'combined']:
        pl = jl(RES/'planning/plans'/f's0_{m}.json'); q = np.array(pl['scenario']['quality']); ch = pl['assignment']['target_orders'][0]
        quals = np.array([q[a, b] for a, b in zip(ch[:-1], ch[1:])])
        sim = pc.simulate(quals, U, mode='ge')
        e = np.abs(sim['errs']).max(1); t = np.arange(len(e))*pc.h
        ax.plot(t, e, color=METC[m], lw=.9, ls='-' if m == 'combined' else (0, (4, 1.6)), label=METN[m]+L('，平均 ')+f"$q={quals.mean():.2f}$")
    ax.axvspan(4, 14, color='#999999', alpha=.12, lw=0, label=L('头车减速 4 m/s'))
    ax.set_xlabel(L('时间 (s)')); ax.set_ylabel(L('最大间距误差 (m)')); ax.legend(loc='upper right', fontsize=6.2); ax.set_xlim(0, 36)
    ax.set_ylim(0, .48); tag(ax, '(d)')
    save(fig, 'fig_comm')


# ------------------------------------------------------------------ computation
def fig_compute():
    A = jl(RES/'analysis.json'); S = A['scale']; ns = [12, 18, 24, 30, 36]
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.85), layout='constrained')
    ax = axes[0]
    ax.plot(ns, [S[str(n)]['qp_median_ms']['mean'] for n in ns], marker='o', ms=3, color='#0077BB', label=L('中位数'))
    ax.plot(ns, [S[str(n)]['qp_p95_ms']['mean'] for n in ns], marker='s', ms=3, color='#0077BB', ls=(0, (4, 1.6)), mfc='white', label=L('95\\% 分位数'))
    ax.set_xlabel(L('车辆数 $N$')); ax.set_ylabel(L('单次求解时间 (ms)')); ax.set_ylim(0, 22); ax.set_xticks(ns); ax.legend(loc='lower right', fontsize=6.4)
    tag(ax, '(a)')
    ax = axes[1]
    w = [S[str(n)]['wall_per_step'] for n in ns]
    ax.bar(range(5), w, .62, color=[mcolors.to_rgba('#0077BB', .7) if v <= .4 else mcolors.to_rgba('#CC3311', .7) for v in w],
           edgecolor=['#0077BB' if v <= .4 else '#CC3311' for v in w], lw=.6, zorder=2)
    for x, v in enumerate(w): ax.text(x, .415 if .3 < v < .4 else v+.02, f'{v:.2f}', ha='center', va='bottom', fontsize=6.4)
    ax.axhline(.4, color='#30343B', lw=.7, ls=(0, (3, 2)), label=L('采样周期 $h$'))
    ax.set_xticks(range(5)); ax.set_xticklabels([str(n) for n in ns]); ax.set_xlabel(L('车辆数 $N$')); ax.set_ylabel(L('单周期顺序计算时间 (s)'))
    ax.set_ylim(0, 1.05); ax.legend(loc='upper left', fontsize=6.4, bbox_to_anchor=(0, .9)); ax.grid(axis='x', visible=False); tag(ax, '(b)')
    save(fig, 'fig_compute')


if __name__ == '__main__':
    which = ARGS or ['plan_stats', 'main', 'policies', 'mechanisms', 'mixed', 'inner', 'traffic', 'robust', 'comm', 'compute']
    for w in which:
        globals()['fig_'+w](); print('ok', w, flush=True)
