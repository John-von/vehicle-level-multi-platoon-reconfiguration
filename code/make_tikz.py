"""Write the TikZ sources of the scenario figure and the plan figure from the result files."""
from pathlib import Path
import json, sys
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT/'results'
LANG = sys.argv[1] if len(sys.argv) > 1 else 'zh'
OUT = ROOT/'manuscript'/LANG/'figs_tikz'
COL = ['pA', 'pB', 'pC']
TXT = dict(zh=dict(lane='车道', drive='行驶方向',
                   ia=r'$t=0$~s，源队列中夹杂人工驾驶车辆H，阴影为其跟车区间',
                   ib=r'$t={tb:.0f}$~s，网联车辆在区间之外排位和换道',
                   ic=r'$t={tc:.0f}$~s，车道~0 的H成为目标队列成员，车道~2 的H留在原车道',
                   cap_i=r'夹杂人工驾驶车辆时的多队列重组（场景~1，两辆人工驾驶车辆）。',
                   d0='初始间距', tgt='目标队列', src='源队列', rank='目标队列中的位次（从前到后）', mean='平均', minq='最小', weak='弱链路',
                   cap_p=r'三种规划方法给出的目标队列及相邻成员间的链路质量 $q_{ji}$，红色数值表示弱链路（场景 1）。'),
           en=dict(lane='Lane', drive='Driving direction',
                   ia=r'$t=0$~s, HDVs (H) inside the source platoons, with their following intervals shaded',
                   ib=r'$t={tb:.0f}$~s, CAVs stage and change lanes outside the intervals',
                   ic=r'$t={tc:.0f}$~s, H in lane~0 is a member of a target platoon and H in lane~2 stays in its lane',
                   cap_i=r'Multi-platoon reconfiguration with HDVs inside the platoons (scenario~1, two HDVs).',
                   d0='Initial spacing', tgt='Target', src='Source platoon', rank='Rank in the target platoon (front to rear)', mean='Mean', minq='Min.', weak='Weak links',
                   cap_p=r'Target platoons given by three planning methods and the link quality $q_{ji}$ between adjacent members, with weak links in red (scenario~1).'))[LANG]


def car(x, y, ang, col, label, hdv=False):
    rot = f'rotate={ang:.2f},' if abs(ang) > .05 else ''
    if hdv:
        return f'\\pic[{rot}carlabel={{{label}}}] at ({x:.3f},{y:.3f}) {{hdv=pH!55!black}};'
    return f'\\pic[{rot}carlabel={{{label}}}] at ({x:.3f},{y:.3f}) {{car={col}}};'


def scenario_inner(seed=0, n_hdv=2, tb=40.):
    from plan_mixed import ZONE
    run = RES/'inner'/'runs'/f's{seed}_h{n_hdv}_aware'; z = np.load(run/'trajectory.npz')
    s = json.loads((run/'summary.json').read_text(encoding='utf-8'))
    pl = json.loads((RES/'inner'/'plans'/f's{seed}_h{n_hdv}_aware.json').read_text(encoding='utf-8'))
    sc, asg = pl['scenario'], pl['assignment']
    X = z['trajectory']; Hh = z['hdv']; cav = list(z['cav']); h = sc['config']['dt']; w = sc['config']['lane_width']
    v0 = sc['config']['cruise']; src = np.array(sc['source'])
    labels = {}
    for c in range(3):
        for r, i in enumerate([i for i in cav if src[i] == c]): labels[i] = f'{"ABC"[c]}{r+1}'
    tc = s['completion']; snaps = [0., tb, tc]; ks = [min(int(round(t/h)), len(X)-1) for t in snaps]
    xs = np.concatenate([X[k, :, 0] for k in ks]+[Hh[k, :, 0] for k in ks[:2]])
    xmin, xmax = xs.min()-8, xs.max()+8
    W = 12.0; kx = W/(xmax-xmin); lw = .30; ky = lw/w; gapy = 1.55
    TX = dict(a=TXT['ia'], b=TXT['ib'].format(tb=tb), c=TXT['ic'].format(tc=tc))
    L = [r'\begin{figure}[t]', r'\centering', r'\resizebox{0.874\textwidth}{!}{%',
         r'\begin{tikzpicture}[x=1cm,y=1cm,>={Stealth[length=3.4pt,width=2.8pt]},font=\fontsize{8}{9.6}\selectfont,carlen=.38,carwid=.17,carfont=3.6,',
         r'  dim/.style={{Bar[width=2.6pt]Stealth[length=2.4pt]}-{Stealth[length=2.4pt]Bar[width=2.6pt]},line width=.4pt}]']
    for p, (k, key) in enumerate(zip(ks, 'abc')):
        y0 = -p*gapy
        L.append(f'\\fill[road] (0,{y0-.02:.3f}) rectangle ({W:.3f},{y0+3*lw+.02:.3f});')
        if key in 'ab':
            for q in range(Hh.shape[1]):
                lane = int(round(Hh[k, q, 1]/w)); xa = max(0., (Hh[k, q, 0]-ZONE['rear']-xmin)*kx); xb = min(W, (Hh[k, q, 0]+ZONE['front']-xmin)*kx)
                L.append(f'\\fill[pH!22] ({xa:.3f},{y0+lane*lw:.3f}) rectangle ({xb:.3f},{y0+(lane+1)*lw:.3f});')
        L.append(f'\\draw[line width=.55pt,black!75] (0,{y0:.3f}) -- ({W:.3f},{y0:.3f}); \\draw[line width=.55pt,black!75] (0,{y0+3*lw:.3f}) -- ({W:.3f},{y0+3*lw:.3f});')
        for q in (1, 2):
            L.append(f'\\draw[lanemark,line width=.4pt,dash pattern=on 3pt off 3pt] (0,{y0+q*lw:.3f}) -- ({W:.3f},{y0+q*lw:.3f});')
        for q in range(3):
            L.append(f'\\node[anchor=west,inner sep=1pt,font=\\fontsize{{7}}{{8}}\\selectfont] at ({W+.08:.3f},{y0+(q+.5)*lw:.3f}) {{{TXT["lane"]}~{q}}};')
        L.append(f'\\node[anchor=south west,inner sep=0pt] at (0,{y0+3*lw+.09:.3f}) {{\\textbf{{({key})}}\\ {TX[key]}}};')
        for i in range(X.shape[1]):
            px, py, vx, vy = X[k, i]
            ang = np.degrees(np.arctan2(vy*ky, (v0+vx)*kx))
            L.append(car((px-xmin)*kx, y0+(py/w+.5)*lw, ang, COL[src[cav[i]]], labels[cav[i]]))
        for q in range(Hh.shape[1]):
            px, py = Hh[k, q, :2]; xx = (px-xmin)*kx
            if xx > W-.3:
                xx = W-.3; L.append(f'\\draw[->,line width=.5pt,black!55] ({W-.06:.3f},{y0+(py/w+.5)*lw:.3f}) -- ({W+.0:.3f},{y0+(py/w+.5)*lw:.3f});')
            L.append(car(xx, y0+(py/w+.5)*lw, 0., 'pH', 'H', hdv=True))
        if key == 'a':
            if LANG == 'zh':
                L.append(f'\\draw[->,line width=.6pt] ({W-2.3:.3f},{y0+3*lw+.2:.3f}) -- ({W-1.0:.3f},{y0+3*lw+.2:.3f}); \\node[anchor=east,inner sep=1pt,font=\\fontsize{{7}}{{8}}\\selectfont] at ({W-2.35:.3f},{y0+3*lw+.2:.3f}) {{{TXT["drive"]}}};')
            else:
                L.append(f'\\draw[->,line width=.6pt] ({W-1.75:.3f},{y0+1.5*lw:.3f}) -- ({W-.45:.3f},{y0+1.5*lw:.3f}); \\node[inner sep=1pt,font=\\fontsize{{7}}{{8}}\\selectfont] at ({W-1.1:.3f},{y0+2.5*lw:.3f}) {{{TXT["drive"]}}};')
            a1, a2 = [j for j, i in enumerate(cav) if src[i] == 1][:2]
            xa, xb = sorted([(X[k, a1, 0]-xmin)*kx, (X[k, a2, 0]-xmin)*kx])
            L.append(f'\\draw[dim] ({xa:.3f},{y0-.12:.3f}) -- ({xb:.3f},{y0-.12:.3f}); \\node[anchor=west,inner sep=1pt] at ({xb+.03:.3f},{y0-.12:.3f}) {{{TXT["d0"]}}};')
            q = 0; hx = Hh[k, q, 0]; xa, xb = (hx-ZONE['rear']-xmin)*kx, (hx+ZONE['front']-xmin)*kx
            L.append(f'\\draw[dim] ({xa:.3f},{y0-.12:.3f}) -- ({(hx-xmin)*kx:.3f},{y0-.12:.3f}); \\node[anchor=east,inner sep=1pt] at ({xa-.03:.3f},{y0-.12:.3f}) {{$D_{{\\rm h}}^{{\\rm r}}$}};')
            L.append(f'\\draw[dim] ({(hx-xmin)*kx:.3f},{y0-.12:.3f}) -- ({xb:.3f},{y0-.12:.3f}); \\node[anchor=west,inner sep=1pt] at ({xb+.03:.3f},{y0-.12:.3f}) {{$D_{{\\rm h}}^{{\\rm f}}$}};')
        if key == 'b':
            st = np.array(asg['stage_x']); order = [i for i in asg['order'] if i in cav]
            xa, xb = sorted([(st[order[3]]-xmin)*kx, (st[order[4]]-xmin)*kx])
            L.append(f'\\draw[dim] ({xa:.3f},{y0-.12:.3f}) -- ({xb:.3f},{y0-.12:.3f}); \\node[anchor=west,inner sep=1pt] at ({xb+.03:.3f},{y0-.12:.3f}) {{$D_{{\\rm s}}$}};')
        if key == 'c':
            full0 = asg['target_orders_full'][0]; hq = [i for i in full0 if i not in cav]
            if hq and full0.index(hq[0])+1 < len(full0):
                hi_ = full0.index(hq[0]); fol = cav.index(full0[hi_+1]); hx = (Hh[k, 0, 0]-xmin)*kx if int(round(Hh[k, 0, 1]/w)) == 0 else None
                if hx is not None:
                    xf = (X[k, fol, 0]-xmin)*kx
                    L.append(f'\\draw[dim] ({xf:.3f},{y0-.12:.3f}) -- ({hx:.3f},{y0-.12:.3f}); \\node[anchor=east,inner sep=1pt] at ({xf-.03:.3f},{y0-.12:.3f}) {{$D_{{\\rm h}}^{{\\rm r}}$}};')
            ch = [i for i in asg['target_orders_full'][1] if i in cav]
            j2, j3 = cav.index(ch[2]), cav.index(ch[3])
            xa, xb = sorted([(X[k, j2, 0]-xmin)*kx, (X[k, j3, 0]-xmin)*kx])
            L.append(f'\\draw[dim] ({xa:.3f},{y0-.12:.3f}) -- ({xb:.3f},{y0-.12:.3f}); \\node[anchor=west,inner sep=1pt] at ({xb+.03:.3f},{y0-.12:.3f}) {{$D_{{\\rm f}}$}};')
    L += [r'\end{tikzpicture}}', f'\\caption{{{TXT["cap_i"]}}}', r'\label{fig:scenario}', r'\end{figure}']
    (OUT/'fig_scenario.tex').write_text('\n'.join(L)+'\n', encoding='utf-8')


def viridis(v):
    import matplotlib
    r, g, b, _ = matplotlib.colormaps['viridis'](v)
    return f'{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}'


def plan_case(seed=0):
    names = [('source_blocks', 'SB'), ('mda', 'MDA'), ('combined', 'DP-MSL')]
    lo, hi = .70, .92
    dx = .86; y_t = [.0, -.46]; blockh = 1.36
    L = [r'\begin{figure}[htbp]', r'\centering', r'\resizebox{\textwidth}{!}{%',
         r'\begin{tikzpicture}[x=1cm,y=1cm,font=\fontsize{8}{9.6}\selectfont,carlen=.5,carwid=.22,carfont=4.2]']
    for b, (m, nm) in enumerate(names):
        pl = json.loads((RES/'planning'/'plans'/f's{seed}_{m}.json').read_text(encoding='utf-8'))
        sc, a = pl['scenario'], pl['assignment']; q = np.array(sc['quality']); src = np.array(sc['source']); lab = sc['labels']
        y0 = -b*blockh
        qs = []
        for d, chain in enumerate(a['target_orders']):
            yy = y0+y_t[d]
            L.append(f'\\node[anchor=east,inner sep=1pt] at (.55,{yy:.3f}) {{{TXT["tgt"]}~{d}}};')
            for r, (i, j) in enumerate(zip(chain[:-1], chain[1:])):
                v = q[i, j]; qs.append(v); c = viridis((v-lo)/(hi-lo))
                xa, xb = 1.1+r*dx+.26, 1.1+(r+1)*dx-.26
                L.append(f'\\definecolor{{q{b}{d}{r}}}{{HTML}}{{{c}}}\\draw[q{b}{d}{r},line width=2.2pt] ({xa:.3f},{yy:.3f}) -- ({xb:.3f},{yy:.3f});')
                tc = 'brake' if round(v, 2) < .78 else 'black!80'
                L.append(f'\\node[anchor={"south" if d == 0 else "north"},inner sep=1pt,font=\\fontsize{{7}}{{8}}\\selectfont,text={tc}] at ({(xa+xb)/2:.3f},{yy+(.09 if d == 0 else -.09):.3f}) {{{v:.2f}}};')
            for r, i in enumerate(chain):
                L.append(f'\\pic[carlabel={{{lab[i]}}}] at ({1.1+r*dx:.3f},{yy:.3f}) {{car={COL[src[i]]}}};')
        qs = np.array(qs)
        L.append(f'\\node[anchor=north west,align=left,inner sep=0pt,font=\\fontsize{{7.5}}{{8.6}}\\selectfont] at ({1.1+11*dx+.45:.3f},{y0+.3:.3f}) {{\\textbf{{({"abc"[b]})~{nm}}}\\\\{TXT["mean"]}~{qs.mean():.2f}\\\\{TXT["minq"]}~{qs.min():.2f}\\\\{TXT["weak"]}~{int((np.round(qs, 2) < .78).sum())}}};')
        if b < 2: L.append(f'\\draw[black!35,line width=.4pt] (-.9,{y0-blockh+.5:.3f}) -- ({1.1+11*dx+.35:.3f},{y0-blockh+.5:.3f});')
    yb = -2*blockh-.98
    for r in range(12):
        L.append(f'\\node[inner sep=1pt] at ({1.1+r*dx:.3f},{yb:.3f}) {{{r+1}}};')
    L.append(f'\\node[inner sep=1pt] at ({1.1+5.5*dx:.3f},{yb-.32:.3f}) {{{TXT["rank"]}}};')
    # colour bar
    xc = 1.1+11*dx+1.95; ytop = .3; ybot = -2*blockh-.62; n = 40
    for k in range(n):
        c = viridis(k/(n-1)); ya = ybot+(ytop-ybot)*k/n; yb2 = ybot+(ytop-ybot)*(k+1)/n
        L.append(f'\\definecolor{{cb{k}}}{{HTML}}{{{c}}}\\fill[cb{k}] ({xc:.3f},{ya:.4f}) rectangle ({xc+.18:.3f},{yb2+.002:.4f});')
    L.append(f'\\draw[black!60,line width=.3pt] ({xc:.3f},{ybot:.3f}) rectangle ({xc+.18:.3f},{ytop:.3f});')
    for v in [.70, .75, .80, .85, .90]:
        yy = ybot+(ytop-ybot)*(v-lo)/(hi-lo)
        L.append(f'\\draw[black!60,line width=.3pt] ({xc+.18:.3f},{yy:.3f}) -- ({xc+.24:.3f},{yy:.3f}); \\node[anchor=west,inner sep=1pt,font=\\fontsize{{7}}{{8}}\\selectfont] at ({xc+.24:.3f},{yy:.3f}) {{{v:.2f}}};')
    L.append(f'\\node[rotate=90,anchor=north,inner sep=1pt] at ({xc+.78:.3f},{(ytop+ybot)/2:.3f}) {{$q_{{ji}}$}};')
    # legend of source colours
    ly = .75; step = 2.3 if LANG == 'zh' else 3.1; x0 = 2.6 if LANG == 'zh' else 1.9
    for c in range(3):
        L.append(f'\\pic[carlabel={{}}] at ({x0+c*step:.3f},{ly:.3f}) {{car={COL[c]}}}; \\node[anchor=west,inner sep=1pt] at ({x0+.32+c*step:.3f},{ly:.3f}) {{{TXT["src"]}~{"ABC"[c]}}};')
    L += [r'\end{tikzpicture}}', f'\\caption{{{TXT["cap_p"]}}}', r'\label{fig:plancase}', r'\end{figure}']
    (OUT/'fig_plan_case.tex').write_text('\n'.join(L)+'\n', encoding='utf-8')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)

    scenario_inner(); plan_case()
