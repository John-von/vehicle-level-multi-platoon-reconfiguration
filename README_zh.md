# 混行交通中多队列的车辆级重组

本仓库是论文 “Communication-aware planning and distributed control for vehicle-level multi-platoon reconfiguration in mixed traffic” 的程序。多条车道上的网联自动驾驶车辆队列在行驶中重组为新的队列，人工驾驶车辆可能位于车队前方，也可能夹杂在队列中。规划层用精确的动态规划同时决定每辆车的目标队列、交错顺序和新的前后车链路，执行层用逐车触发换道和压缩的顺序分布式模型预测控制执行规划。

English description: [README.md](README.md)

本仓库对应论文 v12，包含代码，不包含预先计算的实验数据（`results/`）和稿件文件。按下方复现步骤运行可生成统计与绘图所需的数据。`python run_exec.py --selftest` 会将一次新运行与已生成的参考结果比较，因此需先完成规划层和 `exec_fixed` 组实验；`python verify.py` 无需实验数据即可运行。

## 运行环境

Python 3.13，依赖见 `requirements.txt`。

```bash
pip install -r requirements.txt
```

`plot_figures.py` 通过 Matplotlib 的 PGF 后端调用 XeLaTeX 排版图中文字，需要安装 TeX Live 和 TeX Gyre Termes 字体。中文图使用 Noto Serif CJK 字体，没有时改用宋体。

## 文件

| 文件 | 内容 | 论文章节 |
| --- | --- | --- |
| `code/scenario.py` | 场景生成、参数和链路质量模型 | 3、6.1 |
| `code/driver_model.py` | 智能驾驶模型及标定的参数分布 | 3 |
| `code/sequence.py` | 保序交错的动态规划 `optimize_joint` 和目标函数各项 | 4 |
| `code/plan_methods.py` | SB、MDA、DP-M、DP-MS 和 DP-MSL 五种规划方法的统一入口 | 6.2 |
| `code/plan_exact.py` | 不截断的精确动态规划 | 6.2 |
| `code/plan_mixed.py` | 源队列中夹杂人工驾驶车辆时的规划 | 4 |
| `code/dmpc.py`、`code/event_mpc.py` | 局部二次规划、参考轨迹、共享计划、接受检验和换道准入 | 5.1、5.2 |
| `code/lag_qp.py`、`code/robust_qp.py` | 计及滞后的预测、约束收紧和软约束恢复 | 5.1 |
| `code/timing.py` | 舒适包络和参考时长 | 5.2 |
| `code/run_exec.py` | 三种执行策略、位次差预留和单间隙消融的闭环仿真 | 5.1、5.2 |
| `code/run_mixed.py` | 车队前方的人工驾驶车辆和群体速度调整 | 5.3 |
| `code/run_inner.py` | 夹杂人工驾驶车辆时的执行 | 5.3 |
| `code/audit_continuous_safety.py` | 采样间分离裕度的精确计算，独立于控制器 | 5.4 |
| `code/cai2022.py` | 对比方法 TAPPA（目标指派与冲突搜索） | 6.2 |
| `code/traffic_impact.py` | 上游车流仿真和延误统计 | 6.3 |
| `code/platoon_comm.py` | 伯努利与 Gilbert-Elliott 丢包条件下的跟驰 | 6.2 |
| `code/batch_exec.py`、`code/batch_mixed.py`、`code/batch_inner.py` | 闭环实验的批处理 | 6 |
| `code/analysis.py`、`code/analysis_lag.py` | 全部运行的统计和连续时间检验 | 6 |
| `code/paper_numbers.py` | 论文中引用的数字和表 2 的数据行 | 6 |
| `code/plot_figures.py`、`code/make_tikz.py` | 论文中的图 | 6 |
| `code/verify.py` | 在小规模算例上穷举检验规划结果 | 4 |

## 实验

以下命令都在 `code/` 下运行，各实验程序的输出写入与 `code/` 同级的 `results/`。

| 实验组 | 命令 | 输出位置 |
| --- | --- | --- |
| EX1 规划方法比较 | `python plan_methods.py --task main` | `results/planning/planning_main_0-99.json`、`results/planning/plans/` |
| EX2 精确解与截断搜索 | `python plan_exact.py --seeds 0 99`，`python plan_methods.py --task exact ...` | `results/planning/exact_main_0-99.json`、`exact_gap_*.json` |
| EX3 三种执行策略 | `python batch_exec.py --groups exec_adaptive` | `results/exec_adaptive/` |
| EX4 各规划方法的闭环执行 | `python batch_exec.py --groups exec_methods` | `results/exec_methods/` |
| EX5 单间隙消融 | `python batch_exec.py --groups ablation` | `results/ablation/` |
| EX6 舒适包络与均匀时长 | `python batch_exec.py --groups jsweep exec_fixed` | `results/jsweep/`、`results/exec_fixed/` |
| EX7 滞后与扰动 | `python batch_exec.py --groups robust lagaware lagmismatch` | `results/robust/`、`results/lagaware/`、`results/lagmismatch/` |
| EX8 丢包条件下的跟驰 | `python platoon_comm.py --mode bernoulli`，`python platoon_comm.py --mode ge` | `results/comm_loop/` |
| EX9 规模与计算时间 | `python batch_exec.py --groups scale` | `results/scale/` |
| EX10 前方人工驾驶车辆 | `python batch_mixed.py` | `results/mixed/batch/` |
| EX11 夹杂人工驾驶车辆 | `python batch_inner.py` | `results/inner/plans/`、`results/inner/runs/` |
| EX12 上游车流 | `python traffic_impact.py` | `results/traffic/` |
| 后端基准布置 | `python batch_exec.py --groups exec_rear` | `results/exec_rear/` |
| EX13 与 TAPPA 比较 | `python cai2022.py` | `results/cai/` |
| 统计与连续时间检验 | `python analysis.py`，`python analysis_lag.py` | `results/analysis.json`、`results/robust_analysis.json`、`results/audit/`、`results/tables/` |

每次闭环运行占一个文件夹，其中 `summary.json` 为指标和设置，`scenario.json` 为场景，`trajectory.npz` 为状态、指令和参考。场景 n 对应随机种子 n-1，文件名中的 `s0` 即场景 1。文件名中的 `barrier`、`eager`、`settled` 分别对应统一阶段、逐车即时和逐车稳定，`source_blocks`、`mda`、`motion_only`、`motion_speed`、`combined` 分别对应 SB、MDA、DP-M、DP-MS 和 DP-MSL。闭环运行所用的 DP 计划由宽度为 1600 的截断搜索得到，`plan_exact.py` 将其与不截断的精确解比较。

## 复现步骤

```bash
cd code

# 在小规模算例上穷举检验规划结果，约几秒
python verify.py

# 规划层
python plan_methods.py --task main
python plan_exact.py --seeds 0 99
python plan_methods.py --task exact --m 4 6 8 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
python plan_methods.py --task exact --m 10 --seeds 0 1 2 3 4 5 6 7 8 9

# 闭环执行，已有结果会跳过，可用 --worker k --workers n 分成 n 个进程
python batch_exec.py --groups exec_adaptive exec_methods ablation jsweep exec_fixed exec_rear scale robust lagaware lagmismatch
python batch_mixed.py
python batch_inner.py

# 重新运行场景 1 并与 exec_fixed 组的结果比较，约 2 分钟
python run_exec.py --selftest

# 对比方法、上游车流和丢包跟驰
python cai2022.py
python traffic_impact.py
python platoon_comm.py --mode bernoulli
python platoon_comm.py --mode ge

# 统计、文中数字和图
python analysis.py
python analysis_lag.py
python paper_numbers.py
python plot_figures.py zh plan_stats main policies inner traffic
python make_tikz.py zh
```

批处理程序遇到已有的 `summary.json` 会跳过，重新运行某一组之前需先删除对应的结果文件夹。`paper_numbers.py` 把 `results.tex` 和 `tab_cai_rows.tex` 写入 `manuscript/zh` 和 `manuscript/en`，`plot_figures.py` 把图写入 `manuscript/<lang>/figs`，`make_tikz.py` 把场景示意图和一幅论文中未使用的示例图的 TikZ 源文件写入 `manuscript/<lang>/figs_tikz`，`<lang>` 为 `zh` 或 `en`。`plot_figures.py` 不指定图名时会生成 10 幅图，其中 5 幅论文中没有使用。

## 论文图表与程序的对应

| 图表 | 程序 |
| --- | --- |
| 图 1 场景示意 | `make_tikz.py` |
| 图 3 规划方法统计 | `plot_figures.py plan_stats` |
| 图 4 完整重组过程 | `plot_figures.py main` |
| 图 5 执行策略比较 | `plot_figures.py policies` |
| 图 6 夹杂人工驾驶车辆 | `plot_figures.py inner` |
| 图 7 上游车流 | `plot_figures.py traffic` |
| 表 2 与 TAPPA 的比较 | `paper_numbers.py` |
| 正文中的数字 | `paper_numbers.py` |

图 2 为控制框图，表 1 列出 `scenario.py` 和各运行程序中设定的参数。

## 运行环境与时间

论文结果在 Python 3.13、NumPy 2.5、SciPy 1.18、Clarabel 0.11.1、OSQP 1.1 和 Matplotlib 3.11 下单线程得到，全部闭环运行合计约 12 小时。计算时间与机器有关，重新运行后规划和求解时间以及少量与求解精度有关的末位数字可能略有变化。
