# APF 与 VLSA 数据对比口径

## 耗时对比

两套脚本现在都记录 `safety_layer_time_ms`。VLSA 的边界为接收当前状态和名义动作后，
直到生成安全关节动作，包含几何/Jacobian、CBF/QP 和关节动作适配。你的方案记录每个外层
控制周期内的总安全计算量：APF 宏观避障，加上启用时 50 个子步的接触力提取和接触保护
计算。它还分别保存 `apf_time_ms` 和 `contact_controller_time_ms`，便于做算法内部消融。
两边均不包含 OpenPI 网络推理、图像渲染、Grounding DINO、视频编码和 MuJoCo 物理推进。

VLSA 的 `qp_time_ms` 仍然保留，用来展示其内部 QP 求解器占用；它不能直接与 APF 的
`apf_time_ms` 或整套方案的 `safety_layer_time_ms` 混为同一指标。主横向结论应使用两边
都有的 `safety_layer_time_ms`；内部组成在另一张堆叠图或表格展示。

PPT 建议同时报告 mean、P50、P95、P99、max 和控制周期超时率。均值体现总体开销，
P95/P99 和超时率体现实时稳定性。首次迭代可能包含求解器或缓存预热，正式统计可以分别
报告“含首次迭代”和“去除每回合首个样本”，但必须在图注中说明，不能只删除较慢样本。

必须在同一台机器、同一电源/GPU模式、相同场景、相同障碍来源、相同配对初始序列、相同
回合数和控制周期下采集。两套脚本使用相同 `--eval_seed`，会生成逐回合一致、跨回合变化的
目标坐标并写入 NPZ；出图工具会检查坐标是否逐回合匹配。先运行少量预热回合，再交替运行
APF、VLSA，减少温度和后台负载造成的顺序偏差。建议每种方法至少 30 回合；最终数值同时
保留逐回合原始数据。

控制算法本身的正式配对实验建议先统一使用 oracle 障碍几何，隔离感知误差：

```bash
cd /home/jun/dummyx-sim

# B8：APF，无接触保护
/home/jun/anaconda3/envs/vlsa_env/bin/python \
  scripts/deploy/deploy_screwdriver_client.py \
  --host localhost --port 8000 \
  --case 4 --num_episodes 50 --eval_seed 20260923 \
  --obstacle_source oracle --settle_before_start

# B11（形式 I）：APF + 导纳接触保护
/home/jun/anaconda3/envs/vlsa_env/bin/python \
  scripts/deploy/deploy_screwdriver_client.py \
  --host localhost --port 8000 \
  --case 8 --control_mode admittance \
  --num_episodes 50 --eval_seed 20260923 \
  --obstacle_source oracle --settle_before_start

# B12（形式 II）：APF + 阻抗接触保护
/home/jun/anaconda3/envs/vlsa_env/bin/python \
  scripts/deploy/deploy_screwdriver_client.py \
  --host localhost --port 8000 \
  --case 8 --control_mode impedance \
  --num_episodes 50 --eval_seed 20260923 \
  --obstacle_source oracle --settle_before_start

# VLSA/AEGIS
/home/jun/anaconda3/envs/vlsa_env/bin/python \
  scripts/deploy/deploy_screwdriver_client_vlsa.py \
  --host localhost --port 8000 \
  --mode vlsa --obstacle_source oracle \
  --num_episodes 50 --max_steps 600 \
  --eval_seed 20260923 --settle_before_start --trace_window 0
```

如果论文还要比较端到端感知系统，再另做一组 Grounding DINO/perception 实验，并把感知耗时、
定位误差和感知失败率单列。不要把 oracle 控制结果和视觉感知结果混在同一统计总体。

## 一键生成表格和图片

重新采集 APF 和 VLSA 数据后执行：

```bash
cd /home/jun/dummyx-sim
/home/jun/anaconda3/envs/vlsa_env/bin/python \
  scripts/tools/compare_safety_algorithms.py \
  --case8-run recordings/run_20260923_111805_none_Case4_obs_apf_no_cont_oracle \
  --case11-run recordings/run_20260923_113002_admittance_Case8_obs_apf_cont_oracle \
  --case12-run recordings/run_20260923_114855_impedance_Case8_obs_apf_cont_oracle \
  --vlsa-run recordings/run_20260923_120805_vlsa_oracle_obs1 \
  --output-dir outputs/B8_B11_B12_VLSA_50episodes_seed20260923
```

本次正式配对实验的四组数据均为 50 回合，回合编号为 1–50，且每个回合的初始目标坐标逐元素
完全一致。结果为：B8 的 SR/CR 为 94%/0%，B11（形式 I）为 48%/16%，B12（形式 II）为
48%/16%，VLSA 为 8%/0%。四种方法的安全层平均耗时分别为 0.115、3.871、4.072 和
5.521 ms，50 ms 控制周期超时率均为 0%。完整均值、成功回合均值和置信区间以输出 CSV 为准。

输出包括：

- `latency_summary.csv`：安全层耗时及控制周期超时率。
- `method_summary.csv`：各方法的 50 回合总体均值、成功回合均值、SR/CR 和置信区间。
- `episode_metrics.csv`：每回合成功、碰撞、完成时间、TCP 路径、逐回合延迟统计、力、冲量和力矩。
- `latency_comparison.png`：P50/P95/P99 柱状图和 ECDF 尾延迟图。
- `compute_breakdown.png`：你的 APF/接触保护与 VLSA QP/动作适配的平均耗时组成。
- `average_metrics_comparison.png`：全回合均值与成功回合均值并列比较。
- `task_metrics_comparison.png`：成功率/碰撞率的 95% Wilson 区间，以及成功回合的完成时间、
  TCP 路径长度和接触冲量箱线图。

旧 APF 数据没有统一耗时字段；旧 VLSA 数据只有 `qp_time_ms`。工具会拒绝这类数据，必须用
修改后的脚本重新运行，防止错误对比。

## 最值得加入 PPT 的指标

第一组应回答“能不能安全完成”：

- 任务成功率 SR，附样本数和 95% Wilson 置信区间。
- 碰撞/撞倒率 CR，附 95% Wilson 置信区间。
- 感知失败率；使用 Grounding DINO 时单列，不能混入控制算法失败。
- 最小安全裕度。APF 的欧氏间隙和 VLSA 的 `h` 定义不同，不能直接画在同一纵轴；若要
  横向比较，应新增统一的真值几何间隙评估器，并仅把真值用于离线评价。

第二组应回答“代价多大”：

- 安全层 mean/P95/P99 延迟和控制周期超时率。
- 成功回合完成时间、TCP 路径长度、策略调用次数。
- 干预率和最大/平均干预量。当前 APF 修正是关节增量范数，VLSA 修正是 twist 范数，单位
  不同；可分别展示，不能把原始数值并排声称大小优劣。公平横向比较需要统一成最终关节
  命令差 `||q_safe-q_nom||` 或 TCP 速度差。

第三组应回答“运动是否平稳、接触是否温和”：

- 峰值接触力、接触冲量、峰值关节力矩。
- 关节速度/加速度/jerk 的 RMS 和峰值，最好只统计执行阶段并保持同一滤波方法。
- 成功回合指标和全回合指标分开，避免失败回合提前结束使平均力或路径看起来更小。

如果只能放一页，优先选择：SR、CR、安全层 P95 延迟、超时率、成功回合接触冲量和完成
时间。表格放 mean ± std 和样本数；图中用全部回合的散点/箱线图以及置信区间，不只画均值柱。
