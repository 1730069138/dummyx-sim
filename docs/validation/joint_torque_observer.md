# 关节力矩模型残差外力观察器

本次实现以 `joint_torque` 为默认接触保护输入，`contact` 保留旧实验的桌面高度筛选和正 z 翻转规则。形式 I（admittance）和形式 II（impedance）在新模式下只接收估计力。已有 GroundingDINO、配对实验、绘图脚本和旧实验数据保持不变；记录目录增加力源后缀，并拒绝覆盖同名目录。

## 动力学和符号

力的方向为**环境作用于机器人**，表达在世界坐标系，不对估计的 z 分量翻转。单位使用 rad、s、N、N·m。

MuJoCo 方程：

```
M qdd + qfrc_bias = tau_measured + qfrc_passive + tau_external
qfrc_bias = C(q, qd) + g(q)
tau_external = M qdd + qfrc_bias - qfrc_passive - tau_measured
F_raw = solve(J J.T + damping² I, J tau_external)
```

`tau_measured = data.qfrc_actuator[:6]`；J 为 TCP 平移 Jacobian 的前六列。先计算完整系统的 `M @ qdd`，再取前六行，保留夹爪惯性耦合。核心模块仅需惯性力、bias、passive、关节测量力矩和 J，可由实机动力学库提供。

加速度来自相邻 `qvel` 差分，不读取 `qacc`、`qfrc_constraint`、`qfrc_inverse` 或接触求解力。适配器要求 Euler 积分，并在 `mj_step` 后立即采样；此时质量矩阵、bias、passive、执行器力矩与运动学缓存仍属于积分前状态。Euler 默认隐式处理关节阻尼，因此惯性力加上 `D * (qd_new - qd_old)`，对应 `(M + dt D) qdd`；关闭 EULERDAMP 后不加此项。不能在 step 和观察器采样之间插入 `mj_forward`。当前模型无 tendon damping；适配器不宣称支持任意积分器或 tendon 阻尼模型。

XML 中 damping 已通过 passive 和离散修正处理；`frictionloss=0.01` 是约束摩擦，并非 passive。关节限位、未建模摩擦、模型误差都会进入残差，不能读取总约束力来补偿，否则会同时抹掉接触信号。XML 未被修改。

依据：[MuJoCo 动力学文档](https://mujoco.readthedocs.io/en/3.3.2/computation/)。

## 信号处理与控制时序

原始三维力 → 一阶低通（`alpha = 1-exp(-2π fc dt)`）→ 径向软死区（模长减 deadzone）→ 模长限幅 → 接触进入/退出滞回。未进入接触时控制输入为零。每回合显式清除观察器速度历史、滤波状态与接触状态，原保护内部状态仍由 `reset_scene` 清除。

默认参数：

| CLI | 默认值 | 含义 |
|---|---:|---|
| `--contact_force_source` | `joint_torque` | 新模式；`contact` 为旧输入 |
| `--force_damping` | 0.02 | Jacobian 阻尼 |
| `--force_cutoff_hz` | 20 | 低通截止 Hz |
| `--force_deadzone` | 0.25 | 软死区 N |
| `--force_max_force` | 100 | 输出模长上限 N |
| `--force_enter_force` | 0.8 | 处理后力进入阈值 N |
| `--force_exit_force` | 0.4 | 处理后力退出阈值 N |

一个物理步结束后得到估计，供下一物理步控制使用，因此另有一个物理步的因果延迟。原有几何软着陆和虚拟墙位置判断保留；接触滞回状态不等同于旧 `is_contact`（几何保护启用标志）。保护仍针对水平桌面，并不因此变成任意表面避碰控制器。

安全耗时字段继续包含 APF 和每次策略动作的 50 个保护子步，新模式增加观察器计算耗时；不计入真值评价、记录、物理积分、感知、渲染和 OpenPI。`contact` 模式也计算观察器供离线比较，但该观察器开销不计为它的控制开销。

## NPZ

原有低频字段、峰值和冲量保留原定义。新增物理频率字段有：

| 字段 | 形状/含义 |
|---|---|
| `force_time_s` | N，积分前状态的仿真时间 |
| `f_raw` / `f_estimated` | N×3，原始/滤波死区限幅后的世界力 |
| `f_contact_truth` | N×3，环境对整条机械臂的接触合力 |
| `tau_external` | N×6，关节残差，N·m |
| `force_error` | N×3，`f_estimated - f_contact_truth` |
| `estimated_contact` | N，观察器滞回状态 |
| `truth_contact` | N，真值模长达到 enter_force 的状态，无滞回 |
| `f_control` | N×3，该步实际选取的输入；新模式来自前一步估计 |
| `force_sample_period_s` | 部署日志的物理步周期 |
| `contact_force_source` / `force_observer_config` | 部署日志中的输入来源/JSON 参数 |

新真值根据 geom1/geom2 归属确定正负，去掉机器人内部接触，不做 z 翻转。散放工具的桌面接触不算机械臂受力；抓持后工具接触可通过工具对夹爪的作用反映到机器人合力。旧 `f_real_*` 仍是旧桌面筛选定义，不能与新字段直接混用。

## 复现命令

本机验证环境为 `/home/jun/anaconda3/envs/dummyx_vla/bin/python`，MuJoCo 3.3.2。系统 `python3` 没有安装 MuJoCo。

```bash
cd /home/jun/dummyx-sim
PYTHON=/home/jun/anaconda3/envs/dummyx_vla/bin/python

# 无 OpenPI、无 Viewer，默认新建 outputs/force_observer_<随机后缀>
$PYTHON scripts/tools/calibrate_force_baseline.py --headless
# 等价直接入口
$PYTHON scripts/tools/validate_force_observer.py

$PYTHON -m unittest discover -s tests -v
$PYTHON -m py_compile scripts/deploy/joint_torque_observer.py scripts/deploy/deploy_screwdriver_client.py scripts/tools/calibrate_force_baseline.py scripts/tools/validate_force_observer.py tests/test_joint_torque_observer.py
git diff --check

# 需已有 OpenPI 服务和图形环境；此完整任务未在本次验证中运行
$PYTHON scripts/deploy/deploy_screwdriver_client.py \
  --contact_force_source joint_torque --control_mode both \
  --case 8 --num_episodes 50 --fixed_eval
# 复现旧力输入：将 joint_torque 改为 contact
```

无界面校准沿固定轨迹稳定 1 s，下压 6 s，保持 1 s，抬起 3 s。它不启用保护，用于测量观察器的动态响应，不是保护后接触力上限测试。默认完整场景物理参数保留，只将散放螺丝刀和柱体移到轨迹之外。`--output` 指定目录时拒绝覆盖已有结果文件。

## 2026-09-23 验证记录

最终默认配置结果：`outputs/force_observer_wchmwkg4/metrics.json` 和 `samples.npz`，11000 个样本，步长 1 ms。

| 指标 | 结果 |
|---|---:|
| 接触前原始力平均偏置 xyz | [0.00395, 0.00712, 0.06819] N |
| 接触前原始力模长 RMS | 0.08245 N |
| 接触前处理后偏置/模长 RMS | 0 / 0 N |
| 接触阶段平均估计 z / 真值 z | +68.314 / +68.863 N |
| 接触阶段估计 z 为正的比例 | 100% |
| 接触阶段 z RMSE | 1.74880 N |
| 接触阶段 z 相关系数 | 0.996788 |
| 全程三分量 RMSE | 1.25552 N |
| 首次接触检测延迟 | 0 ms（1 ms 采样分辨率） |
| 最终退出延迟 | 19 ms |
| 接触前假阳性 | 0 |
| 全程非接触样本的假阳性 | 103 / 6605 = 1.56% |
| 接触阶段漏检样本 | 0 / 4395 |

首次真值阈值事件和估计事件都在 5.649 s，最后真值接触在 10.130 s，估计在 10.149 s 退出；103 个假阳性样本发生在抬起阶段的间歇脱离和滤波/滞回保持期间。检测延迟不含下一步控制执行延迟。接触前处理后为零是软死区的效果，不能理解为原始估计没有偏置。

完整测试共 18 项通过，包含原有感知/实验分析测试，以及已知 TCP 力恢复、无外力、隐式阻尼、软死区、低通、限幅、滞回、复位、真值符号与两种保护形式的控制输入隔离。隔离测试将 `mj_contactForce` 替换为抛错函数，估计及两种保护形式仍可运行。

## 实机需要提供的参数与限制

实机参考文件 `/home/jun/下载/deploy_real_arm_vanilla.py` 未被修改。它的执行循环是 6 Hz，位置使用角度，只有夹爪读 `torque/current` 后取绝对值判断；不能直接把这个值当六关节的有符号 N·m 输入。保护观察器应独立于低频策略循环，使用带时间戳的同步反馈。

逐关节至少需要：

- 电机力矩常数 Kt（N·m/A），明确相电流、q 轴电流或总线电流及 RMS/峰值定义；总线电流通常不能直接代入 Kt。
- 减速比 N、电机到关节的方向符号 s、传动效率 η，以及正反驱动差异。
- 电流零偏 I0、驱动器读数单位和尺度、温漂、是否 SDK 已经换算为关节输出力矩。
- 关节零位/方向、编码器单位，同步 q、qd、电流及实际采样间隔；加速度由速度差分，需要评估量化和噪声。
- 连杆质量、质心、惯量、关节轴、几何、重力方向、转子/传动折算惯量、关节阻尼与摩擦、TCP 变换、夹爪和工具负载参数。

基本换算为 `tau_measured_i = s_i * Kt_i * N_i * eta_i * (I_i - I0_i)`；若 SDK 已给关节输出 N·m，不重复乘传动比。实机摩擦模型应放在动力学补偿中，避免与效率补偿重复。

三维平移 Jacobian 只能给出等效 TCP 力。多个连杆接触、接触力矩、工具偏心、关节限位及近奇异位形不能被唯一分离，阻尼还会造成幅值偏差。这里的仿真模型与被观测系统相同，是较理想的验证；没有验证实机动力学误差、电流噪声、总线时延、完整 OpenPI 操作任务或实机保护效果。真实参数辨识后需要重新整定阈值和滤波。
