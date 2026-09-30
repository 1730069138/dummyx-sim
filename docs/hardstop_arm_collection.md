# AGV 场景螺丝刀采集

`scripts/collect/collect_hardstop_screwdriver.py` 直接加载 `models/agv_dummyx/scene.xml`，执行“从桌面抓螺丝刀放进收纳盒”。底盘固定，场景有工厂背景、工作台和收纳盒，**没有障碍物**。机械臂从 `models/agv_dummyx/z_pose.json` 的中立 Z 型姿态开始，关节 1 为 175°，关节 2 为 0。物体只在每条轨迹复位时设置位姿；抓取、搬运、松手全靠 MuJoCo 接触和摩擦。

正常采集 50 条随机初态的成功轨迹，追加到默认的 `datasets/agv_dummyx_screwdriver/`：

```bash
conda activate dummyx_vla
cd /home/jun/dummyx-sim
python scripts/collect/collect_hardstop_screwdriver.py \
  --num_episodes 50 --seed 42 --realtime
```

`--num_episodes` 指本次运行要新增的成功条数。默认会持续尝试，直到成功条数达到目标；调试时可显式传入 `--max_attempts` 限制总候选数。默认统一写入 `datasets/agv_dummyx_screwdriver/`，成功轨迹按 `ep_0000`、`ep_0001` 等连续编号；再次运行从已有最大编号后追加。失败记录在 `failures/attempt_####/`，每次运行的汇总在 `runs/run_####.json`，不会覆盖已有文件。也可用 `--output` 指定另一数据集目录，同样按编号续录。`--fixed_eval` 使用柄中心 `(0.415, -0.16)` m 和朝向 90°，适合检查固定轨迹；不代表随机范围的结果。`--no_images` 只用于快速物理诊断，不能产出图像训练数据。不加 `--headless` 会打开 MuJoCo Viewer，`--realtime` 才按仿真速率播放。

要在窗口里看完整抓放过程，但不计入正式数据集，指定独立的演示输出目录：

```bash
conda activate dummyx_vla
cd /home/jun/dummyx-sim
python scripts/collect/collect_hardstop_screwdriver.py \
  --fixed_eval --realtime --num_episodes 1 \
  --output outputs/agv_dummyx/collection_demo
```

演示命令仍会在独立目录保存轨迹、图像和运行记录，但不会写入 `datasets/agv_dummyx_screwdriver/`。采集 Viewer 默认显示机械臂、工作台和青色随机位置矩形；独立保存的相机图像不包含该矩形。看随机过程时删去 `--fixed_eval`。在有桌面的终端运行，不要设置 `MUJOCO_GL=egl` 或加 `--headless`。

随机候选的**柄中心**在 `models/agv_dummyx/spawn_region.json` 的矩形内分别均匀取 X、Y；平放朝向以固定评估的 90° 为中心，在 **60°～120°（±30°）** 内均匀采样。整把工具越过桌边、进入预留障碍空间、逆运动学不可达、丢失双指接触或未稳定留盒的候选保存在 `failures/`，并计入本次的 `runs/run_####.json` 汇总。保留的成功轨迹经过可达性及执行筛选，**不能解释为整块矩形和完整朝向范围上的均匀成功数据**。预留障碍只用于筛样；模型中没有障碍碰撞体。随机区域的青色矩形是 Viewer 调试覆盖层，不在采集相机图像里。

每个 `ep_####` 包含 `cam_fixed/` 的 **640×360 D415 RGB 原始比例** JPEG、`cam_wrist/` 的 **256×256** JPEG、`joint_data.npz`、`metadata.json`、`instruction.txt`。杆顶 `overview` 与 `d415_rgb` 共用一个光学视角，采集器使用 `overview`；腕部相机使用 `wrist_cam`。新采集的数据版本为 `agv_dummyx_screwdriver_v3`，旧 `v2` 数据仍可用 `scene_pre_d415.xml` 回放。采样率 50 Hz；7 维状态和动作依次是关节 1～6（rad）与双指总开口（m）。记录在每次 20 ms 控制周期之前，动作是随后周期的绝对位置目标。`actions_exact` 保留执行用的 float64 目标，`actions` 为 float32 训练数组。元数据保存初始采样位姿、失败原因或成功判据，以及模型文件哈希。D415 深度、真实镜头畸变和噪声尚未模拟。

成功要求：夹爪下降时保持全开，到抓取位置后闭合；抓取后先抬升到 TCP 高度 0.915 m，转正和水平搬运的 TCP 目标高度提高到 0.965 m。转正后如果整把工具的最低点仍低于盒沿 0.853 m，保持夹持并先向上补抬末端，再转运到收纳盒。当前无障碍采集不限制预留立柱顶部高度，搬运时工具最低点仍须高于盒沿。搬运阶段持续双指接触；释放后工具的几何中心位于盒内、下端进入盒子、与盒子有实际接触、速度低于 0.03 m/s，且开爪无接触连续保持至少 1 秒。允许斜靠盒沿和部分顶点越过盒内边界，不要求平躺。失败轨迹仍保留诊断文件，但不能当成功训练轨迹。

检查图像、数组和可复现物理回放：

```bash
/home/jun/anaconda3/envs/dummyx_vla/bin/python \
  scripts/tools/check_hardstop_dataset.py datasets/agv_dummyx_screwdriver \
  --replay --output outputs/agv_dummyx/dataset_check
```

此采集器不调用 OpenPI 或 PACE。纯 Pi0 本地仿真推理脚本为 `scripts/deploy/deploy_agv_dummyx_pi0.py`；带障碍 PACE 推理入口仍需后续单独实现。

Pi0 的 OpenPI 训练预处理会把相机图像按比例缩放并补边到 224×224，所以采集时保留杆顶相机的 640×360 原图；不要把它强行拉伸成正方形。将成功轨迹转换为 LeRobot（需要 `/home/jun/openpi/.venv`）：

```bash
/home/jun/openpi/.venv/bin/python scripts/convert/convert_agv_dummyx_to_lerobot.py \
  --data-dir datasets/agv_dummyx_screwdriver \
  --output-root datasets/lerobot
```

转换器保留两路图像原尺寸、50 Hz、7 维状态/绝对位置动作和任务文本；先检查全部轨迹，且不会覆盖已有输出。生成 LeRobot 数据只是训练准备的一步：OpenPI 的自定义数据配置还需将 `image`、`wrist_image`、`state`、`actions` 映射到策略输入，并给前 6 维绝对关节动作应用 delta 转换、保留第 7 维夹爪绝对开口。现有 `pi0_libero_low_mem_finetune` 配置的 `repo_id` 和机器人映射仍是 LIBERO，不能直接作为这份数据的训练命令。

## Pi0 推理

训练机需已添加 `pi0_agv_dummyx_lora` 配置，且 10000 步检查点和对应归一化统计量均存在。在本机终端 SSH 登录并转发端口：

```bash
ssh -L 8000:localhost:8000 jun@100.64.142.55
cd ~/VLA/openpi
CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_agv_dummyx_lora --policy.dir=checkpoints/pi0_agv_dummyx_lora/agv_dummyx_lora_run01/10000
```

再在本机启动 MuJoCo 推理并实时查看 Viewer：

```bash
conda activate dummyx_vla
cd /home/jun/dummyx-sim
python scripts/deploy/deploy_agv_dummyx_pi0.py \
  --host 127.0.0.1 --port 8000 \
  --num_episodes 1 --fixed_eval --realtime
```

批量固定起点评估时，可将最后一条命令改为 
python scripts/deploy/deploy_agv_dummyx_pi0.py \
  --host 127.0.0.1 --port 8000 \
  --num_episodes 50 --fixed_eval --headless

交互运行关闭 Viewer 会结束当前进程；结果写入 `outputs/agv_dummyx/pi0_inference/run_时间戳/`。

## 历史工作台方块清扫采集

在仓库根目录采集 50 条随机初态的成功轨迹，同时保存固定相机和腕部相机图像：

```bash
conda activate dummyx_vla
python archive/workbench_cleanup/collect_workbench_cleanup.py \
  --num_episodes 50 --seed 42 --realtime
```

默认追加到 `datasets/agv_dummyx_workbench_cleanup_regrasp/`。不加 `--fixed_eval` 时，海绵初始 XY 分别在 ±12/15 mm 内随机，三块碎屑各自的 XY 在 ±3 mm 内随机；物体朝向不随机。不要加 `--no_images`，否则只保存物理诊断数据。此脚本保存原始仿真示教数据；螺丝刀专用 LeRobot 转换器不接受清扫数据的 `agv_dummyx_workbench_cleanup_arc_v9` schema。
