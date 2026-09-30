# dummyx-sim

整理后的 DummyX MuJoCo 仿真工作空间，目标平台为 Ubuntu 22.04。项目不包含真实机械臂、CAN/USB2CAN 或真实相机控制代码，也不提交本地数据集与实验录像。

## 目录

- `models/`：MuJoCo 机器人模型、场景和网格资源。
- `scripts/collect/`：纯仿真数据采集。
- `scripts/convert/`：数据集转换工具。
- `scripts/deploy/`：仿真策略部署；PACE 使用 `deploy_screwdriver_client.py`，VLSA 使用 `deploy_screwdriver_client_vlsa.py`。
- `scripts/analysis/`：实验结果绘图。
- `scripts/tools/`：模型检查、标定和媒体工具。
- `archive/experiments/`：保留的旧实验版本，不作为主入口维护。
- `archive/broken_scenes/`：依赖缺失模型的旧场景，仅供追溯。

`datasets/` 与 `recordings/` 由程序按需生成，已通过 `.gitignore` 排除。

## Ubuntu 22.04 环境

PACE 与 VLSA 使用两个独立的 Python 3.10.20 环境。两个 requirements 只保留项目直接依赖，避免复制本机环境中无关的 ROS、Azure、Jupyter 和其他开发包。

先在项目同级目录准备两个环境共用的 Grounding DINO 源码：

```bash
cd /path/to/dummyx-sim
git clone https://github.com/IDEA-Research/GroundingDINO.git ../GroundingDINO
git -C ../GroundingDINO checkout 856dde20aee659246248e20734ef9ba5214f5e44
git -C ../GroundingDINO apply "$PWD/patches/groundingdino-torch-2.6.patch"
```

PACE/APF、接触保护与关节力矩观察器使用 `dummyx_vla`：

```bash
conda create -n dummyx_vla python=3.10.20 -y
conda activate dummyx_vla
python -m pip install --upgrade pip
python -m pip install -r requirements-dummyx-vla.txt
python -m pip install -e ../GroundingDINO
MUJOCO_GL=egl python -m unittest discover -s tests -p 'test_groundingdino_obstacle.py' -v
MUJOCO_GL=egl python -m unittest discover -s tests -p 'test_joint_torque_observer.py' -v
```

VLSA/AEGIS、CBF-QP、Grounding DINO 与 GLM 障碍命名使用 `vlsa_env`：

```bash
conda create -n vlsa_env python=3.10.20 -y
conda activate vlsa_env
python -m pip install --upgrade pip
python -m pip install -r requirements-vlsa-env.txt
python -m pip install -e ../GroundingDINO
MUJOCO_GL=egl python -m unittest discover -s tests -p 'test_vlsa_intervention_tuning.py' -v
```

两套环境都只包含 OpenPI WebSocket 客户端；OpenPI 策略服务器及模型检查点需要按其自身仓库另建 Python 3.11+ 环境。Grounding DINO 固定到提交 `856dde20aee659246248e20734ef9ba5214f5e44`，并应用仓库内的 PyTorch 2.6 CUDA API 兼容补丁。Grounding DINO 权重不包含在 Git 中，当前验证权重 `groundingdino_swint_ogc.pth` 的 SHA-256 为 `3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799`。历史环境全量快照保留在 `requirements-dummyx-lock.txt`，仅用于追溯，不推荐作为新设备的首选安装文件。

## 常用命令

```bash
# 采集纯仿真数据
python scripts/collect/data_collector_v2.py

# 检查默认 MuJoCo 场景
python scripts/tools/check_xml.py

# 将图像序列转成视频
python scripts/tools/img2video.py datasets/example/cam_wrist outputs/preview.mp4 --fps 60
```

脚本使用相对于项目根目录的路径，不依赖原电脑上的 `/home/jun/...` 绝对路径。

## 仿真数据采集

AGV 底盘与新模型 `models/arm_description` 的无障碍 MuJoCo 螺丝刀采集使用入口
`scripts/collect/collect_hardstop_screwdriver.py`，启动命令、7 维数据约定及仿真假设见
[新机械臂采集说明](docs/hardstop_arm_collection.md)。下面的命令仍使用原 DummyX 模型。

```bash
cd /path/to/dummyx-sim
conda activate dummyx_vla
python scripts/collect/data_collector_v2.py
```

每次运行会新采集 50 条成功轨迹，并从已有的最大 `ep_N` 编号继续保存，不覆盖旧数据。输出目录为：

```text
datasets/dataset_anomaly_cleanup/ep_N/
```

检查最近采集的数据：

```bash
python scripts/tools/check_dataset.py \
  --dataset-dir datasets/dataset_anomaly_cleanup \
  --output outputs/verify_camera_dual.png
```

## VLSA 推理端

除两个感知探针外，推理脚本需要先启动 OpenPI WebSocket 策略服务器。默认连接 `localhost:8000`；远程服务器通过 `--host` 和 `--port` 指定。

### Baseline

有障碍物：

```bash
python scripts/deploy/deploy_screwdriver_client_vlsa.py \
  --host localhost --port 8000 \
  --mode baseline --fixed_eval \
  --num_episodes 10 --max_steps 600
```

无障碍物：

```bash
python scripts/deploy/deploy_screwdriver_client_vlsa.py \
  --host localhost --port 8000 \
  --mode baseline --fixed_eval --no_obstacle \
  --num_episodes 10 --max_steps 600
```

### Oracle VLSA

该模式使用 MuJoCo 中的真实障碍物几何，适合先验证 CBF-QP 控制逻辑；不需要 GroundingDINO 或 VLM API。

```bash
python scripts/deploy/deploy_screwdriver_client_vlsa.py \
  --host localhost --port 8000 \
  --mode vlsa --obstacle_source oracle --fixed_eval \
  --show_ellipsoids --show_oracle_reference \
  --num_episodes 10 --max_steps 600
```

### Oracle Shadow

计算并记录安全修正，但仍执行 OpenPI 原始动作：

```bash
python scripts/deploy/deploy_screwdriver_client_vlsa.py \
  --host localhost --port 8000 \
  --mode shadow --obstacle_source oracle --fixed_eval \
  --show_ellipsoids \
  --num_episodes 10 --max_steps 600
```

### GroundingDINO 文件

Perception 模式默认从以下位置读取配置和权重：

```text
GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
GroundingDINO/groundingdino_swint_ogc.pth
```

也可通过参数显式指定：

```bash
--groundingdino_config /path/to/GroundingDINO_SwinT_OGC.py \
--groundingdino_checkpoint /path/to/groundingdino_swint_ogc.pth
```

### Perception Probe

探针会在连接 OpenPI 前退出，因此不需要策略服务器：

```bash
python scripts/deploy/deploy_screwdriver_client_vlsa.py \
  --perception_probe --fixed_eval \
  --groundingdino_config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --groundingdino_checkpoint GroundingDINO/groundingdino_swint_ogc.pth \
  --device cuda
```

双视角候选配对探针：

```bash
python scripts/deploy/deploy_screwdriver_client_vlsa.py \
  --candidate_pair_probe --fixed_eval \
  --obstacle_text "gray pillar" --pair_probe_topk 15 \
  --groundingdino_config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --groundingdino_checkpoint GroundingDINO/groundingdino_swint_ogc.pth \
  --device cuda
```

### Perception VLSA：固定文本调试

固定 `--obstacle_text` 会绕过 GLM-4.5V，只使用 GroundingDINO：

```bash
python scripts/deploy/deploy_screwdriver_client_vlsa.py \
  --host localhost --port 8000 \
  --mode vlsa --obstacle_source perception --fixed_eval \
  --obstacle_text "gray pillar" \
  --groundingdino_config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --groundingdino_checkpoint GroundingDINO/groundingdino_swint_ogc.pth \
  --device cuda --show_ellipsoids --save_perception_debug \
  --num_episodes 10 --max_steps 600
```

### 完整 Perception VLSA：GLM-4.5V + GroundingDINO

安装智谱客户端并通过环境变量提供密钥；不要把密钥提交到 Git：

```bash
python -m pip install zai-sdk
export ZHIPU_API_KEY='your-api-key'

python scripts/deploy/deploy_screwdriver_client_vlsa.py \
  --host localhost --port 8000 \
  --mode vlsa --obstacle_source perception --fixed_eval \
  --groundingdino_config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --groundingdino_checkpoint GroundingDINO/groundingdino_swint_ogc.pth \
  --device cuda --show_ellipsoids --save_perception_debug \
  --num_episodes 10 --max_steps 600
```

完整模式不要传 `--obstacle_text`，否则会绕过 GLM-4.5V。无 CUDA 时可用 `--device cpu` 做功能验证，但速度会显著下降。

### Pickup Gate 与干预追踪

```bash
python scripts/deploy/deploy_screwdriver_client_vlsa.py \
  --host localhost --port 8000 \
  --mode vlsa --obstacle_source oracle --fixed_eval \
  --debug_gate_until_pickup --pickup_z 0.235 \
  --trace_window 20 --trace_step 180 \
  --num_episodes 1 --max_steps 600
```

`--debug_gate_until_pickup` 仅用于诊断，不应用于正式实验结果。推理记录统一保存在：

```text
recordings/run_<timestamp>_<mode>_<obstacle_source>_obs<0|1>/
```

推荐验证顺序：Baseline 无障碍 → Baseline 有障碍 → Oracle VLSA → Perception Probe → 固定文本 Perception VLSA → 完整 GLM-4.5V Perception VLSA。
