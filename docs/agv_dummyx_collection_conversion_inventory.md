# AGV 螺丝刀采集脚本使用的文件

采集入口是 `scripts/collect/collect_hardstop_screwdriver.py`。路径均相对仓库根目录；同用途的机械臂 STL 合并说明。

| 文件 | 功能 |
| --- | --- |
| `scripts/collect/collect_hardstop_screwdriver.py` | 执行 MuJoCo 螺丝刀随机抓取采集，保存成功轨迹、图像和元数据。 |
| `scripts/tools/agv_spawn_region.py` | 校验螺丝刀随机位置区域，并在调试 Viewer 中显示该区域。 |
| `models/arm_description/scripts/gripper_kinematics.py` | 计算夹爪联动和开度，由采集脚本导入。 |
| `models/agv_dummyx/scene.xml` | 无障碍采集场景入口；组合 AGV、机械臂和任务物体，并定义圆孔朝下、粗杆在相机下方止住的 D415 支架及俯视相机。 |
| `models/agv_dummyx/factory.xml` | 定义工厂背景、灯光、地面和工作台。 |
| `models/agv_dummyx/task_props.xml` | 定义螺丝刀与收纳箱的形状和碰撞。 |
| `models/agv_dummyx/z_pose.json` | 保存机械臂 Z 型初始及回位姿态。 |
| `models/agv_dummyx/spawn_region.json` | 定义螺丝刀柄中心的随机位置矩形。 |
| `models/agv_dummyx/obstacle.xml` | 提供预留立柱的位置，用于采样避让；默认采集场景不加载立柱。 |
| `models/agv_dummyx/base_link.STL` | AGV 底座网格，复制自原巡检工作区。 |
| `models/agv_dummyx/d415_camera.STL` | 运行时 D415 外形网格，采用米单位；只提供外观，不生成相机图像。 |
| `models/arm_description/meshes/part_*.stl` | 场景引用的 34 个机械臂与夹爪 STL 网格；各文件分别提供对应 link 的外形和碰撞形状，具体文件名见 `scene.xml`。 |
| `models/arm_description/meshes/part_34.msh` | `link31` 的 MuJoCo 网格，由源 STL 转换，场景用于显示和碰撞。 |

以下文件用于来源核对或构建，不由当前采集脚本运行时读取：

| 文件 | 功能 |
| --- | --- |
| `models/arm_description/meshes/part_34.stl` | `part_34.msh` 的原始 STL。 |
| `models/arm_description/urdf/arm_portable.urdf` | 定义原机械臂的 link、joint 和网格引用。 |
| `models/agv_dummyx/cad/D415_Solid.SLDPRT` | D415 的 SolidWorks CAD 源文件，供以后重新导出网格；仿真不直接读取。 |
| `models/agv_dummyx/provenance.json` | 记录 D415 网格来源、校验哈希及支架安装参数，供人工追溯。 |



## 采集命令

```bash
cd /home/jun/dummyx-sim
conda activate dummyx_vla
python scripts/collect/collect_hardstop_screwdriver.py \
  --num_episodes 50 \
  --seed 42 \
  --realtime
```
采集结果默认写入 `datasets/agv_dummyx_screwdriver/`；朝向随机范围 ±30° 写在采集脚本中。

## 转换命令（采集机）

```bash
cd /home/jun/dummyx-sim
conda activate dummyx_vla
python scripts/convert/convert_agv_dummyx_to_lerobot.py \
  --data-dir datasets/agv_dummyx_screwdriver \
  --output-root datasets/lerobot \
  --repo-id local/agv_dummyx_screwdriver
```

转换结果为 `datasets/lerobot/local/agv_dummyx_screwdriver/`。将整个数据集目录复制到训练机的 `<HF_LEROBOT_HOME>/local/agv_dummyx_screwdriver/`；在训练机运行以下命令可查看 `HF_LEROBOT_HOME` 的实际路径。

```bash
cd ~/VLA/openpi
source .venv/bin/activate
uv run python -c "from lerobot.common.constants import HF_LEROBOT_HOME; print(HF_LEROBOT_HOME)"
```

## 归一化与 LoRA 训练命令（训练机）

先在训练机的 `src/openpi/training/config.py` 中添加 `pi0_agv_dummyx_lora` 配置，使其读取 `local/agv_dummyx_screwdriver`，映射 `image`、`wrist_image`、`state`、`actions`，并对前 6 维绝对关节动作应用 delta 转换；现有 `pi0_dummyx_lora` 配置不能直接读取这份数据。完成配置和数据集复制后运行：

```bash
cd ~/VLA/openpi
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py \
  --config-name pi0_agv_dummyx_lora

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_agv_dummyx_lora \
  --exp-name agv_dummyx_lora_run01
```
