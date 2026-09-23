# APF 的 Grounding DINO 障碍物定位

入口为 `scripts/deploy/deploy_screwdriver_client.py`。默认 `--obstacle_source oracle`
保留原有真值胶囊体；显式选择 `groundingdino` 后，APF 的障碍物几何来自视觉定位。
现有力控、VLA 图像尺寸和动作流程保持原样。

## 第一版范围

- 对象是当前场景的 4 × 4 × 16 cm 方柱，直立、与世界 XY 轴对齐，桌面高度为 0.20 m。
- 每回合复位后自动等待柱体稳定，再从 `rear_cam` 采集 640 × 480 RGB-D。
- DINO 根据 `red pillar.` 检测候选框；从框内深度反投影点云，用顶面连通区域、尺寸和可见高度检查筛选。
- 由足够完整的顶面估计 XY；中心 Z、尺寸和方向来自上述明确的先验。
- 胶囊体半径为方柱截面半对角线加 1 cm 初始余量，约 0.03828 m。真值基线仍使用原有 0.025 m，因此两组几何膨胀不同，不能把表现差异全部归因于检测误差。
- 无检测、没有合格几何或出现多个不同位置的合格目标时，保存证据并以异常终止运行，不执行本回合策略动作，不回退真值。失败回合不会生成正常的实验汇总，需要单独计入感知失败。
- 感知时物理推进暂停，不把 DINO 加进控制循环。成功结果只在本回合内缓存，下一回合重新采集。

这不是动态障碍物跟踪：柱体在回合中移动、倾倒后不会刷新缓存，也没有实现任意形状估计。
几何检查是针对当前先验的筛选，不保证识别所有违反先验的物体。顶面严重遮挡会导致拒绝。
仿真真值仅用于原有碰撞/距离评估和感知检查报告，不用于修正视觉估计。
原有 APF 在抓起螺丝刀后才启用障碍物作用的条件保持不变。

## 先单独检查感知

以下命令使用本机已验证的 `vlsa_env`、源码和权重；无需 OpenPI 服务，也不会打开 Viewer。
`HF_HUB_OFFLINE=1` 使用当前本机已缓存的 BERT，其他机器未缓存时应先准备依赖。

```bash
cd /home/jun/dummyx-sim
MUJOCO_GL=egl HF_HUB_OFFLINE=1 \
  /home/jun/anaconda3/envs/vlsa_env/bin/python \
  scripts/deploy/deploy_screwdriver_client.py \
  --perception_probe \
  --groundingdino_config /home/jun/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --groundingdino_checkpoint /home/jun/GroundingDINO/weights/groundingdino_swint_ogc.pth
```

检查结果保存在 `recordings/groundingdino_probe/<时间>/`：

- `rgb.png`、`detections.png`：原图及候选框；绿色表示几何检查通过，红色表示拒绝。
- `capture.npz`：深度、相机内外参、采集仿真时间。
- `points.npy`：被接受的局部点云（成功时）。
- `perception.json`：文本、阈值、候选框、拒绝原因、胶囊体、耗时。
- `evaluation.json`：与真值的中心误差及方柱八个顶点到胶囊体边界的最小余量。余量为负表示至少一个顶点未被包住；只作诊断。

## 运行一个视觉 APF 回合

先启动现有 OpenPI 策略服务，在可打开 MuJoCo Viewer 的桌面终端执行：

```bash
cd /home/jun/dummyx-sim
HF_HUB_OFFLINE=1 /home/jun/anaconda3/envs/vlsa_env/bin/python \
  scripts/deploy/deploy_screwdriver_client.py \
  --host localhost --port 8000 \
  --case 4 --num_episodes 1 --fixed_eval \
  --obstacle_source groundingdino \
  --groundingdino_config /home/jun/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --groundingdino_checkpoint /home/jun/GroundingDINO/weights/groundingdino_swint_ogc.pth
```

Case 4 是有障碍物、启用 APF、无末端接触保护。需要原有双环保护时改为
`--case 8 --control_mode admittance`（或 `impedance`）。
Viewer 中的黄色障碍胶囊体使用本回合的视觉结果，仍按原有抓起后启用的条件显示。
每回合的感知证据保存到对应运行目录的 `perception_ep_XXX/`。
运行目录、动作 NPZ 和汇总文本均标明障碍物来源。

真值基线使用相同 Case、初始状态和回合数，替换为
`--obstacle_source oracle --settle_before_start`，无需 DINO 配置和权重参数。
默认 oracle 不增加 Grounding DINO 依赖；上述 DINO 权重只在感知模式加载一次。

## 检查代码

```bash
cd /home/jun/dummyx-sim
MUJOCO_GL=egl /home/jun/anaconda3/envs/vlsa_env/bin/python \
  -m unittest discover -s tests -v
```

测试覆盖几何包络、无效/不完整点云、多目标歧义、相机坐标约定、失败记录、
APF 真值基线兼容性及视觉模式禁止回退真值。单元测试与感知 probe 不等于完成 OpenPI 闭环任务验证。
