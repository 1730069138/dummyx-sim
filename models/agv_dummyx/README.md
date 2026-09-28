# agv_dummyx：巡检工厂抓取演示

`scene.xml` 把巡检工作空间的 AGV 底座与新机械臂组合。本代码分支只保存本次制作的场景和脚本，不提交聊天前已有的机械臂源文件与复制来的 STL 网格。运行前需在本地准备：

- `models/agv_dummyx/base_link.STL`：来自巡检工作空间的 `assets/meshes/base_link.STL`。
- `models/agv_dummyx/d415_camera.STL`：来自巡检工作空间的 `assets/meshes/camera_link.STL`。
- `arm_description_gui_verified_zero/arm_description/meshes/`：原机械臂网格。
- `arm_description_gui_verified_zero/arm_description/scripts/gripper_kinematics.py`：夹爪联动计算。使用构建工具及几何测试时还需原 `urdf/arm_portable.urdf` 和 `config/initial_pose.json`、`config/straight7_pose.json`。

这些文件仍在原本的本地工作区，仓库的代码分支不会提供它们；缺少时 MuJoCo 场景无法加载。

机械臂安装点沿用原 URDF 的固定 `joint0`：
相对 AGV 为 `(0.12866757754516, -0.001, 0.732)` m，安装旋转为零。
AGV 网格最低点原为 `z=-0.06991460919380188` m，整机相应抬高到地面。
新机械臂安装坐标系的世界高度因此为约 `0.801915` m。
未额外加入转接板，也未验证两套模型的真实安装孔位。

使用现有环境打开预览：

```bash
cd /home/jun/dummyx-sim
/home/jun/anaconda3/envs/dummyx_vla/bin/python scripts/tools/preview_agv_dummyx.py
```

默认显示 `z_pose.json` 保存的 Z 型姿态、夹爪张开；`--pose straight7` 可查看原直“7”姿态。鼠标可旋转、平移、缩放。
脚本只作运动学预览，底盘固定，物理计算暂停，不执行采集动作。
工厂照明关闭投影阴影，避免 CAD 细分表面上的自阴影条纹；保留照明和材质。

```bash
# 查看源模型全零、夹爪闭合姿态
python scripts/tools/preview_agv_dummyx.py --pose zero

# 无窗口生成预览图
python scripts/tools/preview_agv_dummyx.py --headless \
  --output outputs/agv_dummyx/preview.png
```

新臂结构保存在本目录的 `scene.xml`，`part_34.msh` 也保存在本目录；其余机械臂网格通过相对路径引用上面的本地源目录。
`provenance.json` 记录复制资源的哈希、源安装点和整机抬高量。依赖文件准备齐全后，运行时不再读取原巡检工作空间。
底盘保持固定，未建立轮子关节或移动底盘动力学；机械臂抓放演示使用 MuJoCo 接触物理。

## 工厂环境与工作台

`factory.xml` 定义灰色厂房墙面、立柱与横梁、顶灯、配电柜、货架、
地面接缝及黄色工位/行驶标线。背景设施是静态视觉布景；工作台与地面带碰撞。

车辆前方为世界坐标 **+X**。工作台尺寸为 **0.70 m（前后）× 1.20 m（左右）**，
桌面离地 **0.80 m**，厚 **0.04 m**，距机械臂安装平台高度约差 **1.9 mm**。
桌子中心为 `(0.7109761143, 0, 0)`，靠车桌沿为 `x=0.3609761143 m`，
比车身网格最前端 `x=0.3459761143 m` 多 **0.015 m**，即 **1.5 cm** 间隙。
桌腿向内缩进，带横撑和脚垫。

## 车载 D415 全局相机

车体大圆孔位于 AGV 局部 `(-0.099992, -0.200)` m，顶板高度在世界系约 **0.602 m**，
孔半径 **0.030 m**。`scene.xml` 在孔位安装半径 13 mm 的固定支架杆，
杆顶和 D415 相机本体中心为世界高度 **1.640 m**。机械臂在向上伸展的检查姿态中，
可视网格最高点的保守界为约 **1.539 m**；相机杆顶高约 **10 cm**。
相机外形复制自原巡检工作空间的 `camera_link.STL`，在本目录保存为 `d415_camera.STL`；
源文件不改。杆、法兰和相机本体均有碰撞近似，底座固定。

`overview` 和 `d415_rgb` 是同一个杆顶 RGB 光学视角，朝向工作台；
采用 D415 RGB 的约 **69.4° 水平、42.5° 垂直**视场。
在 640×360 图像中，工作台桌面四角都在视场内。
这里的 D415 是相机外形与理想针孔 RGB 渲染，未模拟真实深度、畸变或噪声。
采集器现在把 `overview` 原始 640×360 图像写入 `cam_fixed/`，腕部相机仍为 256×256；
数据版本为 `agv_dummyx_screwdriver_v3`。旧 `scene_pre_d415.xml` 只用于旧数据回放。
预览图见 `outputs/agv_dummyx/d415_mount_overview.png` 和 `d415_camera_initial_raw.png`。

无窗口加载及渲染已检查，直“7”与全零两种姿态均无工作台碰撞接触。
下面的 demo 检查固定轨迹中的接触及放置结果；随机采集的验证和限制见 `../../docs/hardstop_arm_collection.md`。
预览图与检查记录见 `outputs/agv_dummyx/`。

## Z 型起始姿态与物品

关节 2 为 **0 rad**；关节 3 为 **-2.957736043 rad（约 -169.47°）**。
它从直“7”姿态的关节 3 减去原关节 2 角度，补偿肩部变化，使后段继续近似平行桌面。
关节 1 设为 **175°（3.054326191 rad）**，补偿源模型关节 2 安装变换中的 5° 水平旋转，
使末端前向对齐车头 +X；这是朝前中立姿态，不修改源 URDF 硬限位零位。
其余关节与原直“7”姿态相同。预览、demo 起始及回位均读取此配置。此配置另存于 `z_pose.json`，源 URDF 和源姿态文件保持原样。

从车后朝车头看，+Y 为左、-Y 为右。`task_props.xml` 定义：

- 左侧收纳箱：底板中心 `(0.471, 0.253, 0.80)` m，主体外形约 **22 × 23 × 5.3 cm**，
  内部净开口约 **20 × 21 cm**，底厚 **8 mm**。静态固定，底板与四边分别碰撞，箱内可落物。
- 右侧一字螺丝刀：柄中心初始 `(0.415, -0.16, 0.8135)` m，长约 **19 cm**，
  刀杆和刀头朝车头前方 **+X**，橙色倒角柄 **9 × 3 × 2.6 cm**，黑色防滑装饰、金属杆与扁平刀头。
  质量设为 **0.10 kg**，为仿真假设；自由物体，由重力和接触运动。
  主要可视网格与碰撞网格一致，防滑细节只参与显示。

物品处于机械臂的可达范围内。螺丝刀向右侧移开并旋转 90°，与立柱在 Y 方向的初始表面间隙约 12.4 cm。

## 抓取到收纳箱 demo

```bash
cd /home/jun/dummyx-sim
/home/jun/anaconda3/envs/dummyx_vla/bin/python scripts/tools/demo_agv_dummyx.py
```

打开 Viewer，以接近实时速度完成一轮演示后退出，同时保存视频、阶段截图和轨迹。
默认输出路径为 `outputs/agv_dummyx/demo_日期_时间/`。
无需 OpenPI、GroundingDINO 或 ROS。

```bash
# 无窗口运行并生成视频
/home/jun/anaconda3/envs/dummyx_vla/bin/python scripts/tools/demo_agv_dummyx.py --headless

# 仅验证物理，不渲染视频
/home/jun/anaconda3/envs/dummyx_vla/bin/python scripts/tools/demo_agv_dummyx.py --headless --no-video
```

动作顺序：Z 型起始 → 到达螺丝刀上方 → 预收爪 → 下降 → 夹紧 → 抬升 →
向左搬运 → 箱内上方 → 松爪 → 撤回 → 回到 Z 型 → 检查稳定留存。
仅 reset 设置物理状态，执行过程中使用位置伺服与真实接触，不绑定或瞬移螺丝刀。
demo 与采集器复用机械臂 IK 和夹爪控制；采集器现在默认使用本 AGV 场景。

通过条件包括：双指接触、至少抬升 5 cm、整个搬运阶段每个 20 ms 控制帧都保持双指接触、
松爪后全部螺丝刀碰撞顶点位于箱内、物体线速度低于 0.03 m/s、夹爪张开且不再接触物体，
连续满足最后这些留存条件至少 1 s。发现机械臂与桌子/箱子/底座或非相邻自身穿透超过 1 mm 会中止。
这是固定布局的脚本 demo，非 VLA 策略测试或真实硬件验证；机器人惯性/伺服沿用现有仿真假设。

当前立柱居中、箱子从左侧车沿向内移回 5 cm 的布局已验证输出：`outputs/agv_dummyx/demo_higher_lift_verified/`，包含 `demo.mp4`、`result.json`、
`trace.npz` 和各阶段 PNG。`trace.npz` 的状态在步进后记录，动作是该步实际执行的目标；
此文件用于演示审计，不等同于采集器的训练数据格式。

## 可选障碍物：仅在测试场景加载

默认 `scene.xml` **不包含障碍物的 body、geom 或 freejoint**，所以既不会出现在 RGB/深度图中，
也不会存在隐藏碰撞。当前 demo 和采集器均加载此无障碍场景。

`scene_obstacle.xml` 包含同一个 `scene.xml`，额外加载 `obstacle.xml`。
机器人、Z 型姿态、桌子、螺丝刀和小收纳箱保持一致，只增加红色立柱。
立柱尺寸 **4 × 4 × 16 cm**，中心 `(0.46, -0.001, 0.88)` m，底部在桌面 `z=0.80 m`。
它位于右侧螺丝刀与左侧箱子之间，具有碰撞和自由关节，可以被碰动或撞倒；质量设为 0.20 kg。
初始状态与螺丝刀、箱子、机械臂无接触，仅接触桌面。

```bash
# 默认：无障碍物，查看采集时使用的布局
/home/jun/anaconda3/envs/dummyx_vla/bin/python scripts/tools/preview_agv_dummyx.py

# 可选：显示带障碍物的测试场景（静态预览，不执行 PACE）
/home/jun/anaconda3/envs/dummyx_vla/bin/python scripts/tools/preview_agv_dummyx.py --obstacle

# 保存带障碍物的预览图，默认写入 outputs/agv_dummyx/preview_obstacle.png
/home/jun/anaconda3/envs/dummyx_vla/bin/python scripts/tools/preview_agv_dummyx.py --obstacle --headless
```

Python 加载预览可用 `load_model(obstacle=True)`；正式仿真可直接加载
`models/agv_dummyx/scene_obstacle.xml`。增加自由立柱后 qpos/qvel 维度会变化，后续控制脚本应按
关节名称查地址，不应假定固定索引。保留原场景的 `dynamic_pillar`、`pillar_geom`、`pillar_joint` 命名。

几何回放确认：无障碍 demo 的直达搬运轨迹会与立柱相交，第一处发生在 `transfer` 阶段。
这只用于检验障碍物位置，不代表一次物理障碍物试验或 PACE 性能结果。
回放记录见 `outputs/agv_dummyx/obstacle_route_check.json`。

**PACE 推理脚本暂不编写**。按当前约定，后续为 agv_dummyx 单独创建新推理入口，旧推理脚本保持原样。
届时需要接入新机械臂、相机和 0.80 m 桌高，不能直接照搬旧视觉定位中的 0.20 m 桌高假设。
采集入口已改为新 AGV 场景；原 DummyX 数据采集入口仍独立。

### 当前对齐关系

- 立柱中心 Y = **-0.001 m**，与机械臂安装坐标系的 Y 相同，沿 +X 位于机械臂正前方。
- 收纳箱中心 Y = **+0.253 m**，从 AGV 左侧外边缘向机械臂方向移回 **5 cm**；箱体仍完全落在桌面上。
- 螺丝刀的位置和朝前方向保持不变。
- 无障碍 demo 保持箱子位置，采用约 **25°** 的放置倾角，将螺丝刀放到箱内靠机械臂的一侧，
  放置目标按 `drop_box` 中心计算。由于 TCP 抓在手柄附近、相对整把工具的几何中心在 X 方向偏移约 52 mm，当前 TCP 目标为 `(0.419, 0.253, 0.872)` m，使释放后的完整螺丝刀尽量位于箱体中心。
- 正交俯视图见 `outputs/agv_dummyx/layout_top.png`。图中车头朝上，左侧就是 +Y。

## 螺丝刀位置区域（仅调试可见）

`spawn_region.json` 定义螺丝刀**柄中心**的位置矩形，目前 X 为 `[0.415, 0.515]` m、
Y 为 `[-0.28, -0.08]` m，即 **10 × 20 cm**。矩形靠障碍物的一侧与预留立柱表面间距 **5.9 cm**。
`obstacle_clearance_m=0.03` 要求矩形边界与预留立柱至少间隔 3 cm；加载时检查矩形完全位于桌面内。
即使当前场景没有立柱，也使用 `obstacle.xml` 的位置预留空间。

demo 保持固定螺丝刀位置；采集器使用此矩形随机采样柄中心，并以刀头朝前的 90° 为中心，在 ±30° 内均匀采样朝向。
采集器在执行前检查整把工具是否越过桌边或预留障碍空间，随后通过 IK、接触及留盒判据筛选。成功分布并不均匀；详见采集说明。
调整上述 JSON 的 X/Y 上下界后，重新打开预览即可看到新范围。

```bash
# 默认在调试预览中显示青色矩形，同时显示预留的障碍物
/home/jun/anaconda3/envs/dummyx_vla/bin/python scripts/tools/preview_agv_dummyx.py --obstacle

# 隐藏矩形
/home/jun/anaconda3/envs/dummyx_vla/bin/python scripts/tools/preview_agv_dummyx.py --obstacle --hide-spawn-region

# 在动态 demo 的实时调试窗口中显示矩形；保存的视频仍不含矩形
/home/jun/anaconda3/envs/dummyx_vla/bin/python scripts/tools/demo_agv_dummyx.py --show-spawn-region
```

矩形由 `scripts/tools/agv_spawn_region.py` 加到 Viewer 的 `user_scn`，不写入模型 XML、
不添加物理 geom/site、不参与碰撞。Viewer 选择模型内的固定相机视角时也自动隐藏，回到自由视角恢复。
独立观察相机使用自己的 Renderer，因此 RGB、深度和分割图均不包含此区域；不要把 Viewer 截图作为采集图。
`preview_agv_dummyx.py --headless` 是专门的调试截图，会按预览设置显示矩形；
实际 demo 视频及独立模型相机输出不加这层显示。

已用 `overview`、`wrist_cam` 两个相机对比 RGB/深度/分割输出，调试叠加前后逐像素一致。
调试示意图：`outputs/agv_dummyx/spawn_region_debug.png`，俯视图：`spawn_region_top.png`。

### 抬升与搬运高度

当前 `CARRY_HEIGHT = 0.915 m`，比原来的 0.885 m 抬高 **3 cm**。
`lift` 在升高时平滑过渡到 25° 放置倾角，`transfer`、`above_bin` 保持 0.915 m 的 TCP 目标高度；
到箱内放置点上方后再进入 `lower` 降低松爪。预抓取高度保持原值。
盒沿顶面为 **0.853 m**，预留立柱顶面为 **0.960 m**。
已检查完整螺丝刀可视/碰撞几何在抬升完成及所有横移控制帧中的高度，结果见
`outputs/agv_dummyx/demo_higher_lift_verified/height_check.json`。
本高度约束针对搬运的螺丝刀和 TCP，不要求整条机械臂低于立柱。

## AGV 数据采集（2026-09-28）

`../../scripts/collect/collect_hardstop_screwdriver.py` 现在默认使用这个无障碍场景。从调试矩形采样柄中心位置，在 60°～120° 采样朝向，记录成功与拒绝原因。命令、数据格式与回放检查见 `../../docs/hardstop_arm_collection.md`。
