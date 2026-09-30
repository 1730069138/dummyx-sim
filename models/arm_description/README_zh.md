# 全零接触检查版

默认启动全部关节 0，夹爪闭合。原直7角度保存在 config/straight7_pose.json。URDF 零位几何和范围未修改，已重新核验局部挡块。J1-J4 及夹爪在 CAD 接触容差内；J5/J6 仅验证曲面相交边界在 0.0001° 内，不能作为实机精度保证。未重新测量全行程或全臂碰撞。SolidWorks 关节坐标统一为 URDF 全零坐标。旧初始截图仍为直7参考图。

# J5 X 轴与所选边缘相切版

J5 X 正向延长线已在垂直 Z 轴投影平面内与用户所选 XW5 挡块边缘圆弧相切。保持原点、Z 轴、角度符号、硬限位零点数值和当前姿态。SolidWorks 与 URDF 均已更新；保留之前的 J3、J4 修改。此次仅修改坐标参考，不重新测量行程。

# J4 X 轴与挡块上边界相切版

J4 X 正方向已调整为从关节原点出发，在垂直 Z 轴的投影平面内与用户选中的挡块上边缘圆弧相切；不是指向挡块中心或边中点。保持原点、Z 轴、角度符号、硬限位零点数值和当前姿态。SolidWorks 与 URDF 均已更新；J3 保留上一版朝左方向。此次仅修改坐标参考，不重新测量行程。

# J4 X 轴指向挡块版

J4 X 正方向已按用户确认设为从关节原点指向 XW4 挡块中心，并投影至垂直于 Z 轴的平面。保持原点、Z 轴、角度符号、硬限位零点数值和当前姿态。SolidWorks 与 URDF 均已更新；同时保留前一版 J3 X 轴朝左的调整。此次仅修改坐标参考，不重新测量行程。

# J3 X 轴朝左版

J3 X 正方向已按用户确认调整至侧视水平朝左，与所画红线平行。保持原点、Z 轴、角度符号、硬限位零点数值和当前姿态。SolidWorks 与 URDF 均已更新；URDF 保留之前确认的 J3 负角度约定。此次仅修改坐标参考，不重新测量行程。

# 第四关节 X 轴反向版

SolidWorks 已删除 XW2-M2_5X10-1 挡块，并将 J4 的 X、Y 轴同时反向。Z 轴与原点不变，保持右手坐标系。URDF 同步更新了 joint4/link4 坐标定义及下游相对变换，因此各关节角度、硬限位零点、范围、当前外观姿态均不变。以下较早版本的“未修改 SolidWorks”说明已被本次修改取代。

# 移除指定挡块版

已从 URDF 的显示及碰撞模型移除 XW2-M2_5X10-1（part_14），共 35 个零件实例。七个控制关节的零点、范围、初始姿态及其他零件均保持上一版不变；未因移除挡块扩大行程。此版本仅修改 URDF 功能包，不修改 SolidWorks 原装配。历史截图和校准记录仍保留挡块，仅供溯源。未重新计算动力学惯量，未作 ROS 2 实机运行验证。

启动：`ros2 launch arm_description gui.launch.py`。

# 最新确认装配与配色版

滑块和配合圆块已转到用户确认的另一侧，保持原轴向位置；孔按同一朝向与侧视高度一致定位。末端支架保持水平朝下。相机、夹爪及其他机械姿态保留。当前 SolidWorks 已移除 JGJ-5-1，包中同步移除该重复圆块，保留 JGJ-5_2-1，共 36 个零件实例。

J6 新初始角为 135.925567984581°，范围 0～275.46748185157776°；这是新挡块位置下的硬限位角度。配置和 GUI 已同步。旧版本截图及零点记录仅作历史参考。

SolidWorks 与 URDF 按同一 RGB 配色表设置。为确保 RViz 正确显示不同颜色，每个显示 link 仅一个 visual，附加连接均为零位移 fixed；主运动 joint1..joint7 保持原命名。

启动：`ros2 launch arm_description gui.launch.py`。

# 末端支架朝下水平版

JGJ-6-_ASM 支架的指定外侧面已调整为水平朝下。相机、夹爪和其他零件姿态及全部关节参数保持原值；更新支架的 visual/collision 局部安装变换。SolidWorks 装配已同步保存。检查的相邻实体无新增重叠；原模型与 RELAY-POWER 的已有重叠保留，螺孔适配未作实机装配认证。

# 夹爪串色修正版

两侧夹指均为黑色，滑块/支架保留蓝色。为避免部分 RViz 版本把同一 link 的首个 visual 材质用于所有 visual，将两侧滑块和轴承拆到 link16..link19，使用 joint16..joint19 零位移固定连接。原有运动关节、夹爪联动、初始值、网格与空间姿态不变，GUI 仍为七个滑块。此修复通过结构检查，未在本机 ROS 运行时验证。

参考：https://github.com/ros2/rviz/issues/1293

# 实机参考配色版

银灰连杆、深灰电机关节、黑色底座和夹指、蓝色夹爪支架、浅银色相机。按照片做零件级近似配色，不改变网格、关节轴、限位、初始姿态及 GUI。单个网格包含多种实物材质时使用主色，未增加照片中的线缆。

角度映射已按用户确认记录为直接对应；未进行额外硬件操作。

# J3 负方向测试版

本版 joint3 的 X 轴不变，Y/Z 同时反向，仍使用局部 +Z 右手正转；q新=-q旧。范围 [-282.433989°, 0°]，初始值 -89.464792°。硬限位零点与初始机械外观不变。用户已说明“负一百多度”为估计值，并确认当前讨论的关节角定义直接对应真机；不再追加零偏或幅值修正。

本次只更新 URDF 测试包，不修改 SolidWorks 源装配。calibration 中原 CAD 快照为历史数据；以 joint3_direction_revision.json 和 config/initial_pose.json 为本版准据。已补上 RViz 鼠标视角工具。

启动仍用 `ros2 launch arm_description gui.launch.py`；不要把本版和旧版同名包同时放入工作空间。

# 已确认初始姿态与 URDF

本包的初始姿态是用户确认的直“7”形：上臂竖直、前臂水平、指定的 LINK4 板面垂直地面、相机在上、夹爪开口 101 mm。SolidWorks 装配体仍保持此姿态。

## 文件

- `urdf/arm.urdf`：ROS package 网格路径。
- `urdf/arm_portable.urdf`：相对于 URDF 文件的网格路径，适合支持相对路径的独立加载器。不要只复制 URDF 而漏掉 meshes。
- `meshes/`：37 个零件实例的原生 CAD 三角网格，单位米。包含 LINK5 的两个曲面体。
- `config/initial_positions.yaml`、`initial_pose.json`：初始关节值；图片 `initial_pose.png` 为已确认姿态。
- `config/name_mapping.json`：数字名称与功能对应。
- `calibration/`：硬限位校准、原始姿态和限位依据。
- `validation.json`：导出验证与未验证事项。

## 命名

| 名称 | 功能 |
|---|---|
| link0 | 基座，Z 向上 |
| link1..link6 / joint1..joint6 | 机械臂六个关节的子连杆 / 转动关节 |
| link7 / joint7 | 夹爪驱动曲柄 / 驱动转角 |
| link8 / joint8 | 工具安装坐标系 / 固定连接 |
| link9 / joint9 | 夹爪固定基座 / 固定连接 |
| link10 / joint10 | 正侧夹爪滑块 / 从动移动关节 |
| link11 / joint11 | 正侧连杆 / 从动转动关节 |
| link12 / joint12 | 负侧夹爪滑块 / 从动移动关节 |
| link13 / joint13 | 负侧连杆 / 从动转动关节 |
| link14 / joint14 | 夹爪闭合中点 TCP / 固定连接 |
| link15 / joint15 | 相机本体坐标系 / 固定连接（不是已标定光学坐标系） |

## 初始值与零位

| 关节 | CAD 初始角度（度） |
|---|---:|
| joint1 | 180.001 |
| joint2 | 80.001 |
| joint3 | -89.4647921791 |
| joint4 | -111.2863215804 |
| joint5 | 110.4000252485 |
| joint6 | 135.9255679846 |
| joint7 | -114.5518941883 |

URDF 转角单位是弧度，移动关节是米。joint1..joint6 的轴均为局部 +Z，正向遵守右手定则；X 是已确认的零位参考方向。各关节 q=0 仍为已确认的硬限位，夹爪 q=0 为闭合；初始姿态不是零位。URDF 标准不自动应用初始位置，必须加载配置或启动下方程序。多个关节同时为零不意味着该整机姿态无碰撞。

用户已明确确认本版角度定义与真机直接对应，无需额外符号、零偏或传动比修正。此结论依据用户确认，并非本次另行测量；ROS 关节值用弧度，原电机日志用度，接口单位仍须正确换算。J6 已按新挡块安装位置重新确定零点，初始值约 135.926°。

## ROS 2 显示

将 arm_description 文件夹放入已有 ROS 2 工作空间 src 中，在该工作空间执行：

```sh
colcon build --packages-select arm_description
source install/setup.bash
ros2 launch arm_description display.launch.py
```

启动后按初始配置发布 joint_states，并显示模型。不要同时启动另一个 joint_state_publisher。可向 `/arm_command` 发布 `sensor_msgs/msg/JointState`，name 仅接受 joint1..joint7，position 使用弧度；程序会自动补齐夹爪从动关节。该话题只用于此显示程序，绝不发送硬件指令。此 Windows 导出环境未安装 ROS，启动流程尚未做 ROS 运行时验证。

## ROS 2 GUI 滑块测试

启动文件为 `launch/gui.launch.py`，只显示 joint1..joint7 七个滑块。滑块使用 URDF 的硬限位范围，数值单位为弧度。默认从已确认的直“7”姿态开始；GUI 的 Center 按钮恢复此初始姿态（不是硬限位零位）。GUI 滑块存在量化精度，显示值可能与保存值有微小差异。

Ubuntu ROS 2 环境，先 source 对应 ROS 2 发行版，再安装依赖并构建：

```sh
sudo apt install ros-$ROS_DISTRO-joint-state-publisher-gui ros-$ROS_DISTRO-robot-state-publisher ros-$ROS_DISTRO-rviz2
# 将本包放到 ROS 2 工作空间 src/arm_description，在工作空间根目录执行：
colcon build --packages-select arm_description
source install/setup.bash
ros2 launch arm_description gui.launch.py
```

只开滑块、不另开 RViz：

```sh
ros2 launch arm_description gui.launch.py rviz:=false
```

不要与 `display.launch.py` 或其他 joint_states 发布器同时运行。GUI 的输出重映射到 `/arm_command`，夹爪联动节点是唯一的 `/joint_states` 发布器。`config/gui_controls.urdf` 仅用于给 GUI 提供七个滑块及其范围，不是显示或运动学模型；RViz 与 robot_state_publisher 始终使用完整的 `urdf/arm.urdf`。

此工具仅用于离线观察几何关节范围，不连接真机、不做碰撞拦截，也不能证明所有范围组合均无碰撞。不要在连接真机控制器的同一 ROS 图里进行随机滑块测试。J5/J6 曲面挡块范围、全臂自碰撞和线缆范围仍需复核。effort/velocity 的未配置值不用于 GUI 滑块范围计算；未因此获得动力学有效性。

实现依据：[ROS 官方 joint_state_publisher 参数说明](https://github.com/ros/joint_state_publisher/blob/ros2/joint_state_publisher/README.md)。已做静态配置及联动检查，当前 Windows 环境未进行 ROS 2 / Qt / RViz 实机启动验证。

## 夹爪联动细节

joint7 是实际曲柄角，负值张开，范围约 [-114.551894°, 0°]。joint10、joint12 为从动滑块，负值张开，每侧最多 0.0505 m。joint12 与 joint10 可线性 mimic，但滑块与 joint7 之间不是线性关系。joint11、joint13 的连杆姿态也需非线性计算。

`scripts/gripper_kinematics.py` 提供该计算；`initial_pose_publisher.py` 自动使用它。单独读取 URDF 的软件不会自动解算该闭环机构；必须同步提供从动值。为符合 URDF 树结构，连杆的另一端闭环由计算保证，不创建循环 joint。原装配中重叠的 2T-1 零件保留，未擅自删除。

## 验证与限制

- 序列化 URDF 重新计算的初始姿态与 37 个 CAD 零件实例一致，最大变换矩阵系数误差约 5.1e-14；网格路径完整，初始关节值在已记录范围内。
- 导出保留了原生几何，但目前使用的是显示细分网格；不等于精密接触分析网格。
- 碰撞几何暂与视觉网格相同，部分复杂电路板网格较密；LINK5 为曲面，不是封闭实体。不适合作为已验证的动力学接触模型。
- J5/J6 限位由曲面边界检查获得；其余限位也仅来自局部挡块，不包含整机自碰撞、线缆或真实驱动器保护范围。
- 未填入未经确认的质量及惯量，因此无 inertial 标签。URDF 必填的 effort/velocity 暂为 0，明确表示未配置，而非电机额定性能；运动规划和动力学控制前必须填写实测/厂商参数。
- 这是几何/运动学描述包，不是已调试的 Gazebo/MuJoCo/ros2_control 动力学或真机控制包。

包内维护者邮箱仅为 `.invalid` 占位信息；未代表用户作公开授权或对外发布。
