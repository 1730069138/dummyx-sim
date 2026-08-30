import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import os

# 1. 直接加载你的机械臂本体 XML 文件
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model = mujoco.MjModel.from_xml_path(os.path.join(PROJECT_ROOT, "models", "dummy_apf.xml"))
data = mujoco.MjData(model)

# ========================================================
# [新增] 第一阶段：初始 0度 位置的干涉诊断
# ========================================================
data.qpos[:] = 0.0
mujoco.mj_forward(model, data)

if data.ncon > 0:
    print("⚠️ 【警告】扫描终止！检测到机械臂在初始 0 度位置就已经发生穿模干涉。")
    print("以下是发生碰撞的跨级部件，请将下面的代码追加到 xunjian_arm.xml 的 <contact> 标签内：\n")
    
    excludes = set()
    for i in range(data.ncon):
        contact = data.contact[i]
        b1_id = model.geom_bodyid[contact.geom1]
        b2_id = model.geom_bodyid[contact.geom2]
        
        b1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1_id)
        b2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2_id)
        
        # 去重，避免重复输出相同的碰撞对
        if b1_name and b2_name:
            pair = tuple(sorted([b1_name, b2_name]))
            if pair[0] != pair[1]:
                excludes.add(pair)
            
    for b1, b2 in excludes:
        print(f'    <exclude body1="{b1}" body2="{b2}" />')
        
    print("\n💡 【操作提示】请将上面的几行 <exclude ... /> 复制粘贴到 XML 底部，保存后再次运行本脚本！")
    sys.exit(0) # 退出程序，等待用户修改 XML

# ========================================================
# 第二阶段：开始可视化扫描限位
# ========================================================
joint_names = [f"joint{i}" for i in range(1, 7)]

print("========== 初始位置正常，开始可视化本体干涉扫描 ==========")
print("请将注意力集中在弹出的 3D 窗口上...")

# 使用被动查看器启动 3D 窗口
with mujoco.viewer.launch_passive(model, data) as viewer:
    for j_name in joint_names:
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
        if j_id == -1:
            continue
            
        # 临时关闭该关节的默认限制，允许自由旋转
        model.jnt_limited[j_id] = 0
        print(f"\n>>> 正在扫描 {j_name} ...")
        
        # --- 探测正向极限值 ---
        max_limit = 3.14
        data.qpos[:] = 0.0 # 全身回 0 位
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(0.5)
        
        # 慢速正向旋转步进
        for angle in np.linspace(0.0, 3.14, 200): 
            if not viewer.is_running():
                break
                
            data.qpos[model.jnt_qposadr[j_id]] = angle
            mujoco.mj_forward(model, data) 
            viewer.sync()
            time.sleep(0.01)
            
            if data.ncon > 0:
                max_limit = angle - 0.05
                print(f"  [!] 正向发生碰撞！记录最大值: {max_limit:.3f}")
                time.sleep(0.5)
                break
                
        # --- 探测反向极限值 ---
        min_limit = -3.14
        data.qpos[:] = 0.0
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(0.5)
        
        for angle in np.linspace(0.0, -3.14, 200):
            if not viewer.is_running():
                break
                
            data.qpos[model.jnt_qposadr[j_id]] = angle
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)
            
            if data.ncon > 0:
                min_limit = angle + 0.05
                print(f"  [!] 反向发生碰撞！记录最小值: {min_limit:.3f}")
                time.sleep(0.5)
                break

        print(f'')
        print(f'<joint name="{j_name}" type="hinge" range="{min_limit:.3f} {max_limit:.3f}" limited="true" />')

print("\n=============================================================")
print("所有关节扫描完毕！请将上面的 <joint ...> 标签复制回您的 XML 文件中。")
