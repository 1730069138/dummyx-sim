import mujoco
import mujoco.viewer
import numpy as np
import time
import os

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    xml_path = os.path.join(project_root, "models", "dummyx_apf_scene.xml")
    
    # 1. 加载模型
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # 2. 找到 TCP 对应的 Site ID
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
    
    if tcp_id == -1:
        print("❌ 找不到 'tcp_site'！请确认它是否在 dummy_apf.xml 中被定义。")
        return
    else:
        print(f"✅ 成功找到 'tcp_site'，它的 ID 是: {tcp_id}")

    # 3. 启动交互式查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # 步进物理引擎
            mujoco.mj_step(model, data)
            
            # --- 核心可视化逻辑 ---
            # 每次渲染前重置自定义几何体数量
            viewer.user_scn.ngeom = 0 
            
            # 获取当前 TCP 的三维世界坐标
            tcp_pos = data.site_xpos[tcp_id]
            
            # 初始化一个球体 geom 到 user_scn 中
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,  # 形状：球体
                [0.015, 0.015, 0.015],         # 尺寸：半径 1.5cm
                np.zeros(3),                   # 局部坐标位置 (填0即可)
                np.zeros(9),                   # 旋转矩阵 (球体不需要关心)
                [1.0, 0.0, 0.0, 1.0]           # 颜色：RGBA，纯红，不透明
            )
            
            # 将球体的位置绑定为当前 TCP 的位置
            viewer.user_scn.geoms[viewer.user_scn.ngeom].pos = tcp_pos
            viewer.user_scn.ngeom += 1
            # ----------------------
            
            # 同步画面
            viewer.sync()
            time.sleep(0.01) # 稍微控制一下渲染帧率

if __name__ == "__main__":
    main()
