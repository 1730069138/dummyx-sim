import mujoco
import mujoco.viewer
import numpy as np
import os
import time
import matplotlib.pyplot as plt

def damped_pinv(J, rho=0.05):
    return J.T @ np.linalg.inv(J @ J.T + (rho**2) * np.eye(J.shape[0]))

def run_infinite_press_with_derivatives():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    xml_path = os.path.join(BASE_DIR, "models", "dummyx_apf_scene.xml")
    
    print("🔧 [系统初始化] 正在加载物理引擎...")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
    
    mujoco.mj_resetData(model, data)
    hover_q = np.array([0.0, 0.4, 0.5, 0.0, 1.57, 0.0]) 
    data.qpos[:6] = hover_q
    mujoco.mj_forward(model, data)

    current_q = data.qpos[:6].copy()

    # 数据容器
    history_steps = []
    history_z = []
    history_taus = {i: [] for i in range(6)}

    print("\n🚀 [无尽下探 + 导数分析版启动]")
    print("💡 提示：按 [Ctrl + C] 结束，系统将生成【高度-力矩-力矩导数】三联图！")
    print("="*70)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        try:
            while viewer.is_running():
                J_tcp = np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, J_tcp, None, tcp_id)
                J_trans = J_tcp[:3, :6]
                
                tcp_pos = data.site_xpos[tcp_id]
                current_taus = np.abs(data.qfrc_actuator[:6])
                
                history_steps.append(step)
                history_z.append(tcp_pos[2])
                for i in range(6):
                    history_taus[i].append(current_taus[i])
                
                v_down = np.array([0.0, 0.0, -0.05]) 
                q_dot = damped_pinv(J_trans, 0.1) @ v_down
                current_q += q_dot * 0.002 
                data.ctrl[:6] = current_q
                mujoco.mj_step(model, data)
                
                if step % 10 == 0:
                    viewer.sync()
                    time.sleep(0.01)
                step += 1
        except KeyboardInterrupt:
            print("\n\n🛑 停止下压，正在计算导数并绘图...")

    # ==========================================
    # 📈 三维分析引擎：高度 vs 力矩 vs 导数
    # ==========================================
    if len(history_steps) > 5:
        # 计算最大力矩的导数：d(max_tau)/dt
        all_max_taus = np.array([np.max(np.array([history_taus[i][s] for i in range(6)])) for s in range(len(history_steps))])
        tau_derivatives = np.gradient(all_max_taus) # 核心：使用 numpy 梯度计算

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # 1. 高度图
        ax1.plot(history_steps, history_z, 'k', linewidth=2.5, label='TCP Height')
        ax1.set_ylabel('Height (m)')
        ax1.set_title('Contact Event: Analysis of Force Derivatives')
        ax1.grid(True, linestyle='--')

        # 2. 力矩图
        colors = ['r', 'g', 'b', 'orange', 'purple', 'c']
        for i in range(6):
            ax2.plot(history_steps, history_taus[i], color=colors[i], label=f'J{i+1}')
        ax2.set_ylabel('Torque (N·m)')
        ax2.grid(True, linestyle='--')

        # 3. 导数图 (这是你最关心的突变点)
        ax3.plot(history_steps, tau_derivatives, 'm', linewidth=2, label='d(MaxTorque)/dt')
        ax3.set_ylabel('Force Change Rate')
        ax3.set_xlabel('Steps')
        ax3.grid(True, linestyle='--')
        ax3.axhline(0, color='gray', linewidth=1) # 增加零基准线，方便看正负脉冲

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_infinite_press_with_derivatives()