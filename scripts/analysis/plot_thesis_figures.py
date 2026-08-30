import numpy as np
import matplotlib.pyplot as plt
import os

# ========== Ubuntu 中文乱码修复【核心配置】 ==========
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
# 全局绘图样式
plt.rcParams.update({
    'font.size': 14,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.6
})

def apply_custom_legend(ax):
    """复用防遮挡图例样式，适配6关节曲线"""
    ax.legend(
        loc='upper right',
        ncol=3,
        fontsize=7,
        handlelength=0.8,
        handleheight=0.6,
        columnspacing=0.4,
        borderpad=0.3,
        framealpha=0.8
    )

def plot_episode_data(npz_file_path):
    """
    适配最新空间高度防护算法的实验数据绘图，实现 X/Y/Z 三维受力彻底解耦
    并智能识别 阻抗(Impedance) / 导纳(Admittance) 模式
    """
    data = np.load(npz_file_path)
    steps = data['steps']
    
    # 读取部署脚本已保存的字段
    min_dist = data['min_dist']
    influence = data['influence']
    is_apf_active = data['is_apf_active']
    
    is_contact = data['is_contact']
    f_norm = data['f_z']  # 原综合力模长
    tcp_pos = data['tcp_pos']
    
    # 🚨 提取真实的X, Y, Z独立受力（兼容老数据防报错）
    try:
        f_x = data['f_real_x']
        f_y = data['f_real_y']
        f_z_true = data['f_real_z']
    except KeyError:
        f_z_true = f_norm
        f_x = np.zeros_like(f_norm)
        f_y = np.zeros_like(f_norm)
    
    q_pos = data['q_pos'][:, :6]
    q_vel = data['q_vel'][:, :6]
    q_acc = data['q_acc'][:, :6]
    q_tau = data['q_tau'][:, :6]
    q_pos_vla = data['q_pos_vla'][:, :6]
    q_dot_vla = data['q_dot_vla'][:, :6]
    q_dot = data['q_dot'][:, :6]

    fig = plt.figure(figsize=(22, 16))
    
    # 🚨 核心升级：智能识别当前数据使用的力控模式
    mode_title = "未知力控模式"
    if "impedance" in npz_file_path.lower():
        mode_title = "【二阶阻抗控制 Impedance Control】"
    elif "admittance" in npz_file_path.lower():
        mode_title = "【一阶导纳控制 Admittance Control】"
        
    fig.suptitle(f"螺丝刀操作实验多维物理分析 - {mode_title}\n{os.path.basename(npz_file_path)}", 
                 fontsize=20, fontweight='bold', y=0.95)
                 
    plt.subplots_adjust(hspace=0.4, wspace=0.25, top=0.90)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    joint_labels = [f'Joint {i+1}' for i in range(6)]

    # ================= 第一列：宏微观安全保护状态分析 =================
    
    # 1. 距离障碍物距离
    ax1 = plt.subplot(4, 3, 1)
    ax1.plot(steps, min_dist, color='#1f77b4', linewidth=2, label='TCP到障碍物距离')
    ax1.axhline(y=0.08, color='orange', linestyle='--', alpha=0.8, label='最大动态预警距离 (~0.08m)')
    ax1.axhline(y=0.025, color='r', linestyle='--', alpha=0.8, label='核心安全裕度 (0.025m)')
    ax1.set_title('末端距离障碍物安全余量', fontweight='bold')
    ax1.set_ylabel('Distance (m)')
    ax1.legend(fontsize=8, loc='upper right')

    # 2. 宏观势场(APF)斥力影响权重
    ax2 = plt.subplot(4, 3, 4)
    ax2.plot(steps, influence, color='#ff7f0e', linewidth=2, label='斥力影响权重 (Influence)')
    ax2.fill_between(steps, influence, alpha=0.2, color='#ff7f0e')
    ax2.set_title('APF 斥力场激活权重', fontweight='bold')
    ax2.set_ylabel('Weight (0-1)')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=8, loc='upper right')

    # 3. 避障控制激活状态 (Bool)
    ax3 = plt.subplot(4, 3, 7)
    ax3.plot(steps, is_apf_active, color='#2ca02c', linewidth=2, label='避障滤波器激活 (APF Active)')
    ax3.fill_between(steps, is_apf_active, alpha=0.2, color='#2ca02c')
    ax3.set_title('宏观3D平移避障状态', fontweight='bold')
    ax3.set_ylabel('Status (0/1)')
    ax3.set_ylim(-0.1, 1.1)
    ax3.legend(fontsize=8, loc='upper right')

    # 4. 微观零空间防护激活状态 (Bool)
    ax4 = plt.subplot(4, 3, 10)
    ax4.plot(steps, is_contact, color='#d62728', linewidth=2, label='空间防线激活 (Spatial Wall Active)')
    ax4.fill_between(steps, is_contact, alpha=0.2, color='#d62728')
    ax4.set_title('基于TCP高度的无状态防护区', fontweight='bold') 
    ax4.set_xlabel('控制步长')
    ax4.set_ylabel('Status (0/1)')
    ax4.set_ylim(-0.1, 1.1)
    ax4.legend(fontsize=8, loc='upper right')

    # ================= 第二列：接触力学与运动学演变 =================

    # 5. 桌面接触力 3D 物理分解 
    ax5 = plt.subplot(4, 3, 2)
    ax5.plot(steps, f_x, color='#d62728', linewidth=1.5, label='X轴摩擦力 (F_x)') 
    ax5.plot(steps, f_y, color='#2ca02c', linewidth=1.5, label='Y轴摩擦力 (F_y)') 
    ax5.plot(steps, f_z_true, color='#1f77b4', linewidth=1.5, label='Z轴法向力 (F_z)') 
    ax5.plot(steps, f_norm, color='gray', linewidth=1.5, linestyle=':', label='综合总受力 (Norm)', alpha=0.4)
    ax5.axhline(y=0.5, color='purple', linestyle='--', alpha=0.8, label='防护触发死区 (0.5N)')
    ax5.axhline(y=0.0, color='black', linestyle='-', linewidth=0.8, alpha=0.5) 
    ax5.set_title('末端与桌面 3D 接触力解耦 (X/Y/Z)', fontweight='bold')
    ax5.set_ylabel('Force (N)')
    ax5.legend(fontsize=7, loc='upper right')

    # 6. TCP Z 轴高度 
    ax6 = plt.subplot(4, 3, 5)
    z_heights = tcp_pos[:, 2]
    ax6.plot(steps, z_heights, color='#8c564b', linewidth=2, label='TCP Z坐标')
    ax6.axhline(y=0.235, color='orange', linestyle='--', alpha=0.8, label='警戒空域触发线 (0.235m)') 
    ax6.axhline(y=0.212, color='red', linestyle='-', alpha=0.8, label='意图绝对截断死线 (0.212m)') 
    ax6.set_title('TCP Z轴高度与空间几何防线', fontweight='bold')
    ax6.set_ylabel('Height (m)')
    ax6.legend(fontsize=8, loc='lower right')

    # 7. 6关节力矩
    ax7 = plt.subplot(4, 3, 8)
    for i in range(6):
        ax7.plot(steps, q_tau[:, i], color=colors[i], linewidth=1.5, label=joint_labels[i])
    ax7.set_title('底层6轴执行器输出力矩', fontweight='bold')
    ax7.set_ylabel('Torque (N·m)')
    apply_custom_legend(ax7)

    # 8. 6关节角速度
    ax8 = plt.subplot(4, 3, 11)
    for i in range(6):
        ax8.plot(steps, q_vel[:, i], color=colors[i], linewidth=1.5, label=joint_labels[i])
    ax8.set_title('底层6轴执行器实际角速度', fontweight='bold')
    ax8.set_xlabel('控制步长')
    ax8.set_ylabel('Velocity (rad/s)')
    apply_custom_legend(ax8)

    # ================= 第三列：大模型意图 vs 实际执行 =================

    # 9. 关节角指令分布：VLA原始输出
    ax9 = plt.subplot(4, 3, 3)
    for i in range(6):
        ax9.plot(steps, q_pos_vla[:, i], color=colors[i], linewidth=1.5, linestyle=':', label=f'VLA_{i+1}')
    ax9.set_title('VLA 原始关节位置指令', fontweight='bold')
    ax9.set_ylabel('Position (rad)')
    apply_custom_legend(ax9)

    # 10. 关节角指令分布：底层实际执行
    ax10 = plt.subplot(4, 3, 6)
    for i in range(6):
        ax10.plot(steps, q_pos[:, i], color=colors[i], linewidth=1.5, label=f'Real_{i+1}')
    ax10.set_title('安全清洗后实际执行关节位置', fontweight='bold')
    ax10.set_ylabel('Position (rad)')
    apply_custom_legend(ax10)

    # 11. 预期速度 vs 实际速度 (范数对比)
    ax11 = plt.subplot(4, 3, 9)
    norm_dot_vla = np.linalg.norm(q_dot_vla, axis=1)
    norm_dot_real = np.linalg.norm(q_dot, axis=1)
    ax11.plot(steps, norm_dot_vla, color='#17becf', linewidth=1.5, linestyle='--', label='VLA 预期关节速度')
    ax11.plot(steps, norm_dot_real, color='#e377c2', linewidth=1.5, label='实际下发关节速度')
    ax11.set_title('模型意图与实际运动学响应', fontweight='bold')
    ax11.set_ylabel('Velocity Norm')
    ax11.legend(fontsize=8)

    # 12. 最小二乘干预幅度分析
    ax12 = plt.subplot(4, 3, 12)
    dot_intervention = np.linalg.norm(q_dot - q_dot_vla, axis=1)
    ax12.plot(steps, dot_intervention, color='#d62728', linewidth=2, label='速度/位置解耦干预量')
    ax12.fill_between(steps, dot_intervention, alpha=0.3, color='#d62728')
    ax12.set_title('底层安全系统对 VLA 意图的干预幅度', fontweight='bold')
    ax12.set_xlabel('控制步长')
    ax12.set_ylabel('Intervention Norm')
    ax12.legend(fontsize=8)

    # 保存高清图片
    save_path = npz_file_path.replace('.npz', '_deploy_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 多维分析图表生成成功: {save_path}")

def batch_plot_from_recordings(root_record_dir):
    """批量处理deploy生成的recordings目录下的npz文件"""
    for root, dirs, files in os.walk(root_record_dir):
        for file in files:
            if file.endswith('.npz') and 'data_ep' in file:
                npz_path = os.path.join(root, file)
                try:
                    plot_episode_data(npz_path)
                except Exception as e:
                    print(f"❌ 处理 {npz_path} 失败: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="recordings", help="包含 npz 数据的根目录")
    args = parser.parse_args()
    
    # 自动定位到上一级 dummyx/recordings 目录
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    target_dir = args.dir if os.path.isabs(args.dir) else os.path.join(base_dir, args.dir)
    
    if os.path.exists(target_dir):
        print(f"🔍 正在扫描并处理目录: {target_dir}")
        batch_plot_from_recordings(target_dir)
        print("🎉 所有分析图表生成完毕！可以去对应文件夹查看对比图了！")
    else:
        print(f"❌ 找不到目录: {target_dir}，请确认路径是否正确。")
