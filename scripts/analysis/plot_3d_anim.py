import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os
import argparse

def create_multi_trajectory_gif(npz_file_paths, labels, colors, output_name="multi_comparison_anim.gif"):
    print("正在加载 4 种工况的轨迹数据...")
    tcp_data_list = []
    for path in npz_file_paths:
        if os.path.exists(path):
            data = np.load(path)
            tcp_data_list.append(data['tcp_pos'])
        else:
            print(f"❌ 找不到文件: {path}")
            return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # ==================== 1. 绘制静态背景 ====================
    # 因为 fixed_eval 起点终点一致，取第一条轨迹的起点和终点即可
    base_tcp = tcp_data_list[0]
    ax.scatter(base_tcp[0, 0], base_tcp[0, 1], base_tcp[0, 2], color='green', s=100, label='Start', zorder=5)
    
    # 绘制障碍物圆柱体 (半透明)
    theta = np.linspace(0, 2*np.pi, 50)
    z = np.linspace(0.3, 0.46, 20)
    theta, z = np.meshgrid(theta, z)
    x = 0.11 + 0.025 * np.cos(theta)
    y = -0.1 + 0.025 * np.sin(theta)
    ax.plot_surface(x, y, z, color='gray', alpha=0.3)

    # 绘制半透明桌面提供空间参照
    table_x, table_y = np.meshgrid(np.linspace(0.0, 0.35, 10), np.linspace(-0.2, 0.1, 10))
    table_z = np.full_like(table_x, 0.2) 
    ax.plot_surface(table_x, table_y, table_z, color='saddlebrown', alpha=0.15)

    # ==================== 2. 坐标轴锁定与视角调整 (针对所有轨迹求极值) ====================
    all_x = np.concatenate([traj[:, 0] for traj in tcp_data_list])
    all_y = np.concatenate([traj[:, 1] for traj in tcp_data_list])
    all_z = np.concatenate([traj[:, 2] for traj in tcp_data_list])

    # 💡 强制 1:1:1 真实物理比例 (全局)
    max_range = np.array([all_x.max()-all_x.min(), 
                          all_y.max()-all_y.min(), 
                          all_z.max()-all_z.min()]).max() / 2.0
    mid_x = (all_x.max()+all_x.min()) * 0.5
    mid_y = (all_y.max()+all_y.min()) * 0.5
    mid_z = (all_z.max()+all_z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 💡 保留你的自定义视角参数
    ax.invert_yaxis()
    ax.view_init(elev=15, azim=-90)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m) [Inverted]')
    ax.set_zlabel('Z (m)')
    ax.set_title('4-Condition TCP Trajectory Comparison', fontweight='bold')

    # ==================== 3. 动画生成逻辑 ====================
    # 初始化 4 条空的轨迹线
    lines = []
    for i in range(len(tcp_data_list)):
        line, = ax.plot([], [], [], color=colors[i], linewidth=2.5, label=labels[i])
        lines.append(line)
        
    ax.legend(loc='upper right', fontsize=10)

    # 更新每一帧的函数
    def update(frame):
        for i, line in enumerate(lines):
            traj = tcp_data_list[i]
            # 如果某条轨迹步数较少，让它停留在最后一帧，不会报错
            idx = min(frame, len(traj) - 1)
            
            current_x = traj[:idx+1, 0]
            current_y = traj[:idx+1, 1]
            current_z = traj[:idx+1, 2]
            
            line.set_data(current_x, current_y)
            line.set_3d_properties(current_z)
        return lines

    # 找到四条轨迹中最长的一条
    max_len = max(len(traj) for traj in tcp_data_list)
    step_size = max(1, max_len // 100)
    frames = np.arange(0, max_len, step_size)
    if frames[-1] != max_len - 1:
        frames = np.append(frames, max_len - 1)

    print(f"正在渲染合并动画，共 {len(frames)} 帧，这可能需要几十秒...")
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=False)

    anim.save(output_name, writer='pillow', fps=20)
    print(f"✅ 合并 GIF 动画已成功生成并保存至: {output_name}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser = argparse.ArgumentParser(description="Compare four recorded TCP trajectories.")
    parser.add_argument("npz_files", nargs=4, help="Four NPZ paths, absolute or relative to the project root")
    parser.add_argument("--output", default="outputs/thesis_4case_comparison.gif")
    args = parser.parse_args()

    file_paths = [
        path if os.path.isabs(path) else os.path.join(BASE_DIR, path)
        for path in args.npz_files
    ]
    
    labels = [
        "No Obs + No APF",
        "No Obs + APF",
        "Obs + No APF (Collision)",
        "Obs + APF (Safe)"
    ]
    
    # 配色方案：蓝色、绿色、红色(警告)、紫色(护盾)
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    
    output_path = args.output if os.path.isabs(args.output) else os.path.join(BASE_DIR, args.output)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    create_multi_trajectory_gif(file_paths, labels, colors, output_name=output_path)
