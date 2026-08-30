import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import cv2

# ==========================================
# 🧮 逆运动学与空间几何底层支持
# ==========================================
def get_site_jacobian_6d(model, data, site_id):
    """获取包含平移和旋转的 6D 雅成比矩阵"""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return np.vstack((jacp, jacr))[:, :6]

def damped_pinv(J, rho=0.10):
    return J.T @ np.linalg.inv(J @ J.T + rho**2 * np.eye(J.shape[0]))

def get_rotation_matrix_from_vectors(vec1, vec2):
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-6: return np.eye(3)
    if c < -1 + 1e-6:
        axis = np.cross(a, np.array([1, 0, 0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0, 1, 0]))
        axis = axis / np.linalg.norm(axis)
        return np.eye(3) + 2 * np.outer(axis, axis)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

def get_orientation_error(target_mat, current_mat):
    err_mat = target_mat @ current_mat.T
    angle = np.arccos(np.clip((np.trace(err_mat) - 1) / 2, -1.0, 1.0))
    if angle < 1e-6: return np.zeros(3)
    rx = err_mat[2, 1] - err_mat[1, 2]
    ry = err_mat[0, 2] - err_mat[2, 0]
    rz = err_mat[1, 0] - err_mat[0, 1]
    axis = np.array([rx, ry, rz])
    norm = np.linalg.norm(axis)
    if norm < 1e-6: return np.zeros(3)
    return (axis / norm) * angle

def compute_6d_twist(current_pos, target_pos, current_xmat, target_xmat, speed_lin):
    v_lin = target_pos - current_pos
    dist_lin = np.linalg.norm(v_lin)
    if dist_lin > 1e-4:
        v_lin = (v_lin / dist_lin) * min(dist_lin * 10.0, speed_lin)
    
    w_err = get_orientation_error(target_xmat, current_xmat)
    dist_ang = np.linalg.norm(w_err)
    speed_ang = speed_lin * 4.0 
    
    if dist_ang > 1e-4:
        w_ang = (w_err / dist_ang) * min(dist_ang * 10.0, speed_ang)
    else:
        w_ang = np.zeros(3)
        
    return np.concatenate([v_lin, w_ang])

def compute_3d_velocity(current_pos, target_pos, speed_lin):
    v_lin = target_pos - current_pos
    dist_lin = np.linalg.norm(v_lin)
    if dist_lin > 1e-4:
        return (v_lin / dist_lin) * min(dist_lin * 10.0, speed_lin)
    return np.zeros(3)

# ==========================================
# 🌍 场景复位与多目标智能散布
# ==========================================
def reset_scene(model, data):
    spawn_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "chip_tray")
    if spawn_id != -1:
        spawn_pos = model.geom_pos[spawn_id]
        spawn_size = model.geom_size[spawn_id]
        x_min, x_max = spawn_pos[0] - spawn_size[0] + 0.03, spawn_pos[0] + spawn_size[0] - 0.03
        y_min, y_max = spawn_pos[1] - spawn_size[1] + 0.03, spawn_pos[1] + spawn_size[1] - 0.03
    else:
        x_min, x_max = 0.15, 0.35
        y_min, y_max = -0.05, 0.15

    screw_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "fj_screwdriver")
    sponge_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "fj_sponge")
    screw_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "real_screwdriver")
    sponge_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "yellow_sponge")

    scenario = np.random.randint(0, 3) 
    active_targets = []

    def place_obj(jnt_id, body_id, is_active, fixed_x=None):
        if jnt_id == -1: return None
        adr = model.jnt_qposadr[jnt_id]
        if is_active:
            x = fixed_x if fixed_x is not None else np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            yaw = np.random.uniform(-np.pi/3, np.pi/3)
            data.qpos[adr:adr+3] = [x, y, 0.25] 
            data.qpos[adr+3:adr+7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
            return (body_id, yaw)
        else:
            data.qpos[adr:adr+3] = [10.0 + np.random.rand(), 10.0, -10.0]
            data.qpos[adr+3:adr+7] = [1, 0, 0, 0]
            return None

    if scenario == 0:
        info = place_obj(screw_jnt, screw_body, True)
        place_obj(sponge_jnt, sponge_body, False)
        if info: active_targets.append(info)
    elif scenario == 1:
        place_obj(screw_jnt, "real_screwdriver", False)
        info = place_obj(sponge_jnt, sponge_body, True)
        if info: active_targets.append(info)
    else:
        mid_x = (x_min + x_max) / 2
        x1 = np.random.uniform(x_min, mid_x - 0.02)
        x2 = np.random.uniform(mid_x + 0.02, x_max)
        if np.random.rand() > 0.5: x1, x2 = x2, x1
        
        info1 = place_obj(screw_jnt, screw_body, True, fixed_x=x1)
        info2 = place_obj(sponge_jnt, sponge_body, True, fixed_x=x2)
        
        if np.random.rand() > 0.5:
            if info1: active_targets.append(info1)
            if info2: active_targets.append(info2)
        else:
            if info2: active_targets.append(info2)
            if info1: active_targets.append(info1)

    obs_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pillar_joint")
    if obs_jnt_id != -1:
        obs_adr = model.jnt_qposadr[obs_jnt_id]
        data.qpos[obs_adr : obs_adr+3] = [10.0, 10.0, -10.0]
        data.qpos[obs_adr+3:obs_adr+7] = [1, 0, 0, 0]

    data.qpos[:8] = 0
    data.qvel[:] = 0
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)
    
    for _ in range(100):
        mujoco.mj_step(model, data)
        
    return active_targets

# ==========================================
# 🚀 主程序
# ==========================================
def main():
    # 👇 修改位置 1：动态获取项目根目录 dummyx/ 的绝对路径
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 👇 修改位置 2：基于根目录读取 xml
    xml_path = os.path.join(BASE_DIR, "models", "dummyx_apf_scene.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=256, width=256)
    
    # 👇 修改位置 3：统一将数据集存放在根目录下的 datasets 文件夹内
    save_dir = os.path.join(BASE_DIR, "datasets", "dataset_anomaly_cleanup")
    os.makedirs(save_dir, exist_ok=True)

    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
    spawn_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "chip_tray")

    GRIPPER_OPEN = 0.04   
    GRIPPER_CLOSE = 0.0    
    TARGET_SUCCESS_EPISODES = 50 
    
    fps = 50
    sim_steps_per_record = int((1.0 / fps) / model.opt.timestep)
    
    language_instructions = [
        "Clear all foreign objects from the chip tray into the box.",
        "Handle abnormal situation: pick up the irrelevant items and drop them into the bin.",
        "Clean up the workspace by placing any anomalies into the storage box.",
        "Remove all foreign tools and sponges to secure the chip production line."
    ]
    
    success_count = 0
    if os.path.exists(save_dir):
        existing_eps = []
        for d in os.listdir(save_dir):
            if d.startswith("ep_") and os.path.isdir(os.path.join(save_dir, d)):
                try:
                    num = int(d.split("_")[1])
                    existing_eps.append(num)
                except ValueError:
                    pass
        if existing_eps:
            success_count = max(existing_eps) + 1
            
    start_count = success_count 
    
    print(f"🚀 启动 [多目标连环清理 + 动态材质适配] 全新状态机！")
    if success_count > 0:
        print(f"📂 检测到历史数据，本次将从 ep_{success_count} 开始续录。")
    print(f"🎯 本次运行目标采集新数据: {TARGET_SUCCESS_EPISODES} 条...")

    total_attempts = 0
    SAFE_Z = 0.35  

    def get_drop_point():
        box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drop_box")
        if box_id != -1:
            dp = data.xpos[box_id].copy()
            dp[0] += np.random.uniform(-0.03, 0.03)
            dp[1] += np.random.uniform(-0.03, 0.03)
            dp[2] += 0.02
            return dp
        return np.array([0.0, -0.1, 0.22])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and (success_count - start_count) < TARGET_SUCCESS_EPISODES:
            total_attempts += 1
            active_targets = reset_scene(model, data)
            if not active_targets:
                continue
            
            home_point = data.site_xpos[tcp_id].copy()
            home_qpos = data.qpos[:6].copy()
            home_tcp_xmat = data.site_xmat[tcp_id].reshape(3, 3).copy()
            
            v_current_z = home_tcp_xmat[:, 2]
            v_down = np.array([0.0, 0.0, -1.0])
            R_pitch_down = get_rotation_matrix_from_vectors(v_current_z, v_down)
            vertical_xmat = R_pitch_down @ home_tcp_xmat
            
            def update_target_orientation(yaw):
                R_z = np.array([
                    [np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw),  np.cos(yaw), 0],
                    [0, 0, 1]
                ])
                return R_z @ vertical_xmat

            target_idx = 0
            current_target_id, current_target_yaw = active_targets[target_idx]
            target_tcp_xmat = update_target_orientation(current_target_yaw)
            
            phase = "HOVER_A"
            wait_steps = 0
            phase_steps = 0
            success = False
            
            ep_images_fixed, ep_images_wrist = [], []
            ep_qpos, ep_actions = [], []
            step_counter = 0
            
            intentional_miss = (np.random.rand() < 0.10)
            has_retried = False
            drop_point_b = get_drop_point()
            
            print(f"\n--- 尝试第 {total_attempts} 轮 (已集: {success_count - start_count}/{TARGET_SUCCESS_EPISODES}) ---")
            print(f"📦 本轮发现 {len(active_targets)} 个异物！开始执行连环清理策略...")

            while viewer.is_running():
                phase_steps += 1
                
                if phase_steps > 4000:
                    print(f"⚠️ 动作在 [{phase}] 阶段死锁，自动作废重开！")
                    success = False
                    break

                target_pos = data.xpos[current_target_id].copy()
                tcp_pos = data.site_xpos[tcp_id].copy()
                current_tcp_xmat = data.site_xmat[tcp_id].reshape(3, 3).copy()
                
                if target_pos[2] < 0.19:
                    print("⚠️ 物体掉落桌下，作废重开！")
                    success = False
                    break 
                
                miss_offset = np.array([0.0, 0.08, 0.0]) if (intentional_miss and not has_retried) else np.zeros(3)
                
                # 👇 修改位置：由于方块体形变小，将下潜偏移量加深至 -0.010，增加抓取稳定性
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, current_target_id)
                if body_name and "sponge" in body_name:
                    z_offset = -0.010 # 适配变小后的海绵，加深向下吃饱方块的深度
                else:
                    z_offset = -0.015 
                
                hover_point_a = np.array([target_pos[0], target_pos[1], SAFE_Z]) + miss_offset
                grasp_point_a = target_pos + np.array([0.0, 0.0, z_offset]) + miss_offset
                release_point_b = drop_point_b + np.array([0.0, 0.0, 0.03])

                q_dot_final = np.zeros(6)
                gripper_target = GRIPPER_OPEN
                
                J_6d = get_site_jacobian_6d(model, data, tcp_id)
                J_3d = J_6d[:3, :] 

                if phase == "HOVER_A":
                    dist = np.linalg.norm(hover_point_a - tcp_pos)
                    w_err = get_orientation_error(target_tcp_xmat, current_tcp_xmat)
                    dist_q = np.linalg.norm(w_err)
                    
                    if (dist < 0.05 and dist_q < 0.35) or (phase_steps > 50 and dist < 0.08):
                        phase = "DESCEND_A"
                        phase_steps = 0 
                    else:
                        twist = compute_6d_twist(tcp_pos, hover_point_a, current_tcp_xmat, target_tcp_xmat, 0.60)
                        q_dot_final = damped_pinv(J_6d) @ twist

                elif phase == "DESCEND_A":
                    dist = np.linalg.norm(grasp_point_a - tcp_pos)
                    if dist < 0.015 or (phase_steps > 75 and dist < 0.03):
                        phase = "GRASP"
                        phase_steps = 0
                    else:
                        twist = compute_6d_twist(tcp_pos, grasp_point_a, current_tcp_xmat, target_tcp_xmat, 0.30)
                        q_dot_final = damped_pinv(J_6d) @ twist

                elif phase == "GRASP":
                    q_dot_final = np.zeros(6) 
                    gripper_target = GRIPPER_CLOSE 
                    if step_counter % sim_steps_per_record == 0:
                        wait_steps += 1
                    if wait_steps >= int(0.4 * fps):
                        print(f"👉 1. [{body_name}] 抓取成功！解除手腕封印，画弧搬运...")
                        phase = "MOVE_ARC"
                        phase_steps = 0
                        wait_steps = 0 
                        
                        arc_start_pos = tcp_pos.copy()
                        arc_end_pos = release_point_b.copy()
                        vec_xy = arc_end_pos[:2] - arc_start_pos[:2]
                        L_sq = np.dot(vec_xy, vec_xy)
                        H_peak = 0.18 

                elif phase == "MOVE_ARC":
                    dist_to_obj = np.linalg.norm(tcp_pos - data.xpos[current_target_id])
                    if dist_to_obj > 0.06:
                        print("⚠️ 脱手！触发纠错重试...")
                        phase = "RECOVER_OPEN"
                        phase_steps = 0
                        wait_steps = 0
                        continue

                    gripper_target = GRIPPER_CLOSE
                    
                    w = tcp_pos[:2] - arc_start_pos[:2]
                    p = np.dot(w, vec_xy) / L_sq if L_sq > 1e-6 else 1.0
                    p = np.clip(p, 0.0, 1.0)
                    
                    if p >= 0.98 and np.linalg.norm(arc_end_pos - tcp_pos) < 0.04:
                        phase = "RELEASE"
                        phase_steps = 0
                        print("👉 2. 抵达箱子底部，松开夹爪！")
                    else:
                        p_target = np.clip(p + 0.10, 0.0, 1.0)
                        target_xy = arc_start_pos[:2] + vec_xy * p_target
                        target_z = arc_start_pos[2] + (arc_end_pos[2] - arc_start_pos[2]) * p_target + H_peak * np.sin(p_target * np.pi)
                        virtual_rabbit = np.array([target_xy[0], target_xy[1], target_z])
                        
                        v_lin = compute_3d_velocity(tcp_pos, virtual_rabbit, 0.60)
                        q_dot_final = damped_pinv(J_3d) @ v_lin

                elif phase == "RECOVER_OPEN":
                    q_dot_final = np.zeros(6)
                    gripper_target = GRIPPER_OPEN
                    if step_counter % sim_steps_per_record == 0:
                        wait_steps += 1
                    if wait_steps >= int(0.3 * fps): 
                        has_retried = True 
                        phase = "HOVER_A"
                        phase_steps = 0
                        wait_steps = 0

                elif phase == "RELEASE":
                    q_dot_final = np.zeros(6) 
                    gripper_target = GRIPPER_OPEN 
                    if step_counter % sim_steps_per_record == 0:
                        wait_steps += 1
                    if wait_steps >= int(0.3 * fps): 
                        print("👉 3. 落袋完成！再次沿抛物线撤退...")
                        phase = "RETURN_ARC"
                        phase_steps = 0
                        wait_steps = 0
                        
                        ret_start_pos = tcp_pos.copy()
                        ret_end_pos = home_point.copy()
                        ret_vec_xy = ret_end_pos[:2] - ret_start_pos[:2]
                        ret_L_sq = np.dot(ret_vec_xy, ret_vec_xy)
                        ret_H_peak = 0.15 
                        
                elif phase == "RETURN_ARC":
                    gripper_target = GRIPPER_OPEN
                    
                    w = tcp_pos[:2] - ret_start_pos[:2]
                    p = np.dot(w, ret_vec_xy) / ret_L_sq if ret_L_sq > 1e-6 else 1.0
                    p = np.clip(p, 0.0, 1.0)
                    
                    if p >= 0.98 and np.linalg.norm(ret_end_pos - tcp_pos) < 0.05:
                        target_idx += 1
                        if target_idx < len(active_targets):
                            print(f"👉 4. 锁定第 {target_idx+1}/{len(active_targets)} 个异物，发起连环抓取！")
                            current_target_id, current_target_yaw = active_targets[target_idx]
                            target_tcp_xmat = update_target_orientation(current_target_yaw)
                            drop_point_b = get_drop_point() 
                            intentional_miss = (np.random.rand() < 0.10)
                            has_retried = False
                            phase = "HOVER_A"
                            phase_steps = 0
                        else:
                            phase = "RETURN_JOINT"
                            phase_steps = 0
                            print("👉 4. 桌面异物已全部清理完毕，开始解卷姿态...")
                    else:
                        p_target = np.clip(p + 0.10, 0.0, 1.0)
                        target_xy = ret_start_pos[:2] + ret_vec_xy * p_target
                        target_z = ret_start_pos[2] + (ret_end_pos[2] - ret_start_pos[2]) * p_target + ret_H_peak * np.sin(p_target * np.pi)
                        
                        virtual_rabbit = np.array([target_xy[0], target_xy[1], target_z])
                        v_lin = compute_3d_velocity(tcp_pos, virtual_rabbit, 0.70) 
                        q_dot_final = damped_pinv(J_3d) @ v_lin

                elif phase == "RETURN_JOINT":
                    gripper_target = GRIPPER_OPEN
                    q_err = home_qpos - data.qpos[:6]
                    dist_q = np.linalg.norm(q_err)
                    
                    if dist_q < 0.10 or (phase_steps > 100 and dist_q < 0.25):
                        phase = "WAIT_HOME"
                        phase_steps = 0
                        wait_steps = 0
                    else:
                        speed = 3.0 if dist_q > 0.5 else (dist_q / 0.5) * 3.0
                        speed = max(speed, 0.8)
                        q_dot_final = (q_err / dist_q) * speed

                elif phase == "WAIT_HOME":
                    q_dot_final = np.zeros(6)
                    gripper_target = GRIPPER_OPEN
                    if step_counter % sim_steps_per_record == 0:
                        wait_steps += 1
                    if wait_steps >= int(1.0 * fps):
                        print("✅ 5. 任务完美完成！")
                        success = True
                        break

                q_dot_final = np.clip(q_dot_final, -6.0, 6.0)

                if step_counter % sim_steps_per_record == 0:
                    renderer.update_scene(data, camera="rear_cam")
                    img_f = renderer.render()
                    renderer.update_scene(data, camera="wrist_cam")
                    img_w = renderer.render()
                    
                    current_qpos = data.qpos[:8].copy()
                    
                    expert_action = np.zeros(8)
                    expert_action[:6] = current_qpos[:6] + q_dot_final * (1.0 / fps)
                    expert_action[6] = gripper_target
                    expert_action[7] = gripper_target
                    
                    ep_images_fixed.append(img_f)
                    ep_images_wrist.append(img_w)
                    ep_qpos.append(current_qpos)
                    ep_actions.append(expert_action)

                data.ctrl[:6] = data.ctrl[:6] + q_dot_final * model.opt.timestep
                
                ctrl_error = data.ctrl[:6] - data.qpos[:6]
                data.ctrl[:6] = data.qpos[:6] + np.clip(ctrl_error, -0.40, 0.40)

                data.ctrl[6] = gripper_target
                data.ctrl[7] = gripper_target
                
                mujoco.mj_step(model, data)
                
                if step_counter % sim_steps_per_record == 0:
                    viewer.sync()
                    
                step_counter += 1

            if success:
                ep_name = f"ep_{success_count}"
                ep_dir = os.path.join(save_dir, ep_name)
                cam_f_dir = os.path.join(ep_dir, "cam_fixed")
                cam_w_dir = os.path.join(ep_dir, "cam_wrist")
                
                os.makedirs(cam_f_dir, exist_ok=True)
                os.makedirs(cam_w_dir, exist_ok=True)

                for i, (img_f, img_w) in enumerate(zip(ep_images_fixed, ep_images_wrist)):
                    cv2.imwrite(os.path.join(cam_f_dir, f"{i:03d}.jpg"), cv2.cvtColor(img_f, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(cam_w_dir, f"{i:03d}.jpg"), cv2.cvtColor(img_w, cv2.COLOR_RGB2BGR))

                np.savez_compressed(
                    os.path.join(ep_dir, "joint_data.npz"),
                    qpos=np.array(ep_qpos, dtype=np.float32),
                    actions=np.array(ep_actions, dtype=np.float32)
                )
                
                chosen_instruction = np.random.choice(language_instructions)
                with open(os.path.join(ep_dir, "instruction.txt"), "w", encoding="utf-8") as f:
                    f.write(chosen_instruction)
                
                success_count += 1
                print(f"📁 已保存至 {ep_name}，本次总计: {success_count - start_count}/{TARGET_SUCCESS_EPISODES}")

    print(f"\n🎉 大功告成！完美收集 {TARGET_SUCCESS_EPISODES} 条高质量专家数据！")

if __name__ == "__main__":
    main()