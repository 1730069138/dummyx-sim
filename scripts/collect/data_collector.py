import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import cv2

# 💡 新增：用于计算 Site (准星) 的雅可比矩阵
def get_site_jacobian(model, data, site_id):
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)
    return jacp

def damped_pinv(J, rho=0.01):
    return J.T @ np.linalg.inv(J @ J.T + rho**2 * np.eye(J.shape[0]))

def reset_scene(model, data):
    base_x, base_y = 0.2, 0.2
    r_target = np.random.uniform(0.25, 0.35)
    base_angle = -np.pi / 2
    theta_target = base_angle + np.random.uniform(-np.pi/5, np.pi/5)
    target_x = base_x + r_target * np.cos(theta_target)
    target_y = base_y + r_target * np.sin(theta_target)
    target_z = 0.22

    target_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_joint")
    if target_jnt_id != -1:
        target_adr = model.jnt_qposadr[target_jnt_id]
        data.qpos[target_adr : target_adr+3] = [target_x, target_y, target_z]
        data.qpos[target_adr+3 : target_adr+7] = [1, 0, 0, 0]
    else:
        target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
        model.body_pos[target_id] = [target_x, target_y, target_z]

    obs_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_joint")
    if obs_jnt_id != -1:
        obs_adr = model.jnt_qposadr[obs_jnt_id]
        data.qpos[obs_adr : obs_adr+3] = [10.0, 10.0, -10.0]
        data.qpos[obs_adr+3 : obs_adr+7] = [1, 0, 0, 0]
    else:
        obs_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obstacle")
        model.body_pos[obs_id] = [10.0, 10.0, -10.0]

    data.qpos[:8] = 0
    data.qvel[:] = 0
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)

def main():
    # 👇 修改位置 1：动态获取根目录 dummyx/ 绝对路径
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 👇 修改位置 2：基于根目录读取 xml
    xml_path = os.path.join(BASE_DIR, "models", "dummyx_apf_scene.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=256, width=256)
    
    # 👇 修改位置 3：统一存放于根目录的 datasets 文件夹下
    save_dir = os.path.join(BASE_DIR, "datasets", "dataset_pure_grasp") 
    os.makedirs(save_dir, exist_ok=True)

    # 💡 核心修改 1：把追踪目标彻底改为我们刚刚添加的 tcp_site！
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")

    GRIPPER_OPEN = 0.04   
    GRIPPER_CLOSE = 0.0    
    
    TARGET_SUCCESS_EPISODES = 1000 
    fps = 10
    sim_steps_per_record = int((1.0 / fps) / model.opt.timestep)
    
    language_instructions = [
        "Pick up the green cube.",
        "Grasp the green target.",
        "Reach and close the gripper on the green box.",
        "Collect the green object from the table."
    ]
    
    print(f"🚀 启动无敌准星抓取模式！目标: {TARGET_SUCCESS_EPISODES} 条...")

    success_count = 0
    total_attempts = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and success_count < TARGET_SUCCESS_EPISODES:
            total_attempts += 1
            reset_scene(model, data)
            
            phase = "HOVER"
            grasp_record_steps = 0
            success = False
            
            ep_images_fixed, ep_images_wrist = [], []
            ep_qpos, ep_actions = [], []
            step_counter = 0
            
            print(f"\n--- 尝试第 {total_attempts} 轮 (已收集: {success_count}/{TARGET_SUCCESS_EPISODES}) ---")

            while viewer.is_running():
                target_pos = data.xpos[target_id]
                
                # 💡 核心修改 2：读取 site 的位置，而不是 body 的位置
                tcp_pos = data.site_xpos[tcp_id]
                
                # 💡 核心修改 3：因为 TCP 就在手心里，逻辑变得极其清爽！
                # 悬停点：目标正上方 12cm
                hover_point = target_pos + np.array([0.0, 0.0, 0.12])
                # 抓取点：直接就是目标方块的中心点！(稍微抬高5毫米防撞地)
                grasp_point = target_pos + np.array([0.0, 0.0, 0.005])

                dist_to_hover = np.linalg.norm(hover_point - tcp_pos)
                dist_to_grasp = np.linalg.norm(grasp_point - tcp_pos)

                q_dot_final = np.zeros(6)
                gripper_target = GRIPPER_OPEN

                if phase == "HOVER":
                    if dist_to_hover < 0.02:
                        phase = "DESCEND"
                        print("👉 1. 到达正上方，开始垂直下降...")
                    else:
                        v_vla = (hover_point - tcp_pos) / dist_to_hover * 0.15
                        # 💡 注意这里换成了 get_site_jacobian
                        J_tcp = get_site_jacobian(model, data, tcp_id)[:, :6]
                        q_dot_final = damped_pinv(J_tcp, 0.05) @ v_vla

                elif phase == "DESCEND":
                    if dist_to_grasp < 0.01:
                        phase = "GRASP"
                        print("👉 2. 完美套住方块，开始闭合夹爪！")
                    else:
                        v_vla = (grasp_point - tcp_pos) / dist_to_grasp * 0.05 
                        J_tcp = get_site_jacobian(model, data, tcp_id)[:, :6]
                        q_dot_final = damped_pinv(J_tcp, 0.05) @ v_vla

                # ===== 阶段 3：原地闭合夹爪 =====
                elif phase == "GRASP":
                    q_dot_final = np.zeros(6) 
                    gripper_target = GRIPPER_CLOSE 
                    
                    if step_counter % sim_steps_per_record == 0:
                        grasp_record_steps += 1
                        
                    if grasp_record_steps >= int(1.5 * fps):
                        phase = "LIFT"
                        # 💡 核心修复：在进入提起阶段的瞬间，记录下当前TCP位置，
                        # 并将其正上方 15cm 设为“固定绝对目标”，再也不变了！
                        fixed_lift_point = tcp_pos.copy() + np.array([0.0, 0.0, 0.15])
                        print("👉 3. 抓紧了，开始往上提！")

                # ===== 阶段 4：提起方块 =====
                elif phase == "LIFT":
                    gripper_target = GRIPPER_CLOSE 
                    # 💡 核心修复：追踪那个固定目标，而不是会跟着方块跑的 hover_point
                    dist_to_lift = np.linalg.norm(fixed_lift_point - tcp_pos)
                    
                    if dist_to_lift < 0.03:
                        print("✅ 4. 提起成功！全流程录制结束。")
                        success = True
                        break
                    else:
                        v_vla = (fixed_lift_point - tcp_pos) / dist_to_lift * 0.1
                        J_tcp = get_site_jacobian(model, data, tcp_id)[:, :6]
                        q_dot_final = damped_pinv(J_tcp, 0.05) @ v_vla

                q_dot_final = np.clip(q_dot_final, -2.5, 2.5)

                if step_counter % sim_steps_per_record == 0:
                    renderer.update_scene(data, camera="fixed")
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
                data.ctrl[6] = gripper_target
                data.ctrl[7] = gripper_target
                
                mujoco.mj_step(model, data)
                viewer.sync()
                step_counter += 1

                if step_counter > 40000: 
                    print("❌ 动作超时，跳过本轮。")
                    break

            if success:
                ep_name = f"ep_{success_count:04d}_{time.strftime('%H%M%S')}"
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
                print(f"📁 已保存至 {ep_name}")

    print(f"\n🎉 大功告成！")

if __name__ == "__main__":
    main()