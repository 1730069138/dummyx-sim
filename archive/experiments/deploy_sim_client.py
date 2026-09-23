import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse
import cv2  
import os   
from collections import deque

# ==========================================
# 📡 导入 OpenPI 官方 WebSocket 客户端
# ==========================================
from openpi_client import websocket_client_policy

# ==========================================
# 🧮 核心数学工具库 (强化平滑性)
# ==========================================
def get_body_jacobian(model, data, body_id):
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, None, body_id)
    return jacp

def damped_pinv(J, rho=0.12): # 💡 进一步增加阻尼，吸收数值不稳定性
    return J.T @ np.linalg.inv(J @ J.T + rho**2 * np.eye(J.shape[0]))

# ==========================================
# 📍 上帝视角感知 (保持精准跟随)
# ==========================================
def get_ground_truth_obstacle_capsule(model, data):
    obs_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_joint")
    if obs_jnt_id == -1: return None
    obs_body_id = model.jnt_bodyid[obs_jnt_id]
    if model.body_geomnum[obs_body_id] == 0: return None
    geom_adr = model.body_geomadr[obs_body_id]
    geom_pos = data.geom_xpos[geom_adr]
    if geom_pos[2] < 0: return None
    geom_mat = data.geom_xmat[geom_adr].reshape(3, 3)
    z_axis = geom_mat[:, 2] 
    radius = model.geom_size[geom_adr, 0]
    half_len = model.geom_size[geom_adr, 1]
    p3, p4 = geom_pos - z_axis * half_len, geom_pos + z_axis * half_len
    return {"p3": p3, "p4": p4, "r": radius}

def segment_to_segment_distance(p1, p2, p3, p4):
    p1, p2, p3, p4 = map(np.array, (p1, p2, p3, p4))
    d1, d2, r = p2-p1, p4-p3, p1-p3
    a, e, f = np.dot(d1, d1), np.dot(d2, d2), np.dot(d2, r)
    if a <= 1e-6 and e <= 1e-6: s, t = 0.0, 0.0
    elif a <= 1e-6: s, t = 0.0, np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        if e <= 1e-6: t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom != 0.0 else 0.0
            t = (b * s + f) / e
            if t < 0.0: t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0: t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    c1, c2 = p1 + d1 * s, p3 + d2 * t
    vec = c1 - c2
    dist = np.linalg.norm(vec)
    normal = vec / dist if dist > 1e-6 else np.array([1.0, 0.0, 0.0])
    return dist, normal

# ==========================================
# 🛡️ 最终稳定版护盾：低通滤波融合场
# ==========================================
CAPSULES_DEF = [
    {"start": "base_link", "end": "link1_1", "r": 0.055}, 
    {"start": "link1_1",   "end": "link2_1", "r": 0.045}, 
    {"start": "link2_1",   "end": "link3_1", "r": 0.045}, 
    {"start": "link3_1",   "end": "link4_1", "r": 0.040}, 
    {"start": "link4_1",   "end": "link5_1", "r": 0.040}, 
    {"start": "link5_1",   "end": "link6_1", "r": 0.035}, 
    {"start": "link6_1",   "end": "link8",   "r": 0.025}, 
    {"start": "link6_1",   "end": "link9",   "r": 0.025}, 
]

# 用于保存上一帧的速度，实现平滑滤波
last_q_dot = np.zeros(6)

def process_action_and_check_collision(model, data, current_qpos, target_action, dynamic_obs, use_shield):
    global last_q_dot
    q_dot_vla = target_action[:6] - current_qpos[:6]
    is_collision = False 
    
    if dynamic_obs is None:
        safe_target_action = np.zeros(8)
        safe_target_action[:6] = current_qpos[:6] + np.clip(q_dot_vla, -0.08, 0.08)
        safe_target_action[6:] = target_action[6:] 
        return safe_target_action, is_collision
        
    obs_p3, obs_p4, obs_r = dynamic_obs["p3"], dynamic_obs["p4"], dynamic_obs["r"]
    q_dot_avoid = np.zeros(6)
    max_influence = 0.0
    
    for cap in CAPSULES_DEF:
        id_s, id_e = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cap["start"]), mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cap["end"])
        if id_s == -1 or id_e == -1: continue
            
        dist, normal = segment_to_segment_distance(data.xpos[id_s], data.xpos[id_e], obs_p3, obs_p4)
        actual_dist = dist - cap["r"] - obs_r
        if actual_dist <= 0.005: is_collision = True
            
        # 💡 策略 1：动态警戒线。给大臂更大的避让半径，给夹爪更小的。
        safe_margin = 0.08 if "link2" in cap["start"] else 0.05
        
        if use_shield and actual_dist < safe_margin:
            # 💡 策略 2：余弦平滑斥力。相比线性更柔和，减少边缘突变。
            influence = 0.5 * (1.0 + np.cos(np.pi * max(actual_dist, 0.0) / safe_margin))
            max_influence = max(max_influence, influence)
            
            push_speed = influence * 0.12 # 降低推开的最大速度
            v_avoid_task = normal * push_speed
            
            J = get_body_jacobian(model, data, id_e)[:, :6]
            q_dot_avoid += damped_pinv(J, 0.12) @ v_avoid_task

    # 💡 策略 3：非对称权重融合。触发护盾时，大幅削弱 VLA 意图
    # 这样大模型就不会和护盾死磕。
    alpha = np.clip(1.0 - max_influence * 0.85, 0.1, 1.0)
    q_dot_combined = alpha * q_dot_vla + q_dot_avoid

    # 💡 策略 4：一阶低通滤波 (Lpf)。新速度 = 0.2 * 目标 + 0.8 * 上一帧
    # 这是消除抖动的终极法宝。
    lpf_beta = 0.25 
    q_dot_smooth = lpf_beta * q_dot_combined + (1 - lpf_beta) * last_q_dot
    last_q_dot = q_dot_smooth.copy()

    # 💡 策略 5：极度严格的单步截断。
    q_dot_final = np.clip(q_dot_smooth, -0.06, 0.06)

    safe_target_action = np.zeros(8)
    safe_target_action[:6] = current_qpos[:6] + q_dot_final
    safe_target_action[6:] = target_action[6:] 
    return safe_target_action, is_collision

# ==========================================
# 🌍 场景复位与主循环 (全量提供，确保逻辑完整)
# ==========================================
def reset_scene(model, data, use_obstacle=False):
    base_x, base_y = 0.2, 0.2
    r_target = np.random.uniform(0.25, 0.35)
    base_angle = -np.pi / 2
    theta_target = base_angle + np.random.uniform(-np.pi/5, np.pi/5)
    target_x, target_y, target_z = base_x + r_target * np.cos(theta_target), base_y + r_target * np.sin(theta_target), 0.22
    data.qpos[:], data.qvel[:], data.ctrl[:] = 0, 0, 0
    target_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_joint")]
    data.qpos[target_adr : target_adr+3], data.qpos[target_adr+3 : target_adr+7] = [target_x, target_y, target_z], [1, 0, 0, 0]
    obs_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_joint")]
    if use_obstacle:
        ratio = np.random.uniform(0.45, 0.65)
        data.qpos[obs_adr : obs_adr+3] = [base_x + ratio * (target_x - base_x), base_y + ratio * (target_y - base_y), 0.3]
    else: data.qpos[obs_adr : obs_adr+3] = [10.0, 10.0, -10.0]
    data.qpos[obs_adr+3 : obs_adr+7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

def main(args):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    xml_path = os.path.join(project_root, "models", "dummyx_apf_scene.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer_rgb = mujoco.Renderer(model, height=256, width=256)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    success_count, collision_count = 0, 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for episode in range(50):
            if not viewer.is_running(): break
            reset_scene(model, data, use_obstacle=args.use_obstacle)
            viewer.sync()
            step_counter, lift_counter, is_success, episode_col = 0, 0, False, False
            action_chunk_cache, dynamic_obs_info = None, None
            
            while viewer.is_running() and step_counter < 150:
                renderer_rgb.update_scene(data, camera="fixed")
                img_f_rgb = renderer_rgb.render()
                if step_counter % 8 == 0 or action_chunk_cache is None:
                    renderer_rgb.update_scene(data, camera="wrist_cam")
                    img_w_rgb = renderer_rgb.render()
                    dynamic_obs_info = get_ground_truth_obstacle_capsule(model, data) if args.use_obstacle else None
                    action_chunk_cache = policy.infer({"observation/image": img_f_rgb, "observation/wrist_image": img_w_rgb, "observation/state": data.qpos[:8].copy(), "prompt": "Pick up the green cube."})["actions"]
                
                raw_act = np.zeros(8)
                raw_act[:6], raw_act[6:8] = action_chunk_cache[step_counter % 8][:6], (0.0 if action_chunk_cache[step_counter % 8][6] < 0.02 else 0.04)
                
                final_act, step_col = process_action_and_check_collision(model, data, data.qpos[:8].copy(), raw_act, dynamic_obs_info, args.use_shield)
                if step_col: episode_col = True
                data.ctrl[:8] = final_act

                # 🎨 增强可视化
                viewer.user_scn.ngeom = 0 
                if dynamic_obs_info:
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), [1, 1, 0, 0.4])
                    mujoco.mjv_connector(viewer.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_CAPSULE, dynamic_obs_info["r"]+0.005, dynamic_obs_info["p3"], dynamic_obs_info["p4"])
                    viewer.user_scn.ngeom = 1
                if args.use_shield:
                    for cap in CAPSULES_DEF:
                        id_s, id_e = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cap["start"]), mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cap["end"])
                        if id_s == -1 or id_e == -1: continue
                        dist, _ = segment_to_segment_distance(data.xpos[id_s], data.xpos[id_e], dynamic_obs_info["p3"], dynamic_obs_info["p4"]) if dynamic_obs_info else (1.0, None)
                        color = [1, 0, 0, 0.7] if dist - cap["r"] - (dynamic_obs_info["r"] if dynamic_obs_info else 0) < 0.03 else [0, 0.8, 1, 0.2]
                        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), color)
                        mujoco.mjv_connector(viewer.user_scn.geoms[viewer.user_scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, cap["r"], data.xpos[id_s], data.xpos[id_e])
                        viewer.user_scn.ngeom += 1

                for _ in range(int(1500/15/100)): mujoco.mj_step(model, data)
                if data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_joint")] + 2] > 0.25: lift_counter += 1
                else: lift_counter = 0
                if lift_counter >= 20: is_success = True; break
                viewer.sync()
                step_counter += 1
            
            if is_success: success_count += 1
            if episode_col: collision_count += 1
            print(f"📊 回合 {episode+1}: {'✅' if is_success else '❌'} | {'💥' if episode_col else '🛡️'} | 成功率:{(success_count/(episode+1))*100:.1f}% | 碰撞率:{(collision_count/(episode+1))*100:.1f}%")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--use_shield", action="store_true")
    p.add_argument("--use_obstacle", action="store_true")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", default=8000, type=int)
    main(p.parse_args())
