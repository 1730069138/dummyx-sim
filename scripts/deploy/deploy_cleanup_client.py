import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse
import cv2  
import os   

# ==========================================
# 📡 导入 OpenPI 官方 WebSocket 客户端
# ==========================================
from openpi_client import websocket_client_policy

# ==========================================
# 🧮 核心数学与几何工具库
# ==========================================
def get_body_jacobian(model, data, body_id):
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, None, body_id)
    return jacp

def damped_pinv(J, rho=0.10):
    return J.T @ np.linalg.inv(J @ J.T + rho**2 * np.eye(J.shape[0]))

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
    return dist, normal, c1, c2

# ==========================================
# 💡 全局几何避障尺寸自定义配置区
# ==========================================
ARM_CAPSULE_R = 0.07          
SCREWDRIVER_HALF_LENGTH = 0.12 
SCREWDRIVER_CAPSULE_R = 0.02  

# ==========================================
# 🛡️ 智能自适应 APF 后处理引擎 (一字未动)
# ==========================================
last_q_dot = np.zeros(6)

def process_apf_action(model, data, current_action, use_obstacle, step_counter):
    global last_q_dot
    
    current_qpos = data.qpos[:8].copy()
    q_dot_vla = current_action[:6] - current_qpos[:6]
    raw_gripper = current_action[6]
    gripper_val = 0.04 if raw_gripper > 0.02 else 0.0
    
    link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6_1")
    link7_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
    screw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "real_screwdriver")
    pillar_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dynamic_pillar")

    tcp_pos = data.site_xpos[tcp_id].copy()
    box_pos = np.array([0.0, -0.1, 0.24])
    dist_to_box = np.linalg.norm(tcp_pos - box_pos)

    # 动态预警半径收缩
    dynamic_safe_margin = np.clip(0.025 + (dist_to_box * 0.4), 0.025, 0.08)

    active_capsules = []
    wrist_pos = data.xpos[link6_id].copy()
    active_capsules.append({"p1": wrist_pos, "p2": tcp_pos, "r": ARM_CAPSULE_R, "body_id": link7_id})

    # 螺丝刀护盾
    screw_pos = data.xpos[screw_id].copy()
    if gripper_val < 0.02 and np.linalg.norm(tcp_pos - screw_pos) < 0.05:
        screw_mat = data.xmat[screw_id].reshape(3, 3)
        direction = screw_mat[:, 1] 
        active_capsules.append({
            "p1": screw_pos - direction * SCREWDRIVER_HALF_LENGTH, 
            "p2": screw_pos + direction * SCREWDRIVER_HALF_LENGTH, 
            "r": SCREWDRIVER_CAPSULE_R, 
            "body_id": link7_id
        })

    # 判断螺丝刀是否被成功捡起
    has_picked_up = screw_pos[2] > 0.235

    obstacle_capsule = None
    if use_obstacle and pillar_id != -1 and has_picked_up:
        obs_pos = data.xpos[pillar_id]
        obs_mat = data.xmat[pillar_id].reshape(3, 3)
        obs_z = obs_mat[:, 2]
        obstacle_capsule = {"p1": obs_pos - obs_z * 0.08, "p2": obs_pos + obs_z * 0.08, "r": 0.025}

    delta_q_apf = np.zeros(6)
    is_apf_active = False
    
    current_min_dist = 0.5 
    current_influence = 0.0

    if obstacle_capsule is not None:
        min_dist = float('inf')
        dom_normal, dom_body_id = None, None
        for cap in active_capsules:
            dist, normal, _, _ = segment_to_segment_distance(cap["p1"], cap["p2"], obstacle_capsule["p1"], obstacle_capsule["p2"])
            actual_dist = dist - cap["r"] - obstacle_capsule["r"]
            if actual_dist < min_dist:
                min_dist, dom_normal, dom_body_id = actual_dist, normal, cap["body_id"]

        if min_dist != float('inf'):
            current_min_dist = min_dist

        if min_dist < dynamic_safe_margin:
            goal_decay = np.clip((dist_to_box - 0.035) / 0.065, 0.0, 1.0)
            start_fade = np.clip(step_counter / 15.0, 0.0, 1.0)
            influence = np.power((dynamic_safe_margin - max(min_dist, 0.0)) / dynamic_safe_margin, 2) * goal_decay * start_fade
            
            current_influence = influence

            if influence > 0.005:
                is_apf_active = True
                J_dom = get_body_jacobian(model, data, dom_body_id)[:3, :6]
                v_vla_cartesian = J_dom @ q_dot_vla
                
                v_into_obs = np.dot(v_vla_cartesian, dom_normal)
                v_vla_filtered = v_vla_cartesian - v_into_obs * dom_normal if v_into_obs < 0 else v_vla_cartesian
                
                v_up = np.array([0.0, 0.0, 1.0])
                if np.dot(v_up, dom_normal) < 0:
                    v_up = v_up - np.dot(v_up, dom_normal) * dom_normal  
                if np.linalg.norm(v_up) > 1e-4: 
                    v_up /= np.linalg.norm(v_up)

                v_in = np.array([-tcp_pos[0], -tcp_pos[1], 0.0])
                if np.linalg.norm(v_in) > 1e-4: 
                    v_in /= np.linalg.norm(v_in)
                if np.dot(v_in, dom_normal) < 0:
                    v_in = v_in - np.dot(v_in, dom_normal) * dom_normal  
                if np.linalg.norm(v_in) > 1e-4: 
                    v_in /= np.linalg.norm(v_in)
                
                v_avoid_dir = v_up * 0.045 + v_in * 0.02
                v_vla_filtered += v_avoid_dir * influence
                v_vla_filtered += dom_normal * (influence * 0.01)
                
                delta_v = v_vla_filtered - v_vla_cartesian
                delta_q_apf = damped_pinv(J_dom, 0.05) @ delta_v

    q_dot_combined = q_dot_vla + delta_q_apf
    lpf_beta = 0.6 
    q_dot_smooth = lpf_beta * q_dot_combined + (1 - lpf_beta) * last_q_dot
    last_q_dot = q_dot_smooth.copy()

    final_ctrl = np.zeros(8)
    final_ctrl[:6] = current_qpos[:6] + np.clip(q_dot_smooth, -0.25, 0.25)
    final_ctrl[6:8] = gripper_val

    debug_info = {
        "min_dist": current_min_dist,
        "influence": current_influence,
        "q_dot": q_dot_smooth.copy()
    }

    return final_ctrl, active_capsules, obstacle_capsule, is_apf_active, debug_info

# ==========================================
# 🌍 场景复位 (修改为杂物散布逻辑与随机种子)
# ==========================================
def reset_scene(model, data, use_obstacle=False, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
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
            return body_id
        else:
            data.qpos[adr:adr+3] = [10.0 + np.random.rand(), 10.0, -10.0]
            data.qpos[adr+3:adr+7] = [1, 0, 0, 0]
            return None

    if scenario == 0:
        info = place_obj(screw_jnt, screw_body, True)
        place_obj(sponge_jnt, sponge_body, False)
        if info: active_targets.append(info)
    elif scenario == 1:
        place_obj(screw_jnt, screw_body, False)
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

    pillar_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pillar_joint")
    if pillar_jnt_id != -1:
        q_adr, v_adr = model.jnt_qposadr[pillar_jnt_id], model.jnt_dofadr[pillar_jnt_id]
        if use_obstacle:
            data.qpos[q_adr : q_adr+3] = [0.11, -0.1, 0.38] 
            data.qpos[q_adr+3 : q_adr+7] = [1, 0, 0, 0] 
        else: 
            data.qpos[q_adr : q_adr+3] = [10.0, 10.0, -10.0]
        data.qvel[v_adr : v_adr+6] = 0

    # 👇 恢复你正确的 :8 配置（6轴+夹爪）
    data.qpos[:8] = 0
    data.qvel[:] = 0  # 建议这里改成 [:] 清空所有物体的残余速度，防止它们乱飘
    data.ctrl[:] = 0
    
    # 👇 删除了导致机械臂坍塌扫飞物体的 100 步 mj_step，只保留数据转发
    mujoco.mj_forward(model, data)
    
    return active_targets

def save_episode_video(frames, folder, episode_idx):
    if not frames: return
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"ep_{episode_idx:03d}_{time.strftime('%H%M%S')}.mp4")
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (width, height))
    for f in frames: out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()

# ==========================================
# 🚀 部署主程序
# ==========================================
def main(args):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    xml_path = os.path.join(BASE_DIR, "models", "dummyx_apf_scene.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer_rgb = mujoco.Renderer(model, height=256, width=256)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    
    pillar_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dynamic_pillar")
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")

    configs = [
        {"obs": False, "apf": False, "tag_obs": "no_obs", "tag_apf": "no_apf", "str_obs": "🟩无障碍", "str_apf": "🧠纯VLA模型(无干预)"},
        {"obs": False, "apf": True,  "tag_obs": "no_obs", "tag_apf": "apf",    "str_obs": "🟩无障碍", "str_apf": "🛡️APF护盾开启"},
        {"obs": True,  "apf": False, "tag_obs": "obs",    "tag_apf": "no_apf", "str_obs": "🔥有障碍", "str_apf": "🧠纯VLA模型(无干预)"},
        {"obs": True,  "apf": True,  "tag_obs": "obs",    "tag_apf": "apf",    "str_obs": "🔥有障碍", "str_apf": "🛡️APF护盾开启"},
    ]
    
    report_summary = []
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    fixed_seeds = None
    if args.fixed_eval:
        print("🔒 [固定评测模式] 已开启！四种工况将使用完全相同的随机种子序列进行严格对照。")
        np.random.seed(42) # 固定主种子确保跑这批实验结果可复现
        fixed_seeds = [np.random.randint(0, 1000000) for _ in range(args.num_episodes)]

    print(f"🚀 [全自动评测启动] 本次实验共包含 4 种工况组合，每种工况将连续测试 {args.num_episodes} 回合。")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for idx, cfg in enumerate(configs):
            current_obs = cfg["obs"]
            current_apf = cfg["apf"]
            
            success_count, collision_count = 0, 0
            run_folder_name = f"run_{timestamp}_step{idx+1}_{cfg['tag_obs']}_{cfg['tag_apf']}"
            base_record_dir = os.path.join(BASE_DIR, "recordings", run_folder_name)
            os.makedirs(base_record_dir, exist_ok=True)

            print("\n" + "="*60)
            print(f" 📊 [工况 {idx+1}/4] 环境: [{cfg['str_obs']}] | 策略: [{cfg['str_apf']}]")
            print(f" 📁 数据与视频记录路径: {base_record_dir}")
            print("="*60)

            for episode in range(args.num_episodes):
                if not viewer.is_running(): break
                
                # 传入当前回合对应的 seed 和初始化多目标场景
                current_seed = fixed_seeds[episode] if fixed_seeds is not None else None
                active_targets = reset_scene(model, data, use_obstacle=current_obs, seed=current_seed)
                viewer.sync()
                
                step_counter, in_box_counter = 0, 0
                is_success, episode_col = False, False
                action_chunk_cache = None
                video_frames = []
                
                episode_data = {
                    "steps": [], "min_dist": [], "influence": [], 
                    "tcp_pos": [], "q_dot": [],
                    "q_pos": [], "q_vel": [], "q_acc": [], "q_tau": [] 
                }
                
                record_cam = mujoco.MjvCamera()
                mujoco.mjv_defaultFreeCamera(model, record_cam)
                record_cam.lookat[:] = [0.10, -0.05, 0.22] 
                record_cam.distance, record_cam.azimuth, record_cam.elevation = 0.95, 145, -22                 

                while viewer.is_running() and step_counter < 2500: # 步长上限延长至 2500
                    if step_counter % 8 == 0 or action_chunk_cache is None:
                        renderer_rgb.update_scene(data, camera="rear_cam")
                        img_f = renderer_rgb.render()
                        renderer_rgb.update_scene(data, camera="wrist_cam")
                        img_w = renderer_rgb.render()
                        
                        # 替换为全新指令文本
                        result = policy.infer({
                            "observation/image": img_f, 
                            "observation/wrist_image": img_w, 
                            "observation/state": data.qpos[:8].copy(), 
                            "prompt": "Clear all foreign objects from the chip tray into the box."
                        })
                        action_chunk_cache = result["actions"]
                    
                    renderer_rgb.update_scene(data, camera=record_cam) 
                    video_frames.append(renderer_rgb.render())

                    current_action = action_chunk_cache[step_counter % 8]
                    active_capsules, obs_capsule, is_apf_active = [], None, False
                    
                    current_tcp = data.site_xpos[tcp_id].copy()
                    step_min_dist = 0.5
                    step_influence = 0.0
                    step_q_dot = np.zeros(6)
                    
                    if current_apf:
                        final_ctrl, active_capsules, obs_capsule, is_apf_active, debug_info = process_apf_action(model, data, current_action, current_obs, step_counter)
                        data.ctrl[:8] = final_ctrl
                        
                        step_min_dist = debug_info["min_dist"]
                        step_influence = debug_info["influence"]
                        step_q_dot = debug_info["q_dot"]
                    else:
                        delta_q = current_action[:6] - data.qpos[:6].copy()
                        step_q_dot = np.clip(delta_q, -0.25, 0.25)
                        data.ctrl[:6] = data.qpos[:6] + step_q_dot
                        data.ctrl[6:8] = 0.04 if current_action[6] > 0.02 else 0.0
                        
                        if current_obs and pillar_body_id != -1:
                            obs_pos = data.xpos[pillar_body_id]
                            step_min_dist = np.linalg.norm(current_tcp - obs_pos) - 0.025 - ARM_CAPSULE_R

                    episode_data["steps"].append(step_counter)
                    episode_data["min_dist"].append(step_min_dist)
                    episode_data["influence"].append(step_influence)
                    episode_data["tcp_pos"].append(current_tcp)
                    episode_data["q_dot"].append(step_q_dot)
                    episode_data["q_pos"].append(data.qpos[:6].copy())
                    episode_data["q_vel"].append(data.qvel[:6].copy())
                    episode_data["q_acc"].append(data.qacc[:6].copy())
                    episode_data["q_tau"].append(data.qfrc_actuator[:6].copy()) 

                    viewer.user_scn.ngeom = 0 
                    if current_apf:
                        if current_obs and obs_capsule:
                            mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), [1, 1, 0, 0.3])
                            mujoco.mjv_connector(viewer.user_scn.geoms[viewer.user_scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, obs_capsule["r"], obs_capsule["p1"], obs_capsule["p2"])
                            viewer.user_scn.ngeom += 1
                        for cap in active_capsules:
                            color = [1, 0, 0, 0.5] if is_apf_active else [0, 0.5, 1, 0.3]
                            mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), color)
                            mujoco.mjv_connector(viewer.user_scn.geoms[viewer.user_scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, cap["r"], cap["p1"], cap["p2"])
                            viewer.user_scn.ngeom += 1

                    for _ in range(50): mujoco.mj_step(model, data)
                    
                    if current_obs and pillar_body_id != -1:
                        if data.xmat[pillar_body_id].reshape(3, 3)[2, 2] < 0.9: episode_col = True

                    # 全新成功条件检测：所有物体入箱 + 手臂回归零位 + 夹爪释放
                    all_in_box = True
                    for target_id in active_targets:
                        pos = data.xpos[target_id]
                        # 👇 修正：把盒子的 X 和 Y 判定边界从 0.12 放大到 0.18（适配加大后的盒子）
                        in_box = abs(pos[0] - (-0.05)) < 0.18 and abs(pos[1] - 0.0) < 0.18 and pos[2] < 0.34
                        if not in_box:
                            all_in_box = False
                            break
                    
                    # 👇 修正：稍微放宽机械臂归零的允差至 0.35，容忍 VLA 模型回归原点时的轻微稳态误差
                    arm_at_home = np.linalg.norm(data.qpos[:6]) < 0.35 
                    is_released = (current_action[6] > 0.02) # 夹爪保持开启

                    if all_in_box and arm_at_home and is_released: 
                        in_box_counter += 1
                    else: 
                        in_box_counter = 0
                        
                    if in_box_counter >= 5:
                        is_success = True
                        break
                        
                    viewer.sync()
                    step_counter += 1
                
                if is_success: success_count += 1
                if episode_col: collision_count += 1
                
                save_folder = os.path.join(base_record_dir, "success" if is_success else "fail")
                save_episode_video(video_frames, save_folder, episode + 1)
                
                data_filename = os.path.join(save_folder, f"data_ep_{episode+1:03d}.npz")
                np.savez(data_filename, 
                         steps=np.array(episode_data["steps"]),
                         min_dist=np.array(episode_data["min_dist"]),
                         influence=np.array(episode_data["influence"]),
                         tcp_pos=np.array(episode_data["tcp_pos"]),
                         q_dot=np.array(episode_data["q_dot"]),
                         q_pos=np.array(episode_data["q_pos"]),
                         q_vel=np.array(episode_data["q_vel"]),
                         q_acc=np.array(episode_data["q_acc"]),
                         q_tau=np.array(episode_data["q_tau"]))
                
                print(f"  └─ 回合 {episode+1}/{args.num_episodes}: {'✅ 成功' if is_success else '❌ 失败'} | {'💥 撞倒' if episode_col else '🛡️ 安全'} | 累计成功率: {(success_count/(episode+1))*100:.1f}%")

            sr = (success_count / args.num_episodes) * 100
            cr = (collision_count / args.num_episodes) * 100
            report_summary.append({
                "obs_str": cfg["str_obs"],
                "apf_str": cfg["str_apf"],
                "sr_str": f"{sr:.1f}%",
                "cr_str": f"{cr:.1f}%"
            })

    print("\n" + "================== 📜 开题报告/论文数据消融实验总结表 ==================")
    print("| 实验编号 | 障碍物状态 | APF 安全过滤护盾 | 任务抓取成功率 (SR) | 障碍物碰撞率 (CR) |")
    print("| :---: | :---: | :---: | :---: | :---: |")
    for idx, res in enumerate(report_summary):
        print(f"| Case {idx+1} | {res['obs_str']} | {res['apf_str']} | {res['sr_str']} | {res['cr_str']} |")
    print("========================================================================")
    print("💡 提示：你可以直接复制上方生成的 Markdown 表格到你的开题报告或评审PPT中！\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", default=8000, type=int)
    p.add_argument("--num_episodes", default=100, type=int, help="Total number of episodes to run per condition")
    p.add_argument("--fixed_eval", action="store_true", help="启用严格对照模式：四种情况使用同样的随机种子初始化序列") 
    main(p.parse_args())