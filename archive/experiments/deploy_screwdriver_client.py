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

ARM_CAPSULE_R = 0.07          
SCREWDRIVER_HALF_LENGTH = 0.12 
SCREWDRIVER_CAPSULE_R = 0.02  

# ==========================================
# 🛡️ 智能自适应 APF (10Hz 运行)
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

    dynamic_safe_margin = np.clip(0.025 + (dist_to_box * 0.4), 0.025, 0.08)

    active_capsules = []
    wrist_pos = data.xpos[link6_id].copy()
    active_capsules.append({"p1": wrist_pos, "p2": tcp_pos, "r": ARM_CAPSULE_R, "body_id": link7_id})

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
# 🆕 500Hz 级别的精准接触力提取
# ==========================================
def get_target_table_force(model, data, valid_body_ids):
    f_xyz = np.zeros(3)
    for i in range(data.ncon):
        contact = data.contact[i]
        if 0.18 < contact.pos[2] < 0.22:
            geom1 = contact.geom1
            geom2 = contact.geom2
            body1 = model.geom_bodyid[geom1]
            body2 = model.geom_bodyid[geom2]
            
            if body1 in valid_body_ids or body2 in valid_body_ids:
                c_force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(model, data, i, c_force)
                c_mat = contact.frame.reshape(3, 3)
                f_world = c_mat.T @ c_force[:3]
                if f_world[2] < 0: f_world = -f_world
                f_xyz += f_world
    return f_xyz

# ==========================================
# 🆕 阻抗/导纳 双核切换防护模块 (新增接触前“软着陆”制动)
# ==========================================
global_last_v_comp = np.zeros(3) 
global_vc = np.zeros(3) 
global_f_filtered = np.zeros(3)  

def apply_virtual_wall_protection(model, data, ctrl_cmd, tcp_id, current_f_xyz, step_counter, control_mode='admittance'):
    global global_last_v_comp, global_vc, global_f_filtered
    tcp_pos = data.site_xpos[tcp_id]

    current_q = data.qpos[:6].copy()
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, tcp_id)
    J_pos = jacp[:3, :6]  
    
    q_dot_cmd = ctrl_cmd[:6] - current_q
    v_cmd_3d = np.dot(J_pos, q_dot_cmd)  
    
    dist_to_floor = tcp_pos[2] - 0.212 
    if dist_to_floor < 0.03 and v_cmd_3d[2] < 0:
        brake_factor = np.clip(dist_to_floor / 0.03, 0.02, 1.0)
        v_cmd_3d[2] *= (brake_factor ** 2)
    
    force_norm = np.linalg.norm(current_f_xyz)
    is_wall_active = False
    
    v_target_3d = v_cmd_3d.copy()
    
    if tcp_pos[2] < 0.235:
        is_wall_active = True
        
        # 1. 导纳控制模式
        if control_mode == 'admittance':
            if tcp_pos[2] < 0.212 and v_target_3d[2] < 0:
                v_target_3d[2] = 0.0  
            
            force_error = force_norm - 0.5
            v_compensation = np.zeros(3)
            if force_error > 0:
                f_dir = current_f_xyz / force_norm
                f_dir[0] *= 0.1  
                f_dir[1] *= 0.1  
                f_dir = f_dir / (np.linalg.norm(f_dir) + 1e-6) 
                
                K_admittance = 0.001 + 0.003 * force_norm
                v_ideal = force_error * K_admittance * f_dir
                
                v_norm = np.linalg.norm(v_ideal)
                if v_norm > 0.05: v_ideal = (v_ideal / v_norm) * 0.05
                v_compensation = v_ideal
                
            alpha = np.clip(0.02 + 0.01 * force_norm, 0.02, 0.1)
            global_last_v_comp = alpha * v_compensation + (1 - alpha) * global_last_v_comp
            v_target_3d += global_last_v_comp
            
            global_vc = global_last_v_comp.copy()
            global_f_filtered = current_f_xyz.copy()
            
            delta_v_3d = v_target_3d - v_cmd_3d
            delta_q = damped_pinv(J_pos, 0.05) @ delta_v_3d
            ctrl_cmd[:6] = current_q + q_dot_cmd + delta_q

        # 2. 模型参考二阶阻抗控制模式
        elif control_mode == 'impedance':
            if tcp_pos[2] < 0.212 and v_target_3d[2] < 0:
                v_target_3d[2] = 0.0
                
            global_f_filtered = 0.1 * current_f_xyz + 0.9 * global_f_filtered
            f_norm_filt = np.linalg.norm(global_f_filtered)
                
            force_error = f_norm_filt - 0.5
            if force_error > 0:
                f_dir = global_f_filtered / f_norm_filt
                f_dir[0] *= 0.1
                f_dir[1] *= 0.1
                f_dir = f_dir / (np.linalg.norm(f_dir) + 1e-6)
                F_eff = force_error * f_dir
            else:
                F_eff = np.zeros(3)
                
            M = 3.0    # 虚拟质量
            B = 150.0  # 虚拟阻尼
            dt = model.opt.timestep 
            
            a_virtual = (F_eff - B * global_vc) / M
            global_vc += a_virtual * dt
            
            v_norm = np.linalg.norm(global_vc)
            if v_norm > 0.05:
                global_vc = (global_vc / v_norm) * 0.05
                
            v_target_3d += global_vc
            global_last_v_comp = global_vc.copy()
            
            delta_v_3d = v_target_3d - v_cmd_3d
            delta_q = damped_pinv(J_pos, 0.05) @ delta_v_3d
            ctrl_cmd[:6] = current_q + q_dot_cmd + delta_q

    else:
        # 平滑衰减退让记忆
        global_last_v_comp *= 0.92
        global_vc *= 0.92
        global_f_filtered *= 0.8  
        if np.linalg.norm(global_last_v_comp) < 1e-4:
            global_last_v_comp = np.zeros(3)
            global_vc = np.zeros(3)
            global_f_filtered = np.zeros(3)
            
        v_target_3d += global_last_v_comp
        
        delta_v_3d = v_target_3d - v_cmd_3d
        delta_q = damped_pinv(J_pos, 0.05) @ delta_v_3d
        ctrl_cmd[:6] = current_q + q_dot_cmd + delta_q
        
    return ctrl_cmd, is_wall_active

# ==========================================
# 🌍 场景复位
# ==========================================
def reset_scene(model, data, use_obstacle=False, target_xy=None):
    global global_last_v_comp, global_vc, global_f_filtered
    global_last_v_comp = np.zeros(3) 
    global_vc = np.zeros(3)
    global_f_filtered = np.zeros(3)
    
    if target_xy is not None:
        target_x, target_y = target_xy
    else:
        target_x = np.random.uniform(0.20, 0.35)    
        target_y = np.random.uniform(-0.15, -0.05)  
        
    target_z = 0.22
    target_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "fj_screwdriver")
    if target_jnt_id != -1:
        target_adr = model.jnt_qposadr[target_jnt_id]
        data.qpos[target_adr : target_adr+3] = [target_x, target_y, target_z]
        data.qpos[target_adr+3 : target_adr+7] = [1, 0, 0, 0]
    pillar_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pillar_joint")
    if pillar_jnt_id != -1:
        q_adr, v_adr = model.jnt_qposadr[pillar_jnt_id], model.jnt_dofadr[pillar_jnt_id]
        if use_obstacle:
            data.qpos[q_adr : q_adr+3] = [0.11, -0.1, 0.38] 
            data.qpos[q_adr+3 : q_adr+7] = [1, 0, 0, 0] 
        else: data.qpos[q_adr : q_adr+3] = [10.0, 10.0, -10.0]
        data.qvel[v_adr : v_adr+6] = 0
    data.qpos[:8] = 0
    data.qvel[:8] = 0
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)

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
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml_path = os.path.join(BASE_DIR, "dummyx_apf_scene.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer_rgb = mujoco.Renderer(model, height=256, width=256)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "real_screwdriver")
    pillar_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dynamic_pillar")
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
    
    link7_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    link8_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link8")
    link9_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link9")
    
    valid_collision_bodies = [target_body_id]
    if link7_id != -1: valid_collision_bodies.append(link7_id)
    if link8_id != -1: valid_collision_bodies.append(link8_id)
    if link9_id != -1: valid_collision_bodies.append(link9_id)

    all_configs = [
        {"case_id": 1, "obs": False, "apf": False, "contact": False, "tag_obs": "no_obs", "tag_apf": "no_apf", "tag_cont": "no_cont", "str_obs": "🟩无障碍", "str_apf": "🧠纯VLA", "str_cont": "❌无保护"},
        {"case_id": 2, "obs": False, "apf": True,  "contact": False, "tag_obs": "no_obs", "tag_apf": "apf",    "tag_cont": "no_cont", "str_obs": "🟩无障碍", "str_apf": "🛡️仅APF", "str_cont": "❌无保护"},
        {"case_id": 3, "obs": True,  "apf": False, "contact": False, "tag_obs": "obs",    "tag_apf": "no_apf", "tag_cont": "no_cont", "str_obs": "🔥有障碍", "str_apf": "🧠纯VLA", "str_cont": "❌无保护"},
        {"case_id": 4, "obs": True,  "apf": True,  "contact": False, "tag_obs": "obs",    "tag_apf": "apf",    "tag_cont": "no_cont", "str_obs": "🔥有障碍", "str_apf": "🛡️仅APF", "str_cont": "❌无保护"},
        {"case_id": 5, "obs": False, "apf": False, "contact": True,  "tag_obs": "no_obs", "tag_apf": "no_apf", "tag_cont": "cont",    "str_obs": "🟩无障碍", "str_apf": "🧠纯VLA", "str_cont": "✅虚拟墙"},
        {"case_id": 6, "obs": False, "apf": True,  "contact": True,  "tag_obs": "no_obs", "tag_apf": "apf",    "tag_cont": "cont",    "str_obs": "🟩无障碍", "str_apf": "🛡️仅APF", "str_cont": "✅虚拟墙"},
        {"case_id": 7, "obs": True,  "apf": False, "contact": True,  "tag_obs": "obs",    "tag_apf": "no_apf", "tag_cont": "cont",    "str_obs": "🔥有障碍", "str_apf": "🧠纯VLA", "str_cont": "✅虚拟墙"},
        {"case_id": 8, "obs": True,  "apf": True,  "contact": True,  "tag_obs": "obs",    "tag_apf": "apf",    "tag_cont": "cont",    "str_obs": "🔥有障碍", "str_apf": "🛡️仅APF", "str_cont": "✅双环保护"},
    ]
    
    if 1 <= args.case <= 8:
        configs = [all_configs[args.case - 1]]
        print(f"\n🎯 [精准测试模式] 将仅运行 Case {args.case}，测试 {args.num_episodes} 次。")
    else:
        configs = all_configs
        print(f"\n🚀 [全量消融模式] 将按顺序测试所选力控策略对应的工况。")

    # 🚀🚀🚀 核心逻辑：智能构建测试队列 🚀🚀🚀
    test_queue = []
    for cfg in configs:
        if not cfg["contact"]:
            # 无接触保护工况，只测一次
            c = cfg.copy()
            c['mode'] = 'none' 
            c['mode_str'] = '⚪ 无'
            test_queue.append(c)
        else:
            if args.control_mode in ['admittance', 'both']:
                c = cfg.copy()
                c['mode'] = 'admittance'
                c['mode_str'] = '🔵 导纳(Adm)'
                test_queue.append(c)
            if args.control_mode in ['impedance', 'both']:
                c = cfg.copy()
                c['mode'] = 'impedance'
                c['mode_str'] = '🔴 阻抗(Imp)'
                test_queue.append(c)
                
    print(f"========================================================")
    print(f"⚙️  已构建智能测试队列: 共需运行 {len(test_queue)} 组消融实验")
    print(f"========================================================\n")

    report_summary = []
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    # 🚨🚨🚨 修改功能：将随机序列改为绝对定点 🚨🚨🚨
    fixed_targets = None
    if args.fixed_eval:
        print("🔒 [固定评测模式] 开启！各工况将使用完全相同的初始位置。")
        # 将原本的随机生成序列，改为生成 1 个绝对坐标并复制 args.num_episodes 次
        absolute_fixed_xy = (0.28, -0.10) # 设为一个易于 VLA 抓取的中心偏上位置
        fixed_targets = [absolute_fixed_xy for _ in range(args.num_episodes)]
        print(f"📌 [绝对静止] 螺丝刀坐标已焊死在: X={absolute_fixed_xy[0]}, Y={absolute_fixed_xy[1]}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for cfg in test_queue:
            success_count, collision_count = 0, 0
            total_max_tau = 0.0
            
            episode_peak_forces = [] 
            all_impulses = []        
            succ_impulses = []       
            
            run_folder_name = f"run_{timestamp}_{cfg['mode']}_Case{cfg['case_id']}_{cfg['tag_obs']}_{cfg['tag_apf']}_{cfg['tag_cont']}"
            base_record_dir = os.path.join(BASE_DIR, "recordings", run_folder_name)
            os.makedirs(base_record_dir, exist_ok=True)

            print("\n" + "="*80)
            print(f" 📊 [运行工况 Case {cfg['case_id']}] 策略: [{cfg['mode_str']}] | 障碍: [{cfg['str_obs']}] | APF: [{cfg['str_apf']}] | 保护: [{cfg['str_cont']}]")
            print("="*80)

            for episode in range(args.num_episodes):
                if not viewer.is_running(): break
                
                current_target_xy = fixed_targets[episode] if fixed_targets is not None else None
                reset_scene(model, data, use_obstacle=cfg["obs"], target_xy=current_target_xy)
                viewer.sync()
                
                step_counter, in_box_counter = 0, 0
                is_success, episode_col = False, False
                action_chunk_cache = None
                video_frames = []
                episode_taus = []
                
                episode_data = {
                    "steps": [], "min_dist": [], "influence": [], "is_apf_active": [],
                    "tcp_pos": [], "q_dot": [],
                    "q_pos": [], "q_vel": [], "q_acc": [], "q_tau": [],
                    "f_z": [], "is_contact": [],
                    "f_real_x": [], "f_real_y": [], "f_real_z": [], 
                    "q_pos_vla": [], "q_dot_vla": [] 
                }
                
                record_cam = mujoco.MjvCamera()
                mujoco.mjv_defaultFreeCamera(model, record_cam)
                record_cam.lookat[:] = [0.25, -0.05, 0.22]
                record_cam.distance, record_cam.azimuth, record_cam.elevation = 1.0, 180, -20 
                
                ep_true_peak_f = 0.0
                ep_true_impulse = 0.0

                while viewer.is_running() and step_counter < 600:
                    if step_counter % 8 == 0 or action_chunk_cache is None:
                        renderer_rgb.update_scene(data, camera="rear_cam")
                        img_f = renderer_rgb.render()
                        renderer_rgb.update_scene(data, camera="wrist_cam")
                        img_w = renderer_rgb.render()
                        result = policy.infer({"observation/image": img_f, "observation/wrist_image": img_w, "observation/state": data.qpos[:8].copy(), "prompt": "Pick up the screwdriver and drop it into the box."})
                        action_chunk_cache = result["actions"]
                    
                    renderer_rgb.update_scene(data, camera=record_cam) 
                    video_frames.append(renderer_rgb.render())

                    current_action = action_chunk_cache[step_counter % 8]
                    active_capsules, obs_capsule, is_apf_active = [], None, False
                    
                    current_tcp = data.site_xpos[tcp_id].copy()
                    raw_q_pos_vla = current_action[:6].copy()
                    raw_q_dot_vla = raw_q_pos_vla - data.qpos[:6].copy()
                    
                    step_min_dist = 0.5
                    step_influence = 0.0
                    
                    if cfg["apf"]:
                        base_ctrl, active_capsules, obs_capsule, is_apf_active, debug_info = process_apf_action(model, data, current_action, cfg["obs"], step_counter)
                        step_min_dist = debug_info["min_dist"]
                        step_influence = debug_info["influence"]
                        step_q_dot_vla = debug_info["q_dot"]
                    else:
                        step_q_dot_vla = np.clip(raw_q_dot_vla, -0.25, 0.25)
                        base_ctrl = np.zeros(8)
                        base_ctrl[:6] = data.qpos[:6] + step_q_dot_vla
                        base_ctrl[6:8] = 0.04 if current_action[6] > 0.02 else 0.0
                        if cfg["obs"] and pillar_body_id != -1:
                            obs_pos = data.xpos[pillar_body_id]
                            step_min_dist = np.linalg.norm(current_tcp - obs_pos) - 0.025 - ARM_CAPSULE_R

                    is_contact_ui = False
                    max_f_norm_step = 0.0  
                    max_f_xyz_step = np.zeros(3) 
                    
                    for _ in range(50):
                        current_f_xyz = get_target_table_force(model, data, valid_collision_bodies)
                        current_force_norm = np.linalg.norm(current_f_xyz)
                        
                        if current_force_norm > max_f_norm_step:
                            max_f_norm_step = current_force_norm
                            max_f_xyz_step = current_f_xyz.copy()
                            
                        if current_force_norm > ep_true_peak_f:
                            ep_true_peak_f = current_force_norm
                        ep_true_impulse += current_force_norm * model.opt.timestep

                        if cfg["contact"]:
                            current_ctrl, wall_active_step = apply_virtual_wall_protection(
                                model, data, base_ctrl.copy(), tcp_id, current_f_xyz, step_counter, cfg["mode"]
                            )
                            if wall_active_step: 
                                is_contact_ui = True
                        else:
                            current_ctrl = base_ctrl.copy()

                        data.ctrl[:8] = current_ctrl
                        mujoco.mj_step(model, data)
                        
                        current_tau = data.qfrc_actuator[:6].copy()
                        episode_taus.append(np.max(np.abs(current_tau)))
                    
                    final_step_q_dot = current_ctrl[:6] - data.qpos[:6].copy()

                    episode_data["steps"].append(step_counter)
                    episode_data["min_dist"].append(step_min_dist)
                    episode_data["influence"].append(step_influence)
                    episode_data["is_apf_active"].append(is_apf_active)
                    episode_data["tcp_pos"].append(current_tcp)
                    episode_data["q_dot"].append(final_step_q_dot)
                    episode_data["q_pos"].append(data.qpos[:6].copy())
                    episode_data["q_vel"].append(data.qvel[:6].copy())
                    episode_data["q_acc"].append(data.qacc[:6].copy())
                    episode_data["q_tau"].append(current_tau)
                    
                    episode_data["f_z"].append(max_f_norm_step) 
                    episode_data["f_real_x"].append(max_f_xyz_step[0])
                    episode_data["f_real_y"].append(max_f_xyz_step[1])
                    episode_data["f_real_z"].append(max_f_xyz_step[2])
                    
                    episode_data["is_contact"].append(is_contact_ui)
                    episode_data["q_pos_vla"].append(raw_q_pos_vla)
                    episode_data["q_dot_vla"].append(raw_q_dot_vla)

                    viewer.user_scn.ngeom = 0 
                    if cfg["apf"]:
                        if cfg["obs"] and obs_capsule:
                            mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), [1, 1, 0, 0.3])
                            mujoco.mjv_connector(viewer.user_scn.geoms[viewer.user_scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, obs_capsule["r"], obs_capsule["p1"], obs_capsule["p2"])
                            viewer.user_scn.ngeom += 1
                        for cap in active_capsules:
                            color = [1, 0, 0, 0.5] if is_apf_active else [0, 0.5, 1, 0.3]
                            mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), color)
                            mujoco.mjv_connector(viewer.user_scn.geoms[viewer.user_scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, cap["r"], cap["p1"], cap["p2"])
                            viewer.user_scn.ngeom += 1
                            
                    if cfg["contact"]:
                        c_color = [1, 0, 0, 0.8] if is_contact_ui else [0, 1, 0, 0.2]
                        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [0.015, 0.015, 0.015], np.zeros(3), np.zeros(9), c_color)
                        viewer.user_scn.geoms[viewer.user_scn.ngeom].pos = data.site_xpos[tcp_id].copy()
                        viewer.user_scn.ngeom += 1

                    if cfg["obs"] and pillar_body_id != -1:
                        if data.xmat[pillar_body_id].reshape(3, 3)[2, 2] < 0.9: episode_col = True

                    screw_pos = data.xpos[target_body_id]
                    in_box = abs(screw_pos[0] - (-0.05)) < 0.12 and abs(screw_pos[1] - 0.0) < 0.12 and screw_pos[2] < 0.34 
                    is_released = (current_action[6] > 0.02) or (np.linalg.norm(current_tcp - screw_pos) > 0.06)

                    if in_box and is_released: in_box_counter += 1
                    else: in_box_counter = 0
                    if in_box_counter >= 5:
                        is_success = True
                        break
                    
                    viewer.sync()
                    step_counter += 1
                
                if is_success: success_count += 1
                if episode_col: collision_count += 1
                
                ep_max_t = np.max(episode_taus)
                total_max_tau += ep_max_t
                
                episode_peak_forces.append(ep_true_peak_f)
                all_impulses.append(ep_true_impulse)
                if is_success:
                    succ_impulses.append(ep_true_impulse)
                
                save_folder = os.path.join(base_record_dir, "success" if is_success else "fail")
                save_episode_video(video_frames, save_folder, episode + 1)
                
                data_filename = os.path.join(save_folder, f"data_ep_{episode+1:03d}.npz")
                np.savez(data_filename, 
                         steps=np.array(episode_data["steps"]),
                         min_dist=np.array(episode_data["min_dist"]),
                         influence=np.array(episode_data["influence"]),
                         is_apf_active=np.array(episode_data["is_apf_active"]),
                         tcp_pos=np.array(episode_data["tcp_pos"]),
                         q_dot=np.array(episode_data["q_dot"]),
                         q_pos=np.array(episode_data["q_pos"]),
                         q_vel=np.array(episode_data["q_vel"]),
                         q_acc=np.array(episode_data["q_acc"]),
                         q_tau=np.array(episode_data["q_tau"]),
                         f_z=np.array(episode_data["f_z"]),              
                         is_contact=np.array(episode_data["is_contact"]),
                         f_real_x=np.array(episode_data["f_real_x"]), 
                         f_real_y=np.array(episode_data["f_real_y"]), 
                         f_real_z=np.array(episode_data["f_real_z"]), 
                         q_pos_vla=np.array(episode_data["q_pos_vla"]),
                         q_dot_vla=np.array(episode_data["q_dot_vla"])) 
                
                print(f"  └─ 回合 {episode+1}/{args.num_episodes}: {'✅ 成功' if is_success else '❌ 失败'} | {'💥 撞倒' if episode_col else '🛡️ 绕开'} | 最大力矩: {ep_max_t:.2f}N·m | 峰值冲力: {ep_true_peak_f:.2f}N | 冲量: {ep_true_impulse:.2f}N·s")

            sr = (success_count / args.num_episodes) * 100
            cr = (collision_count / args.num_episodes) * 100
            avg_tau = total_max_tau / args.num_episodes
            
            avg_peak_f = np.mean(episode_peak_forces) if episode_peak_forces else 0.0
            avg_imp_all = np.mean(all_impulses) if all_impulses else 0.0
            avg_imp_succ = np.mean(succ_impulses) if succ_impulses else 0.0
            var_imp_succ = np.var(succ_impulses) if succ_impulses else 0.0
            std_imp_succ = np.std(succ_impulses) if succ_impulses else 0.0
            
            report_summary.append({
                "case_id": cfg["case_id"],
                "mode_str": cfg["mode_str"],
                "obs_str": cfg["str_obs"],
                "apf_str": cfg["str_apf"],
                "cont_str": cfg["str_cont"],
                "sr_str": f"{sr:.1f}%",
                "cr_str": f"{cr:.1f}%",
                "tau_str": f"{avg_tau:.2f} N·m",
                "peak_f_str": f"{avg_peak_f:.2f} N",
                "imp_all_str": f"{avg_imp_all:.2f}",
                "imp_succ_str": f"{avg_imp_succ:.2f} ± {std_imp_succ:.2f}",
                "imp_var_str": f"{var_imp_succ:.2f}"
            })

    summary_str = f"\n======================================= 📜 最终消融实验总结表 [多模态协同验证] =======================================\n"
    summary_str += "| 编号 | 力控策略 | 障碍物状态 | 宏观避障策略 | 末端接触保护 | 任务成功率(SR) | 碰撞率(CR) | 平均最大力矩 | 瞬时冲力峰值 | 冲量(全量) | 成功冲量(均值±标准差) | 成功冲量方差 |\n"
    summary_str += "| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |\n"
    for res in report_summary:
        summary_str += f"| Case {res['case_id']} | {res['mode_str']} | {res['obs_str']} | {res['apf_str']} | {res['cont_str']} | {res['sr_str']} | {res['cr_str']} | {res['tau_str']} | {res['peak_f_str']} | {res['imp_all_str']} | {res['imp_succ_str']} | {res['imp_var_str']} |\n"
    summary_str += "======================================================================================================================\n"

    print(summary_str)

    txt_path = os.path.join(BASE_DIR, "recordings", f"ablation_summary_all_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_str)
    print(f"📝 总结报告已同步保存至: {txt_path}\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="VLA 宏微观安全保护策略部署脚本")
    p.add_argument("--host", default="localhost", help="OpenPI WebSocket 服务器 IP")
    p.add_argument("--port", default=8000, type=int, help="OpenPI WebSocket 服务器端口")
    p.add_argument("--num_episodes", default=100, type=int, help="每个工况需要测试的次数（默认：100）")
    p.add_argument("--case", default=0, type=int, help="指定运行某个工况(1-8)。如果为0则按顺序运行全部8个工况。")
    p.add_argument("--fixed_eval", action="store_true", help="启用严格对照模式（每次生成的物体位置序列固定）")
    
    p.add_argument("--control_mode", type=str, default="both", choices=["admittance", "impedance", "both"], help="选择末端力控策略: admittance, impedance 或 both (自动去重执行完备的12组)")
    
    main(p.parse_args())