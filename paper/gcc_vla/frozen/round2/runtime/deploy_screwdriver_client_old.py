import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse
import os   
import json
import hashlib
from datetime import datetime
from tabletop_visuals import draw_box_targets
from tabletop_perception import GroundingDINOWrapper, localize_scene
from tabletop_pace_runtime import TabletopArm, TabletopAdmittance, save_rows, module_provenance
from tabletop_pace_preview import AGVF2Preview, AGVF2Config
from tabletop_box_attraction import AGVBoxAttraction, BoxAttractionConfig
from joint_torque_observer import (MujocoTorqueObserver, ForceEstimate, ForceLog,
    add_observer_arguments, config_from_args, select_control_force,
    robot_contact_truth, arm_body_ids)

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

# Legacy table-contact metrics are retained for historical analysis only.
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

def reset_scene(model, data, use_obstacle=False, target_xy=None):
    # Reset ALL physical state between episodes; controllers are per-episode objects.
    mujoco.mj_resetData(model, data)
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


def settle_pillar(model, data, pillar_joint_id, max_steps=500):
    """Advance physics until the reset pillar is stably resting on the table."""
    if pillar_joint_id == -1:
        return 0

    v_adr = model.jnt_dofadr[pillar_joint_id]
    stable_steps = 0
    for settle_step in range(1, max_steps + 1):
        mujoco.mj_step(model, data)
        linear_speed = np.linalg.norm(data.qvel[v_adr : v_adr + 3])
        angular_speed = np.linalg.norm(data.qvel[v_adr + 3 : v_adr + 6])
        if settle_step >= 100 and linear_speed < 1e-3 and angular_speed < 1e-3:
            stable_steps += 1
            if stable_steps >= 50:
                return settle_step
        else:
            stable_steps = 0
    return max_steps

# ==========================================
# 🚀 部署主程序
# ==========================================
def tool_ground_contact(model, data, tool_geom_ids):
    floor = model.geom('floor').id
    tool_geoms = set(map(int, tool_geom_ids))
    return any(c.dist <= 0 and ((c.geom1 == floor and c.geom2 in tool_geoms)
               or (c.geom2 == floor and c.geom1 in tool_geoms)) for c in data.contact)


def main(args):
    BASE_DIR = '/home/jun/dummyx-sim'  # unchanged read-only scene assets; isolated experiment executable
    if args.box_attraction == 'on' and args.pace_current_guard != 'on':
        raise ValueError('Box attraction requires current reference guard')
    if args.pace_current_guard == 'on' and args.pace_preserve_orientation != 'on':
        raise ValueError('Current reference guard requires orientation preservation')
    if args.contact_force_source != 'joint_torque':
        raise ValueError('Migrated admittance uses joint-torque estimates only; contact truth is evaluation-only')
    if args.num_episodes < 1 or args.max_steps < 1 or args.chunk_steps < 1:
        raise ValueError('Episode and step counts must be positive')
    if not np.isfinite(args.home_tolerance_rad) or args.home_tolerance_rad <= 0:
        raise ValueError('Home tolerance must be finite and positive')
    for name in ('pace_max_correction_m_s', 'box_attraction_speed_m_s', 'box_attraction_decay_power'):
        if not np.isfinite(getattr(args,name)) or getattr(args,name) <= 0:
            raise ValueError(name + ' must be finite and positive')
    output_root = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_root, exist_ok=True)
    xml_path = os.path.join(BASE_DIR, "models", "dummyx_apf_scene.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.noslip_iterations = args.grip_noslip_iterations
    paper_config = None
    if args.paper_experiment:
        from tabletop_paper_ablation import load_config, configure_model
        paper_config = load_config(args.paper_experiment)
        configure_model(model, paper_config)
    data = mujoco.MjData(model)
    renderer_rgb = mujoco.Renderer(model, height=256, width=256)
    for path in (args.groundingdino_config, args.groundingdino_checkpoint):
        if not path or not os.path.isfile(os.path.expanduser(path)):
            raise FileNotFoundError(f'GroundingDINO config/checkpoint required: {path}')
    detector = GroundingDINOWrapper(os.path.expanduser(args.groundingdino_config),
                                   os.path.expanduser(args.groundingdino_checkpoint), args.device)
    perception_renderer = mujoco.Renderer(model, height=480, width=640)
    if args.perception_probe:
        output_dir = os.path.join(output_root, 'groundingdino_probe', datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        try:
            reset_scene(model, data, use_obstacle=True, target_xy=(.28,-.10))
            settle_pillar(model,data,model.joint('pillar_joint').id)
            mujoco.mj_forward(model,data)
            localize_scene(model,data,perception_renderer,detector,output_dir,
                pillar_text=args.obstacle_text,box_text=args.box_text,
                box_threshold=args.box_threshold,text_threshold=args.text_threshold)
        finally:
            print(f'Perception artifacts: {output_dir}')
            perception_renderer.close(); renderer_rgb.close()
        return
    observer_config = config_from_args(args)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "real_screwdriver")
    pillar_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dynamic_pillar")
    pillar_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pillar_joint")
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
    
    torque_observer = MujocoTorqueObserver(model, tcp_id, observer_config)
    robot_bodies = arm_body_ids(model)

    link7_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    link8_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link8")
    link9_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link9")
    
    valid_collision_bodies = [target_body_id]
    if link7_id != -1: valid_collision_bodies.append(link7_id)
    if link8_id != -1: valid_collision_bodies.append(link8_id)
    if link9_id != -1: valid_collision_bodies.append(link9_id)

    all_configs = [
        {"case_id": 1, "obs": False, "apf": False, "contact": False, "tag_obs": "no_obs", "tag_apf": "no_apf", "tag_cont": "no_cont", "str_obs": "🟩无障碍", "str_apf": "🧠纯VLA", "str_cont": "❌无保护"},
        {"case_id": 2, "obs": False, "apf": True,  "contact": False, "tag_obs": "no_obs", "tag_apf": "apf",    "tag_cont": "no_cont", "str_obs": "🟩无障碍", "str_apf": "🛡️F2前瞻", "str_cont": "❌无保护"},
        {"case_id": 3, "obs": True,  "apf": False, "contact": False, "tag_obs": "obs",    "tag_apf": "no_apf", "tag_cont": "no_cont", "str_obs": "🔥有障碍", "str_apf": "🧠纯VLA", "str_cont": "❌无保护"},
        {"case_id": 4, "obs": True,  "apf": True,  "contact": False, "tag_obs": "obs",    "tag_apf": "apf",    "tag_cont": "no_cont", "str_obs": "🔥有障碍", "str_apf": "🛡️F2前瞻", "str_cont": "❌无保护"},
        {"case_id": 5, "obs": False, "apf": False, "contact": True,  "tag_obs": "no_obs", "tag_apf": "no_apf", "tag_cont": "cont",    "str_obs": "🟩无障碍", "str_apf": "🧠纯VLA", "str_cont": "✅虚拟墙"},
        {"case_id": 6, "obs": False, "apf": True,  "contact": True,  "tag_obs": "no_obs", "tag_apf": "apf",    "tag_cont": "cont",    "str_obs": "🟩无障碍", "str_apf": "🛡️F2前瞻", "str_cont": "✅虚拟墙"},
        {"case_id": 7, "obs": True,  "apf": False, "contact": True,  "tag_obs": "obs",    "tag_apf": "no_apf", "tag_cont": "cont",    "str_obs": "🔥有障碍", "str_apf": "🧠纯VLA", "str_cont": "✅虚拟墙"},
        {"case_id": 8, "obs": True,  "apf": True,  "contact": True,  "tag_obs": "obs",    "tag_apf": "apf",    "tag_cont": "cont",    "str_obs": "🔥有障碍", "str_apf": "🛡️F2前瞻", "str_cont": "✅双环保护"},
    ]
    
    if 1 <= args.case <= 8:
        configs = [all_configs[args.case - 1]]
        print(f"\n🎯 [精准测试模式] 将仅运行 Case {args.case}，测试 {args.num_episodes} 次。")
    else:
        configs = all_configs
        print(f"\n🚀 [全量消融模式] 将按顺序测试 8 个工况；接触保护仅使用导纳。")

    # 🚀🚀🚀 核心逻辑：智能构建测试队列 🚀🚀🚀
    test_queue = []
    for cfg in configs:
        if args.pace_preview == 'off' and cfg['apf']:
            cfg['str_apf'] = '当前状态避障'
        if not cfg["contact"]:
            # 无接触保护工况，只测一次
            c = cfg.copy()
            c['mode'] = 'none' 
            c['mode_str'] = '⚪ 无'
            test_queue.append(c)
        else:
            c = cfg.copy()
            c['mode'] = 'admittance'
            c['mode_str'] = '🔵 导纳(Adm)'
            test_queue.append(c)

    print(f"========================================================")
    print(f"⚙️  已构建智能测试队列: 共需运行 {len(test_queue)} 组消融实验")
    print(f"========================================================\n")

    report_summary = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    # 🚨🚨🚨 修改功能：将随机序列改为绝对定点 🚨🚨🚨
    fixed_targets = None
    if args.fixed_eval:
        print("🔒 [固定评测模式] 开启！各工况将使用完全相同的初始位置。")
        # 将原本的随机生成序列，改为生成 1 个绝对坐标并复制 args.num_episodes 次
        absolute_fixed_xy = (0.28, -0.10) # 设为一个易于 VLA 抓取的中心偏上位置
        fixed_targets = [absolute_fixed_xy for _ in range(args.num_episodes)]
        print(f"📌 [绝对静止] 螺丝刀坐标已焊死在: X={absolute_fixed_xy[0]}, Y={absolute_fixed_xy[1]}")
    elif args.eval_seed is not None:
        eval_rng = np.random.default_rng(args.eval_seed)
        fixed_targets = [
            (float(eval_rng.uniform(0.20, 0.35)), float(eval_rng.uniform(-0.15, -0.05)))
            for _ in range(args.num_episodes)
        ]
        print(f"🎲 [配对评测] 使用确定性初始位置序列: seed={args.eval_seed}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for cfg in test_queue:
            success_count, collision_count = 0, 0
            total_max_tau = 0.0
            evaluated_count = 0
            episode_summaries = []
            
            episode_peak_forces = [] 
            all_impulses = []
            succ_impulses = []
            all_safety_layer_times = []
            
            run_folder_name = f"run_{timestamp}_{cfg['mode']}_Case{cfg['case_id']}_{cfg['tag_obs']}_{cfg['tag_apf']}_{cfg['tag_cont']}_{args.obstacle_source}_{args.contact_force_source}"
            base_record_dir = os.path.join(output_root, run_folder_name)
            os.makedirs(base_record_dir, exist_ok=False)
            # Compiled model includes meshes/textures; one shared snapshot per case.
            model_snapshot = os.path.join(base_record_dir, "model.mjb")
            from tabletop_recording_storage import save_model_snapshot
            model_storage=save_model_snapshot(model,model_snapshot,args.model_cache)
            with open(model_snapshot, "rb") as f:
                model_sha256 = hashlib.sha256(f.read()).hexdigest()
            with open(__file__, "rb") as f:
                source_sha256 = hashlib.sha256(f.read()).hexdigest()
            with open(os.path.join(base_record_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(dict(schema_version=2, args=vars(args), case=cfg,
                               modules_sha256=module_provenance(),
                               box_geometry_source="groundingdino_rgbd; no known box pose or dimension fallback",
                               obstacle_geometry_source="groundingdino_rgbd; depth-estimated pillar dimensions",
                               paper_experiment=paper_config,
                               effective=dict(f2=cfg["apf"] and cfg["obs"] and (paper_config is None or paper_config['geometry']),
                                   attraction=args.box_attraction == "on" and (paper_config is not None or (cfg["apf"] and cfg["obs"])),
                                   admittance=cfg["contact"]),
                               mujoco_version=mujoco.__version__, model_sha256=model_sha256,model_storage=model_storage,
                               source_sha256=source_sha256, state_spec="mjSTATE_INTEGRATION",
                               frame_alignment="before control; terminal state stored separately",
                               prompt=args.task_prompt,
                               video="offline state rendering; no viewer overlays",
                               replay_timing="nominal 50ms frame period; inspect actual_control_duration_s and valid_substeps for aborted actions",
                               metric_note="min_dist=F2 current capsule clearance or NaN when not evaluated; influence=F2 clearance deficit in meters; f_z=legacy force norm; q_dot fields=legacy position deltas, not rad/s"),
                          f, ensure_ascii=False, indent=2)
            print(f"数据输出: {base_record_dir}")

            print("\n" + "="*80)
            print(f" 📊 [运行工况 Case {cfg['case_id']}] 策略: [{cfg['mode_str']}] | 障碍: [{cfg['str_obs']}] | APF: [{cfg['str_apf']}] | 保护: [{cfg['str_cont']}]")
            print("="*80)

            for episode in range(args.num_episodes):
                if not viewer.is_running(): break
                
                current_target_xy = fixed_targets[episode] if fixed_targets is not None else None
                reset_scene(model, data, use_obstacle=cfg["obs"], target_xy=current_target_xy)
                episode_initial_target_xy = data.xpos[target_body_id][:2].copy()
                if cfg["obs"] and (args.settle_before_start or detector is not None):
                    settle_pillar(model, data, pillar_jnt_id)
                viewer.sync()
                mujoco.mj_forward(model,data)
                perception = None
                if cfg['obs'] or cfg['contact'] or cfg['apf']:
                    perception_dir = os.path.join(base_record_dir, f'perception_ep_{episode+1:03d}')
                    try:
                        perception, _ = localize_scene(model,data,perception_renderer,detector,perception_dir,
                            need_pillar=cfg['obs'],pillar_text=args.obstacle_text,box_text=args.box_text,
                            box_threshold=args.box_threshold,text_threshold=args.text_threshold)
                    except Exception as exc:
                        failure=dict(episode=episode+1,success=False,reason='perception_failed: '+str(exc),
                            legacy_success=False,obstacle_contact=False,robot_table_contact=False,
                            pillar_tilted=False,control_steps=0,duration_s=0.,assistance='off')
                        episode_summaries.append(failure)
                        with open(os.path.join(base_record_dir,'episodes.json'),'w',encoding='utf-8') as f:
                            json.dump(episode_summaries,f,ensure_ascii=False,indent=2)
                        print(f'感知失败，未执行策略动作：{perception_dir}: {exc}')
                        # Stop this case; never substitute model geometry or continue unprotected.
                        break
                if perception is not None and args.box_attraction_target != 'center':
                    if 'pillar' not in perception:
                        raise ValueError('Front attraction target requires a DINO pillar detection')
                    box = perception['box']
                    box['original_attraction_point'] = np.asarray(box['attraction_point']).copy()
                    goal = box['original_attraction_point'].copy()
                    if args.box_attraction_target in ('front_pillar_aligned', 'front_pillar_above'):
                        goal[1] = (perception['pillar']['p1'][1] + perception['pillar']['p2'][1]) / 2.
                        rule = 'DINO box center X, DINO pillar center Y, original goal Z'
                        if args.box_attraction_target == 'front_pillar_above':
                            goal[2] = max(perception['pillar']['p1'][2], perception['pillar']['p2'][2]) + .05
                            rule = 'DINO box center X, DINO pillar center Y, DINO pillar top + 0.05m Z'
                    elif args.box_attraction_target == 'world_xy_zero':
                        goal[:2] = 0.
                        goal[2] = max(perception['pillar']['p1'][2], perception['pillar']['p2'][2]) + .05
                        rule = 'Explicit user target: world X=0, Y=0; DINO pillar top + 0.05m Z'
                    else:
                        goal[0] += .05
                        rule = 'AGV attraction offset: DINO box opening center + (0.05, 0, 0.05) m'
                    box['attraction_point'] = goal
                    box['attraction_target_rule'] = rule
                    perception_path = os.path.join(perception_dir, 'perception.json')
                    with open(perception_path, encoding='utf-8') as f:
                        recorded_perception = json.load(f)
                    recorded_perception['geometry']['box'].update(
                        original_attraction_point=box['original_attraction_point'].tolist(),
                        attraction_point=goal.tolist(),attraction_target_rule=box['attraction_target_rule'])
                    with open(perception_path, 'w', encoding='utf-8') as f:
                        json.dump(recorded_perception, f, ensure_ascii=False, indent=2)
                arm = TabletopArm(model, data)
                arm.robot_bodies = robot_bodies
                initial_height = float(data.xpos[target_body_id, 2])
                home = arm.state().copy()
                preview = attraction = None
                contact_controller = TabletopAdmittance(arm, perception=perception) if cfg['contact'] else None
                if cfg['apf'] and cfg['obs']:
                    capsule = {k: np.array(perception['pillar'][k], copy=True) for k in ('p1','p2','r')}
                    capsule['placement_goal'] = np.asarray(perception['box']['placement_goal']).copy()
                    preview = AGVF2Preview(arm, capsule, initial_height,
                        AGVF2Config(max_correction_m_s=args.pace_max_correction_m_s, lift_preference=args.pace_lift_preference),
                        distance_fn=segment_to_segment_distance, pickup_mode='latched',
                        joint_filter=args.pace_joint_filter,
                        preserve_orientation=args.pace_preserve_orientation == 'on',
                        current_guard=args.pace_current_guard == 'on',
                        preview_enabled=args.pace_preview == 'on',
                        safety_only=args.pace_safety_only == 'on',
                        route_preference=args.pace_route_preference in ('up_near','staged_up_near'),
                        staged_route=args.pace_route_preference == 'staged_up_near',
                        prediction_mode=args.pace_prediction_mode,tracking_model_path=args.pace_tracking_model,proxy_mode=args.pace_tool_proxy)
                    if args.box_attraction == 'on':
                        attraction = AGVBoxAttraction(arm, perception["box"]["attraction_point"], initial_height,
                            BoxAttractionConfig(maximum_speed_m_s=args.box_attraction_speed_m_s, decay_power=args.box_attraction_decay_power, forbid_outward=args.pace_route_preference == 'up_near',staged_route=args.pace_route_preference == 'staged_up_near',check_descent=args.pace_route_descent_check=='on',commit_transport=args.pace_route_commit=='on'))
                if attraction is not None and args.pace_route_preference in ('up_near','staged_up_near'):
                    from tabletop_route_preference import near_base_direction
                    attraction.near_direction=near_base_direction(data.body('root').xpos,perception['pillar']['p1'])
                    attraction.obstacle=perception['pillar']
                    attraction.tool_extent=preview.tool_extent_m.copy()
                    attraction.tool_radius=preview.tool_radius_m
                    attraction.proxy_mode=preview.proxy_mode
                    attraction.tool_capsules_local=preview.tool_capsules_local
                    attraction.eef_capsules_local=preview.eef_capsules_local
                if paper_config is not None:
                    from tabletop_paper_ablation import build_modules
                    preview, attraction = build_modules(arm, perception, initial_height, args, paper_config)
                    if attraction is not None:
                        # Persist the actual visual/stale target used by this experiment.
                        perception_path = os.path.join(perception_dir, 'perception.json')
                        perception['box']['attraction_point'] = attraction.goal.copy()
                        with open(perception_path, encoding='utf-8') as f:
                            recorded_perception = json.load(f)
                        recorded_perception['geometry']['box']['attraction_point'] = attraction.goal.tolist()
                        recorded_perception['geometry']['box']['paper_target_source'] = paper_config.get('target_reference_file', 'current RGB-D/explicit target rule')
                        with open(perception_path, 'w', encoding='utf-8') as f:
                            json.dump(recorded_perception, f, ensure_ascii=False, indent=2)
                torque_observer.reset(data)
                force_estimate = ForceEstimate(np.zeros(3), np.zeros(6), False)
                force_log = ForceLog()
                step_counter, in_box_counter = 0, 0
                is_success, episode_col = False, False
                last_guidance_phase = None
                action_chunk_cache = None
                chunk_index = 0
                reason = 'step_limit'
                strict_frames = 0
                legacy_success = False
                grasp_seen = False
                max_height = initial_height
                obstacle_contact = robot_table_contact = False
                tool_floor_contact = False
                floor_contact_time_s = None
                contact_rows, guard_rows, staged_targets = [], [], []
                paper_truth_rows, paper_quality_rows = [], []
                partial_execution = False
                simulation_errors = []
                state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
                replay_states, raw_actions, executed_ctrl = [], [], []
                sim_times, end_times, tool_positions, pillar_positions = [], [], [], []
                policy_times = []
                episode_start_time = float(data.time)
                episode_taus = []
                
                episode_data = {
                    "steps": [], "min_dist": [], "influence": [], "is_apf_active": [],
                    "apf_time_ms": [], "contact_controller_time_ms": [],
                    "safety_layer_time_ms": [], "intervention_norm": [],
                    "tcp_pos": [], "q_dot": [],
                    "q_pos": [], "q_vel": [], "q_acc": [], "q_tau": [],
                    "f_z": [], "is_contact": [],
                    "f_real_x": [], "f_real_y": [], "f_real_z": [], 
                    "q_pos_vla": [], "q_dot_vla": [] 
                }
                
                ep_true_peak_f = 0.0
                ep_true_impulse = 0.0

                while viewer.is_running() and step_counter < args.max_steps:
                    if action_chunk_cache is None or chunk_index >= min(args.chunk_steps, len(action_chunk_cache)) or (preview is not None and preview.refresh_pending):
                        renderer_rgb.update_scene(data, camera="rear_cam")
                        img_f = renderer_rgb.render()
                        renderer_rgb.update_scene(data, camera="wrist_cam")
                        img_w = renderer_rgb.render()
                        infer_t0 = time.perf_counter()
                        try:
                            result = policy.infer({"observation/image": img_f, "observation/wrist_image": img_w, "observation/state": data.qpos[:8].copy(), "prompt": args.task_prompt})
                            action_chunk_cache = np.asarray(result['actions'], dtype=float)
                            internal_chunk = arm.policy_chunk(action_chunk_cache)
                            chunk_index = 0
                            if preview is not None:preview.refresh_pending=False
                        except KeyboardInterrupt:
                            reason = 'user_interrupt'
                            break
                        except Exception as exc:
                            reason = 'policy_error: ' + str(exc)
                            break
                        policy_times.append((step_counter, (time.perf_counter() - infer_t0) * 1000))
                    
                    state = np.empty(mujoco.mj_stateSize(model, state_spec))
                    mujoco.mj_getState(model, data, state, state_spec)
                    replay_states.append(state)
                    sim_times.append(float(data.time))

                    current_action = action_chunk_cache[chunk_index]
                    raw_actions.append(current_action.copy())
                    active_capsules, obs_capsule, is_apf_active = [], None, False
                    
                    current_tcp = data.site_xpos[tcp_id].copy()
                    raw_q_pos_vla = current_action[:6].copy()
                    raw_q_dot_vla = raw_q_pos_vla - data.qpos[:6].copy()
                    
                    step_min_dist = np.nan
                    step_influence = 0.0
                    
                    geometry_started = time.perf_counter_ns()
                    nominal_target = internal_chunk[chunk_index].copy()
                    geometry_input = nominal_target.copy()
                    guarded_chunk = internal_chunk.copy()
                    if attraction is not None:
                        geometry_input = attraction.process(geometry_input, step_counter)
                        guarded_chunk = attraction.guide_chunk(guarded_chunk)
                        if attraction.config.staged_route:
                            if preview is not None:
                                preview.route_phase=attraction.phase
                            if attraction.phase != last_guidance_phase:
                                print(f"Guidance phase: {attraction.phase}; remaining proxy-path clearance={attraction.rows[-1]['route_min_clearance_m']:.4f} m", flush=True)
                                last_guidance_phase=attraction.phase
                    if preview is not None:
                        base_target, debug_info = preview.process(guarded_chunk, chunk_index,
                            execution_stop=min(args.chunk_steps,len(guarded_chunk)), action_index=step_counter)
                        step_min_dist = debug_info['current_clearance_m']
                        step_influence = debug_info['deficit_m']
                        is_apf_active = debug_info['geometry_active']
                        step_intervention_norm = float(np.linalg.norm(base_target[:6]-nominal_target[:6]))
                        step_apf_time_ms = (time.perf_counter_ns()-geometry_started)/1e6
                    else:
                        base_target = geometry_input.copy()
                        step_apf_time_ms = 0.
                        step_intervention_norm = 0.
                    base_ctrl = arm.control(base_target)
                    round2_trace = globals().get('ROUND2_TRACE')
                    staged_targets.append(dict(action_index=step_counter,policy_target=nominal_target.copy(),
                        before_geometry_target=geometry_input.copy(),geometry_target=base_target.copy()))

                    is_contact_ui = False
                    max_f_norm_step = 0.0  
                    max_f_xyz_step = np.zeros(3) 
                    contact_compute_ns = 0
                    
                    substep_controls = []
                    current_tau = np.zeros(6)
                    for substep in range(50):
                        contact_t0 = time.perf_counter_ns()
                        current_f_xyz = (contact_controller.latest.force.copy()
                                         if contact_controller is not None else np.zeros(3))
                        # Legacy experiment metrics only; never fed back in joint_torque mode.
                        legacy_truth = get_target_table_force(model, data, valid_collision_bodies)
                        current_force_norm = np.linalg.norm(legacy_truth)
                        contact_t0 = time.perf_counter_ns()
                        
                        if current_force_norm > max_f_norm_step:
                            max_f_norm_step = current_force_norm
                            max_f_xyz_step = legacy_truth.copy()

                        if current_force_norm > ep_true_peak_f:
                            ep_true_peak_f = current_force_norm

                        contact_debug = guard_debug = None
                        final_target = base_target.copy()
                        if contact_controller is not None:
                            final_target, contact_debug = contact_controller.process(base_target)
                            is_contact_ui |= bool(contact_debug['contact_gate'])
                            current_f_xyz = contact_debug['feedback_force_N'].copy()
                        round2_contact_target = final_target.copy()
                        if preview is not None and preview.current_guard:
                            final_target, guard_debug = preview.project_current_target(final_target)
                            guard_rows.append(dict(action_index=step_counter, substep_index=substep,
                                time_s=float(data.time), **guard_debug))
                            if not guard_debug['reference_guard_feasible']:
                                reason = ('safety_abort: up_near reference constraint infeasible' if args.pace_route_preference == 'up_near' else 'safety_abort: final reference constraint infeasible')
                                partial_execution = True
                                break
                        round2_post_guard_target = final_target.copy()
                        round2_limiter_debug = None
                        round2_limit = globals().get('ROUND2_JUMP_LIMIT')
                        round2_previous = globals().get('ROUND2_PREVIOUS_TARGET')
                        if round2_limit is not None and round2_previous is not None:
                            final_target = final_target.copy()
                            final_target[:6] = np.clip(final_target[:6], round2_previous[:6]-round2_limit,
                                                     round2_previous[:6]+round2_limit)
                            round2_limiter_debug = dict(enabled=True, threshold_rad=round2_limit.copy())
                        current_ctrl = arm.control(final_target)
                        contact_compute_ns += time.perf_counter_ns() - contact_t0

                        ep_true_impulse += current_force_norm * model.opt.timestep
                        substep_controls.append(current_ctrl.copy())
                        data.ctrl[:8] = current_ctrl
                        round2_row = None
                        if round2_trace is not None:
                            round2_row = round2_trace.before_step(arm, step_counter, substep,
                                nominal_target=nominal_target, geometry_target=base_target,
                                contact_target=round2_contact_target, final_target=round2_post_guard_target,
                                applied_target=final_target, geometry_debug=debug_info if preview is not None else None,
                                contact_debug=contact_debug, guard_debug=guard_debug,
                                previous_applied=round2_previous, limiter_debug=round2_limiter_debug)
                        time_before_step = float(data.time)
                        try:
                            mujoco.mj_step(model, data)
                            observer_t0 = time.perf_counter_ns()
                            force_estimate = torque_observer.sample_after_step(data)
                        except RuntimeError as exc:
                            reason = 'simulation_error: ' + str(exc)
                            partial_execution = True
                            simulation_errors.append(dict(action_index=step_counter,substep_index=substep,
                                time_before_s=time_before_step,time_after_s=float(data.time),
                                observer_previous_time_s=float(torque_observer.previous_time),
                                timestep_s=float(model.opt.timestep),warning_counts=data.warning.number.tolist(),
                                qpos=data.qpos.tolist(),qvel=data.qvel.tolist(),reason=str(exc)))
                            break
                        if contact_controller is not None:
                            contact_controller.after_step(force_estimate, contact_debug, float(data.time))
                            contact_rows.append(dict(action_index=step_counter, substep_index=substep,
                                time_s=float(data.time), executed_target=final_target.copy(), **contact_debug))
                        contact_compute_ns += time.perf_counter_ns() - observer_t0
                        if round2_trace is not None:
                            round2_trace.after_step(arm, round2_row, estimate=force_estimate, contact_debug=contact_debug)
                        globals()['ROUND2_PREVIOUS_TARGET'] = final_target.copy()
                        truth = robot_contact_truth(model, data, robot_bodies)
                        if paper_config is not None:
                            from tabletop_paper_ablation import contact_evaluation
                            paper_truth_rows.append(contact_evaluation(model, data, robot_bodies | {target_body_id}))
                        force_log.append(data.time - model.opt.timestep, force_estimate,
                                         truth, current_f_xyz, observer_config.enter_force)

                        current_tau = data.qfrc_actuator[:6].copy()
                        # Refresh kinematics/contact geometry only AFTER observer sampling.
                        mujoco.mj_forward(model, data)
                        metric = arm.placement()
                        obstacle_contact |= metric['obstacle_contact'] if cfg['obs'] else False
                        robot_table_contact |= metric['robot_table_contact']
                        grasp_seen |= metric['finger_contacts'] == 2
                        max_height = max(max_height, float(data.xpos[target_body_id,2]))
                        if round2_trace is not None:
                            round2_trace.after_forward(arm, round2_row, metric=metric,
                                phase_metric=dict(grasp_seen=grasp_seen, strict_frames=strict_frames,
                                    tool_floor_contact=tool_floor_contact, pillar_contact_episode=obstacle_contact))
                        episode_taus.append(np.max(np.abs(current_tau)))
                        if tool_ground_contact(model, data, arm.target_geoms):
                            tool_floor_contact = True
                            floor_contact_time_s = float(data.time - episode_start_time)
                            reason = 'tool_floor_contact'
                            partial_execution = True
                            break
                    # Fixed storage shape plus valid count preserves partial/aborted actions.
                    valid_substeps = len(substep_controls)
                    substep_controls += [np.full(8,np.nan)] * (50-valid_substeps)
                    executed_ctrl.append(substep_controls)
                    if valid_substeps == 0:
                        current_ctrl = base_ctrl.copy()
                        current_tau = np.zeros(6)
                    end_times.append(float(data.time))
                    tool_positions.append(data.xpos[target_body_id].copy())
                    pillar_positions.append(data.xpos[pillar_body_id].copy())
                    step_contact_controller_ms = contact_compute_ns / 1e6
                    if preview is not None or contact_controller is not None or attraction is not None:
                        step_safety_layer_ms = step_apf_time_ms + step_contact_controller_ms
                        all_safety_layer_times.append(step_safety_layer_ms)
                    else:
                        step_safety_layer_ms = np.nan
                    
                    final_step_q_dot = current_ctrl[:6] - data.qpos[:6].copy()

                    episode_data["steps"].append(step_counter)
                    episode_data["min_dist"].append(step_min_dist)
                    episode_data["influence"].append(step_influence)
                    episode_data["is_apf_active"].append(is_apf_active)
                    episode_data["apf_time_ms"].append(step_apf_time_ms)
                    episode_data["contact_controller_time_ms"].append(step_contact_controller_ms)
                    episode_data["safety_layer_time_ms"].append(step_safety_layer_ms)
                    episode_data["intervention_norm"].append(step_intervention_norm)
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
                    if preview is not None:
                        obs_capsule = preview.obstacle
                        active_capsules = preview.capsules(data)
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

                    if perception is not None:
                        draw_box_targets(viewer.user_scn, perception['box']['opening_center'],
                            perception['box']['attraction_point'], enabled=attraction is not None,
                            active=bool(attraction is not None and attraction.rows and attraction.rows[-1]['active']),
                            origin=(data.xpos[model.body('link6_1').id]+data.site_xpos[tcp_id])/2)

                    if cfg["obs"] and pillar_body_id != -1:
                        if data.xmat[pillar_body_id].reshape(3, 3)[2, 2] < 0.9: episode_col = True

                    screw_pos = data.xpos[target_body_id]
                    in_box = abs(screw_pos[0] - (-0.05)) < 0.12 and abs(screw_pos[1] - 0.0) < 0.12 and screw_pos[2] < 0.34 
                    is_released = (current_action[6] > 0.02) or (np.linalg.norm(current_tcp - screw_pos) > 0.06)

                    if in_box and is_released: in_box_counter += 1
                    else: in_box_counter = 0
                    legacy_success |= in_box_counter >= 5
                    metric = arm.placement()
                    home_error = float(np.max(np.abs(arm.state()[:6]-home[:6])))
                    if paper_config is not None:
                        paper_quality_rows.append(dict(time_s=float(data.time), valid_substeps=valid_substeps,
                            fully_inside=metric['fully_inside'], nonbox_support=metric['nonbox_support'],
                            tool_center_error_m=metric['tool_center_error_m'], tool_bounds_local=metric['tool_bounds_local'],
                            home_error_rad=home_error, placement_valid=metric['valid']))
                    strict_valid = (not tool_floor_contact and metric['valid'] and grasp_seen and max_height >= initial_height+.05)
                    if args.success_criterion == 'placement_and_home':
                        strict_valid = strict_valid and home_error <= args.home_tolerance_rad
                    strict_frames = strict_frames+1 if strict_valid and valid_substeps == 50 else 0
                    is_success = strict_frames >= int(np.ceil(.5/(50*model.opt.timestep)))
                    chunk_index += 1
                    step_counter += 1
                    if partial_execution:
                        break
                    if is_success:
                        reason = 'success'
                        break
                    viewer.sync()
                if not viewer.is_running() and reason == 'step_limit':
                    reason = 'viewer_closed'
                if reason not in ('viewer_closed','user_interrupt'):
                    evaluated_count += 1
                if is_success: success_count += 1
                if episode_col: collision_count += 1
                
                ep_max_t = np.max(episode_taus) if episode_taus else 0.0
                total_max_tau += ep_max_t
                
                episode_peak_forces.append(ep_true_peak_f)
                all_impulses.append(ep_true_impulse)
                if is_success:
                    succ_impulses.append(ep_true_impulse)
                
                save_folder = os.path.join(base_record_dir, "success" if is_success else "fail")
                os.makedirs(save_folder, exist_ok=True)
                
                data_filename = os.path.join(save_folder, f"data_ep_{episode+1:03d}.npz")
                terminal_state = np.empty(mujoco.mj_stateSize(model, state_spec))
                mujoco.mj_getState(model, data, terminal_state, state_spec)
                save_rows(os.path.join(save_folder, f'stages_ep_{episode+1:03d}.npz'), staged_targets)
                save_rows(os.path.join(save_folder, f'contact_ep_{episode+1:03d}.npz'), contact_rows)
                save_rows(os.path.join(save_folder, f'guard_ep_{episode+1:03d}.npz'), guard_rows)
                save_rows(os.path.join(save_folder, f'paper_truth_ep_{episode+1:03d}.npz'), paper_truth_rows)
                save_rows(os.path.join(save_folder, f'paper_quality_ep_{episode+1:03d}.npz'), paper_quality_rows)
                controllers = dict(preview=preview, attraction=attraction)
                metadata = {}
                for name, controller in controllers.items():
                    if controller is not None:
                        np.savez_compressed(os.path.join(save_folder, f'{name}_ep_{episode+1:03d}.npz'), **controller.arrays())
                        metadata[name] = controller.metadata()
                        if name=='preview' and controller.execution_predictor is not None:
                            arrays=controller.execution_predictor.arrays()
                            if arrays:np.savez_compressed(os.path.join(save_folder,f'prediction_ep_{episode+1:03d}.npz'),**arrays)
                if contact_controller is not None:
                    metadata['admittance'] = contact_controller.metadata()
                metadata['tool_floor_contact'] = tool_floor_contact
                metadata['floor_contact_time_s'] = floor_contact_time_s
                metadata['floor_termination'] = 'Tool collision geom contacts floor with dist <= 0; checked every 1ms physics substep; immediate failure'
                metadata['reason'] = reason
                metadata['simulation_errors'] = simulation_errors
                with open(os.path.join(save_folder, f'metadata_ep_{episode+1:03d}.json'), 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                episode_summary = dict(episode=episode+1, success=bool(is_success), reason=reason,
                    tool_floor_contact=bool(tool_floor_contact), floor_contact_time_s=floor_contact_time_s,
                    legacy_success=bool(legacy_success), obstacle_contact=bool(obstacle_contact),
                    robot_table_contact=bool(robot_table_contact), pillar_tilted=bool(episode_col),
                    control_steps=step_counter, duration_s=float(data.time-episode_start_time),
                    assistance='box_attraction' if attraction is not None else 'off')
                episode_summaries.append(episode_summary)
                with open(os.path.join(base_record_dir, 'episodes.json'), 'w', encoding='utf-8') as f:
                    json.dump(episode_summaries, f, ensure_ascii=False, indent=2)
                np.savez_compressed(data_filename,
                         schema_version=np.array(2),
                         valid_substeps=np.sum(np.isfinite(np.asarray(executed_ctrl)[:,:,0]),axis=1) if executed_ctrl else np.zeros(0,dtype=int),
                         episode_tool_floor_contact=np.array(tool_floor_contact),
                         floor_contact_time_s=np.array(floor_contact_time_s if floor_contact_time_s is not None else np.nan),
                         episode_legacy_success=np.array(legacy_success),
                         episode_obstacle_contact=np.array(obstacle_contact),
                         episode_robot_table_contact=np.array(robot_table_contact),
                         success_criterion=np.array(args.success_criterion),
                         replay_state=np.asarray(replay_states).reshape(-1, terminal_state.size),
                         terminal_state=terminal_state,
                         replay_state_spec=np.array(int(state_spec)),
                         sim_time=np.array(sim_times),
                         end_sim_time=np.array(end_times),
                         episode_start_time=np.array(episode_start_time),
                         episode_duration_s=np.array(data.time - episode_start_time),
                         termination_reason=np.array(reason),
                         raw_action=np.asarray(raw_actions).reshape(-1,action_chunk_cache.shape[1] if action_chunk_cache is not None and action_chunk_cache.ndim == 2 else 7),
                         executed_ctrl=np.asarray(executed_ctrl).reshape(-1,50,8),
                         actual_control_duration_s=np.asarray(end_times)-np.asarray(sim_times),
                         screwdriver_pos=np.array(tool_positions),
                         pillar_pos=np.array(pillar_positions),
                         policy_inference_step_ms=np.asarray(policy_times).reshape(-1, 2),
                         **force_log.arrays(),
                         contact_force_source=np.array(args.contact_force_source),
                         force_observer_config=np.array(json.dumps(vars(observer_config))),
                         force_sample_period_s=np.array(model.opt.timestep),
                         episode_success=np.array(is_success),
                         episode_knockdown=np.array(episode_col),
                         episode_peak_contact_force=np.array(ep_true_peak_f),
                         episode_contact_impulse=np.array(ep_true_impulse),
                         episode_peak_joint_torque=np.array(ep_max_t),
                         control_period_s=np.array(50 * model.opt.timestep),
                         episode_initial_target_xy=episode_initial_target_xy,
                         obstacle_source=np.array(args.obstacle_source),
                         steps=np.array(episode_data["steps"]),
                         min_dist=np.array(episode_data["min_dist"]),
                         influence=np.array(episode_data["influence"]),
                         is_apf_active=np.array(episode_data["is_apf_active"]),
                         apf_time_ms=np.array(episode_data["apf_time_ms"]),
                         contact_controller_time_ms=np.array(episode_data["contact_controller_time_ms"]),
                         safety_layer_time_ms=np.array(episode_data["safety_layer_time_ms"]),
                         intervention_norm=np.array(episode_data["intervention_norm"]),
                         tcp_pos=np.array(episode_data["tcp_pos"]),
                         nominal_joint_velocity_rad_s=(np.array(episode_data["q_dot_vla"])/(50*model.opt.timestep)),
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
                
                latency_text = ""
                ep_latency = np.asarray(episode_data["safety_layer_time_ms"], dtype=float)
                ep_latency = ep_latency[np.isfinite(ep_latency)]
                if len(ep_latency):
                    latency_text = f" | safety={np.mean(ep_latency):.3f}ms, P95={np.percentile(ep_latency, 95):.3f}ms"
                print(f"  └─ 回合 {episode+1}/{args.num_episodes}: {'✅ 成功' if is_success else '❌ 失败'} | {'💥 撞倒' if episode_col else '🛡️ 绕开'} | 最大力矩: {ep_max_t:.2f}N·m | 峰值冲力: {ep_true_peak_f:.2f}N | 冲量: {ep_true_impulse:.2f}N·s{latency_text}")
                if reason == 'user_interrupt':
                    break

            evaluated = [r for r in episode_summaries if r['reason'] not in ('viewer_closed','user_interrupt')]
            sr = 100*sum(r['success'] for r in evaluated)/len(evaluated) if evaluated else np.nan
            cr = 100*sum(r['pillar_tilted'] for r in evaluated)/len(evaluated) if evaluated and cfg['obs'] else np.nan
            avg_tau = total_max_tau / len(episode_summaries) if episode_summaries else np.nan
            with open(os.path.join(base_record_dir, 'summary.json'), 'w', encoding='utf-8') as f:
                json.dump(dict(requested=args.num_episodes, recorded=len(episode_summaries), evaluated=len(evaluated),
                    successes=sum(r['success'] for r in evaluated),
                    obstacle_contact_evaluated=sum(r['duration_s']>0 for r in evaluated) if cfg['obs'] else 0,
                    obstacle_contact_episodes=sum(r['obstacle_contact'] for r in evaluated if r['duration_s']>0) if cfg['obs'] and any(r['duration_s']>0 for r in evaluated) else None,
                    pillar_tilt_episodes=sum(r['pillar_tilted'] for r in evaluated if r['duration_s']>0) if cfg['obs'] and any(r['duration_s']>0 for r in evaluated) else None,
                    perception_failures=sum(r['reason'].startswith('perception_failed') for r in evaluated),
                    safety_aborts=sum(r['reason'].startswith('safety_abort') for r in evaluated),
                    denominator='recorded episodes excluding viewer_closed/user_interrupt; safety aborts and policy errors count as failures'), f, indent=2)
            
            avg_peak_f = np.mean(episode_peak_forces) if episode_peak_forces else 0.0
            avg_imp_all = np.mean(all_impulses) if all_impulses else 0.0
            avg_imp_succ = np.mean(succ_impulses) if succ_impulses else 0.0
            var_imp_succ = np.var(succ_impulses) if succ_impulses else 0.0
            std_imp_succ = np.std(succ_impulses) if succ_impulses else 0.0
            latency = np.asarray(all_safety_layer_times, dtype=float)
            latency_mean = float(np.mean(latency)) if len(latency) else np.nan
            latency_p95 = float(np.percentile(latency, 95)) if len(latency) else np.nan
            latency_p99 = float(np.percentile(latency, 99)) if len(latency) else np.nan
            deadline_ms = 50 * model.opt.timestep * 1000.0
            deadline_miss = float(100.0 * np.mean(latency > deadline_ms)) if len(latency) else np.nan
            
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
                "imp_var_str": f"{var_imp_succ:.2f}",
                "latency_str": (f"{latency_mean:.3f} / {latency_p95:.3f} / "
                                f"{latency_p99:.3f} ms" if len(latency) else "—"),
                "deadline_str": f"{deadline_miss:.2f}%" if len(latency) else "—",
            })

    summary_str = f"\n======================================= 📜 最终消融实验总结表 [多模态协同验证] =======================================\n"
    summary_str += f"Obstacle source: {args.obstacle_source}\n"
    summary_str += f"Contact force source: {args.contact_force_source}; observer: {vars(observer_config)}\n"
    summary_str += "Latency scope: geometry including box attraction + 50 admittance/reference-guard substeps and observer sampling; excludes policy, rendering, perception, truth evaluation and physics.\n"
    summary_str += "Success: strict stable placement for >=0.5s; legacy success saved separately. Tilt is not contact collision. Box attraction and obstacle geometry use GroundingDINO + RGB-D only; no pose fallback or task takeover.\n"
    summary_str += "| 编号 | 力控策略 | 障碍物状态 | 宏观避障策略 | 末端接触保护 | 任务成功率(SR) | 柱体倾倒率 | 平均最大力矩 | 瞬时冲力峰值 | 冲量(全量) | 成功冲量(均值±标准差) | 成功冲量方差 | 安全算法总延迟 mean/P95/P99 | 50ms超时率 |\n"
    summary_str += "| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |\n"
    for res in report_summary:
        summary_str += f"| Case {res['case_id']} | {res['mode_str']} | {res['obs_str']} | {res['apf_str']} | {res['cont_str']} | {res['sr_str']} | {res['cr_str']} | {res['tau_str']} | {res['peak_f_str']} | {res['imp_all_str']} | {res['imp_succ_str']} | {res['imp_var_str']} | {res['latency_str']} | {res['deadline_str']} |\n"
    summary_str += "======================================================================================================================\n"

    print(summary_str)

    txt_path = os.path.join(output_root, f"ablation_summary_all_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_str)
    print(f"📝 总结报告已同步保存至: {txt_path}\n")
    renderer_rgb.close()
    if perception_renderer is not None:
        perception_renderer.close()

def build_argument_parser():
    p = argparse.ArgumentParser(description="桌面 F2 前瞻、箱体吸引和力确认导纳部署")
    p.add_argument("--output_dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs", "screwdriver_eval"), help="压缩数据和报告的输出根目录")
    add_observer_arguments(p)
    p.add_argument("--host", default="localhost", help="OpenPI WebSocket 服务器 IP")
    p.add_argument("--port", default=8000, type=int, help="OpenPI WebSocket 服务器端口")
    p.add_argument("--num_episodes", default=100, type=int, help="每个工况需要测试的次数（默认：100）")
    p.add_argument("--case", default=0, type=int, help="指定运行某个工况(1-8)。如果为0则按顺序运行全部8个工况。")
    p.add_argument("--fixed_eval", action="store_true", help="启用严格对照模式（每次生成的物体位置序列固定）")
    p.add_argument("--eval_seed", type=int, default=None,
                   help="生成可复现的多位置评测序列；两套算法使用相同 seed 可逐回合配对")
    p.add_argument("--settle_before_start", action="store_true", help="复位后等待障碍支柱稳定，再开始感知、推理和录像")
    p.add_argument("--obstacle_source", choices=["groundingdino"], default="groundingdino",
                   help="仅支持 GroundingDINO+RGB-D；不提供已知位置回退")
    p.add_argument("--perception_probe", action="store_true", help="仅检查柱体和箱体的 Grounding DINO RGB-D 感知并保存结果，不连接 OpenPI")
    p.add_argument("--groundingdino_config", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "GroundingDINO", "groundingdino/config/GroundingDINO_SwinT_OGC.py"), help="Grounding DINO 配置文件路径")
    p.add_argument("--groundingdino_checkpoint", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "GroundingDINO", "weights/groundingdino_swint_ogc.pth"), help="Grounding DINO 权重文件路径")
    p.add_argument("--obstacle_text", default="red pillar.", help="Grounding DINO 检测文字描述")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--box_threshold", type=float, default=0.35)
    p.add_argument("--text_threshold", type=float, default=0.25)
    
    p.add_argument('--control_mode', choices=['admittance'], default='admittance', help='只保留导纳，阻抗已移除')
    p.add_argument('--paper_experiment', default=None, help='Frozen paper ablation JSON; default execution profile unchanged')
    p.add_argument('--model_cache', default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'inference_results', 'model_cache'))
    p.add_argument('--max_steps', type=int, default=300, help='300 control steps = 15 seconds simulation time')
    p.add_argument('--chunk_steps', type=int, default=50, help='Number of predicted actions executed before the next policy request')
    p.add_argument('--pace_preview', choices=['on','off'], default='off', help='off 仅关闭安全层动作块前瞻，保留当前状态避障和原策略请求频率')
    p.add_argument('--pace_prediction_mode', choices=['legacy','execution','execution_trusted','execution_carry','execution_refresh','execution_lift','execution_lift_hold'], default='execution_lift')
    p.add_argument('--pace_tool_proxy', choices=['enclosing','components'], default='enclosing')
    p.add_argument('--pace_tracking_model', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tabletop_tracking_model.json'))
    p.add_argument('--pace_preserve_orientation', choices=['on','off'], default='on')
    p.add_argument('--pace_current_guard', choices=['on','off'], default='on')
    p.add_argument('--pace_joint_filter', choices=['on','off','correction_only'], default='on')
    p.add_argument('--pace_route_commit', choices=['on','off'], default='on')
    p.add_argument('--pace_route_descent_check', choices=['on','off'], default='off', help='off checks crossing before transport and leaves descent to local guard')
    p.add_argument('--pace_route_preference', choices=['off','up_near','staged_up_near'], default='off')
    p.add_argument('--pace_lift_preference', type=float, default=1.)
    p.add_argument('--pace_safety_only', choices=['on','off'], default='off')
    p.add_argument('--grip_noslip_iterations', type=int, choices=range(21), default=0)
    p.add_argument('--task_prompt', default='Pick up the screwdriver and drop it into the box.')
    p.add_argument('--pace_max_correction_m_s', type=float, default=.6)
    p.add_argument('--box_attraction', choices=['on','off'], default='on', help='GroundingDINO+RGB-D 箱体吸引；无已知位置回退或任务接管')
    p.add_argument('--box_attraction_speed_m_s', type=float, default=1.2)
    p.add_argument('--box_attraction_target', choices=['center','front_pillar_aligned','front_pillar_above','world_xy_zero','agv_offset'], default='center', help='DINO derived target: box center or box front aligned with pillar Y')
    p.add_argument('--box_attraction_decay_power', type=float, default=1., help='Positive distance decay exponent; 2 gives stronger far-field decay')
    p.add_argument('--box_text', default='open brown box.', help='箱体 DINO 提示词')
    p.add_argument('--success_criterion', choices=['placement','placement_and_home'], default='placement')
    p.add_argument('--home_tolerance_rad', type=float, default=.1)

    return p


if __name__ == "__main__":
    main(build_argument_parser().parse_args())
