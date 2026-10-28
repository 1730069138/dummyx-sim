"""Explicit paper-only factor switches; no policy calls or known target lookup.

Model position changes define the experimental scene. Controllers receive only
the subsequent DINO/RGB-D geometry. Simulator truth remains evaluation-only.
"""
from pathlib import Path
import json
import numpy as np
import mujoco
from tabletop_pace_preview import AGVF2Preview, AGVF2Config
from tabletop_box_attraction import AGVBoxAttraction, BoxAttractionConfig


def load_config(path):
    config = json.loads(Path(path).read_text())
    for key in ('geometry', 'preview', 'attraction', 'contact', 'pillar'):
        if type(config.get(key)) is not bool:
            raise ValueError('Explicit boolean factor required: ' + key)
    if config['preview'] and not config['geometry']:
        raise ValueError('Preview requires the geometry execution branch')
    if (config['geometry'] or config['attraction']) and not config['pillar']:
        raise ValueError('This matrix uses obstacle-free rows only as raw/contact baselines')
    offset = np.asarray(config.get('box_offset_xy', [0., 0.]), float)
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError('Invalid scene offset')
    return config


def configure_model(model, config):
    # Scene generation, never a controller target. Applied before MjData creation.
    model.body_pos[model.body('drop_box').id, :2] += config.get('box_offset_xy', [0., 0.])


def contact_evaluation(model, data, protected_bodies):
    """Evaluation only: robot+tool external contact loads, no internal grip load.

    Sum of individual force norms prevents opposite contacts cancelling. Net
    vectors, individual peaks and contact counts are retained separately.
    """
    total = task = peak = task_peak = 0.
    net = np.zeros(3); task_net = np.zeros(3); count = task_count = 0
    wrench = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        first = int(model.geom_bodyid[c.geom1]) in protected_bodies
        second = int(model.geom_bodyid[c.geom2]) in protected_bodies
        if first == second:
            continue
        mujoco.mj_contactForce(model, data, i, wrench)
        force = (1 if second else -1)*(c.frame.reshape(3, 3).T@wrench[:3])
        amount = float(np.linalg.norm(force)); total += amount; peak = max(peak, amount)
        net += force; count += 1
        if .18 < c.pos[2] < .22:
            task += amount; task_peak = max(task_peak, amount); task_net += force; task_count += 1
    return dict(time_s=float(data.time), all_sum_force_N=total, task_sum_force_N=task,
                all_peak_individual_N=peak, task_peak_individual_N=task_peak,
                all_net_force_N=net, task_net_force_N=task_net,
                all_contacts=count, task_contacts=task_count)


class NominalTargetPredictor:
    """Only the forecasting model changes; the risk-triggered lift stays fixed."""
    def __init__(self, owner):
        self.owner = owner
        self.data = mujoco.MjData(owner.arm.model)
        self.rows = []

    def forecast(self, chunk, index, stop, tool_local, first_target=None, record=True):
        o = self.owner
        self.data.qpos[:] = o.arm.data.qpos
        out = []
        for j in range(index, stop):
            target = first_target if j == index and first_target is not None else chunk[j]
            self.data.qpos[o.arm.qadr] = target[:6]
            self.data.qpos[o.gripper_qadr] = np.clip(target[6] / 2., 0., .04)
            mujoco.mj_kinematics(o.arm.model, self.data)
            mujoco.mj_comPos(o.arm.model, self.data)
            clearance, normal = o.clearance(o.capsules(self.data, tool_local))
            out.append((float(clearance), normal.copy(), j-index+1))
            if record:
                self.rows.append(dict(action_index=len(o.rows), future_step=j-index+1,
                    clearance_m=float(clearance), reference_feasible=True,
                    predicted_q=self.data.qpos[o.arm.qadr].copy(), protected_reference=np.asarray(target).copy()))
        return out

    def arrays(self):
        return {k: np.asarray([r[k] for r in self.rows]) for k in self.rows[0]} if self.rows else {}

    def metadata(self):
        return dict(algorithm='nominal_target_kinematic_forecast_ablation',
                    scope='Same risk scoring and execution_lift controller; only prediction is replaced')


class VLSAAdapter:
    """Shared-perception Cartesian CBF-QP comparator, explicitly an adaptation.

    Uses the existing AGV adapter and original QP, not the original VLM/MVEE
    perception pipeline. No F2, attraction or final F2 projection is added.
    """
    def __init__(self, arm, obstacle):
        from agv_vlsa import SharedPerceptionVLSA
        self.layer = SharedPerceptionVLSA(arm, obstacle, arm.steps*arm.model.opt.timestep)
        self.rows = []
        self.current_guard = False
        self.execution_predictor = None
        self.refresh_pending = False
        self.obstacle = obstacle

    def capsules(self, data):
        return []

    def process(self, chunk, index, **kwargs):
        timing = {}
        out = self.layer.process(np.asarray(chunk[index]).copy(), timing)
        row = dict(action_index=len(self.rows), current_clearance_m=np.nan,
                   deficit_m=0., geometry_active=bool(np.max(np.abs(out-chunk[index])) > 1e-9),
                   geometry_target=out.copy(), solver_time_ms=timing.get('solver_time_ms', np.nan))
        self.rows.append(row)
        return out, row

    def arrays(self):
        return {k: np.asarray([r[k] for r in self.rows]) for k in self.rows[0]} if self.rows else {}

    def metadata(self):
        return dict(**self.layer.geometry_metadata(), qp_failures=self.layer.qp_failures,
                    minimum_h=self.layer.min_h, comparator='adapted shared-perception VLSA, not original end-to-end VLSA')


def build_modules(arm, perception, initial_height, args, config):
    assert config['contact'] == (args.case in (5, 6, 7, 8))
    assert config['pillar'] == (args.case in (3, 4, 7, 8))
    assert config['attraction'] == (args.box_attraction == 'on')
    assert config['preview'] == (args.pace_preview == 'on')
    if not config['geometry'] and not config['attraction'] and config.get('baseline') != 'vlsa':
        return None, None
    obstacle = {k: np.array(perception['pillar'][k], copy=True) for k in ('p1', 'p2', 'r')}
    obstacle['placement_goal'] = np.asarray(perception['box']['placement_goal']).copy()
    if config.get('baseline') == 'vlsa':
        return VLSAAdapter(arm, obstacle), None
    geometry = AGVF2Preview(arm, obstacle, initial_height,
        AGVF2Config(max_correction_m_s=args.pace_max_correction_m_s,
                    lift_preference=0., horizon_s=float(config.get('horizon_s', .4))),
        pickup_mode='latched', joint_filter='off', preserve_orientation=True,
        current_guard=config['geometry'], preview_enabled=config['preview'], safety_only=True,
        route_preference=True, staged_route=True, prediction_mode='execution_lift',
        tracking_model_path=args.pace_tracking_model, proxy_mode=config.get('proxy_mode', 'components'))
    if config.get('predictor', 'execution') == 'nominal' and config['preview']:
        geometry.execution_predictor = NominalTargetPredictor(geometry)
    attraction = None
    if config['attraction']:
        goal = np.asarray(perception['box']['attraction_point']).copy()
        if config.get('target_reference_file'):
            saved = json.loads(Path(config['target_reference_file']).read_text())
            if saved.get('source') != 'groundingdino_rgbd' or saved.get('target_rule') != 'agv_offset':
                raise ValueError('Frozen target must have actual DINO provenance')
            goal = np.asarray(saved['goal_m'], float)
        attraction = AGVBoxAttraction(arm, goal, initial_height,
            BoxAttractionConfig(maximum_speed_m_s=args.box_attraction_speed_m_s,
                decay_power=float(config.get('decay_power', 1.)),
                staged_route=bool(config.get('staged_attraction', True)), check_descent=False,
                commit_transport=bool(config.get('commit_transport', True)),
                preserve_orientation=bool(config.get('attraction_orientation', True))))
        from tabletop_route_preference import near_base_direction
        attraction.near_direction = near_base_direction(arm.data.body('root').xpos, obstacle['p1'])
        attraction.obstacle = obstacle
        for key in ('tool_extent_m', 'tool_radius_m', 'proxy_mode', 'tool_capsules_local', 'eef_capsules_local'):
            setattr(attraction, {'tool_extent_m':'tool_extent', 'tool_radius_m':'tool_radius'}.get(key, key), getattr(geometry, key))
    return (geometry if config['geometry'] else None), attraction
