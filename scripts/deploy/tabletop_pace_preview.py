# Adapted from agv_pace_preview.py; original AGV implementation is unchanged.
"""F2 finite-window kinematic deflection adapted to the AGV action interface.

No policy calls, physics integration or live-state mutation occur here. This is
reference clearance prediction, not a continuous collision/safety certificate.
"""
from dataclasses import asdict, dataclass
import time
import numpy as np


@dataclass(frozen=True)
class AGVF2Config:
    horizon_s: float = .4
    slack_m_s: float = .1       # original .005 m per 50 ms future step
    max_correction_m_s: float = .6  # original .03 m per 50 ms control step
    lift_preference: float = 1.
    clearance_buffer_m: float = .01
    margin_min_m: float = .025
    margin_max_m: float = .08
    dls_damping: float = .05
    max_joint_speed_rad_s: float = 5.  # original .25 rad per 50 ms
    filter_reference_s: float = .05
    filter_beta_reference: float = .6

    def __post_init__(self):
        values = asdict(self)
        if not all(np.isfinite(v) and v > 0 for k, v in values.items() if k != "lift_preference"):
            raise ValueError("F2 configuration must be finite and positive")
        if not np.isfinite(self.lift_preference) or self.lift_preference < 0:
            raise ValueError("Invalid lift preference")
        if self.margin_min_m > self.margin_max_m or not 0 < self.filter_beta_reference < 1:
            raise ValueError("Invalid margin bounds or filter beta")


class AGVF2Preview:
    def __init__(self, arm, obstacle, initial_tool_height, config=None, *, distance_fn=None, pickup_mode="latched", joint_filter="on", preserve_orientation=False, current_guard=False, safety_only=False, six_dof=False, preview_enabled=True, route_preference=False, staged_route=False, prediction_mode="legacy", tracking_model_path=None, proxy_mode="enclosing"):
        import mujoco
        # Shared distance arithmetic only; do not instantiate legacy controllers.
        if distance_fn is None:
            from tabletop_pace_runtime import segment_to_segment_distance
            distance_fn = segment_to_segment_distance
        self.arm, self.config = arm, config or AGVF2Config()
        self.staged_route=bool(staged_route)
        self.route_phase="lift"
        self.route_preference = bool(route_preference)
        self.preview_enabled = bool(preview_enabled)
        self.six_dof = bool(six_dof)
        self.pose_length_m = .15
        self.safety_only = bool(safety_only)
        if self.safety_only and (joint_filter != "off" or self.config.lift_preference != 0.):
            raise ValueError("Safety-only mode requires no policy joint filter and zero lift preference")
        if pickup_mode not in ("legacy_height", "latched"):
            raise ValueError("Invalid pickup qualification")
        self.pickup_mode = pickup_mode
        self.pickup_ever = False
        self.algorithm = "agv_f2_preview_v2_phase_latched" if pickup_mode == "latched" else "agv_f2_preview_v1"
        self.preserve_orientation = bool(preserve_orientation)
        if self.preserve_orientation:
            self.algorithm = "agv_f2_preview_v3_pose_preserved"
        if joint_filter not in ("on", "off", "correction_only"):
            raise ValueError("Invalid joint filter")
        self.joint_filter = joint_filter
        if joint_filter == "correction_only":
            self.algorithm = "agv_f2_preview_v4_correction_filter"
        self.current_guard = bool(current_guard)
        self.guard_failures = []
        self.guard_projection_events = []
        if self.current_guard:
            if not self.preserve_orientation:
                raise ValueError("Current reference guard requires orientation-preserving mapping")
            self.algorithm = "agv_f2_preview_v6_bounded_reference_guard"
        if self.safety_only:
            self.algorithm = "agv_f2_preview_v8_joint_ranges_only"
        if self.six_dof:
            self.algorithm = "agv_f2_preview_v9_six_dof_closest_point"
        self.distance_fn = distance_fn
        self.dt = float(arm.steps * arm.model.opt.timestep)
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Invalid control interval")
        count = self.config.horizon_s / self.dt
        if count < 1 or not np.isclose(count, round(count), atol=1e-8):
            raise ValueError("Preview horizon must be an integer number of control intervals")
        self.window_steps = int(round(count))
        self.beta = 1 - (1 - self.config.filter_beta_reference) ** (self.dt / self.config.filter_reference_s)
        if joint_filter == "off":
            self.beta = 1.
        if obstacle is None:
            raise ValueError("F2 requires explicitly supplied perceived obstacle geometry")
        self.obstacle = {key: np.array(obstacle[key], dtype=float, copy=True)
                         for key in ("p1", "p2", "placement_goal")}
        self.obstacle["r"] = float(obstacle["r"])
        if any(self.obstacle[k].shape != (3,) or not np.isfinite(self.obstacle[k]).all()
               for k in ("p1", "p2", "placement_goal")) or not np.isfinite(self.obstacle["r"]) or self.obstacle["r"] <= 0:
            raise ValueError("Invalid obstacle capsule")
        if not np.isfinite(initial_tool_height):
            raise ValueError("Invalid initial tool height")
        self.initial_tool_height = float(initial_tool_height)
        self.current = mujoco.MjData(arm.model)
        self.scratch = mujoco.MjData(arm.model)
        self.tcp_id = arm.model.site("tcp_site").id
        self.wrist_id = arm.model.body("link6_1").id
        self.tool_id = arm.model.body("real_screwdriver").id
        self.gripper_qadr = arm.gripper_qadr
        self.tool_extent_m = np.array([-.12, .12])
        self.tool_radius_m = .02
        if self.safety_only:
            # Read rigid collision mesh dimensions, never contact truth or bin geometry.
            self.prepare(self.current)
            body_R = self.current.xmat[self.tool_id].reshape(3, 3)
            origin = self.current.xpos[self.tool_id]
            vertices = []
            for geom in arm.target_geoms:
                if not arm.model.geom_contype[geom]:
                    continue
                mesh = arm.model.geom_dataid[geom]
                if mesh < 0:
                    raise ValueError("Tool proxy requires collision mesh geometry")
                begin = arm.model.mesh_vertadr[mesh]
                count = arm.model.mesh_vertnum[mesh]
                world = (arm.model.mesh_vert[begin:begin+count] @
                         self.current.geom_xmat[geom].reshape(3, 3).T + self.current.geom_xpos[geom])
                vertices.append((world-origin) @ body_R)
            vertices = np.concatenate(vertices)
            self.tool_extent_m = np.array([vertices[:, 1].min(), vertices[:, 1].max()])
            self.tool_radius_m = max(.02, float(np.linalg.norm(vertices[:, [0, 2]], axis=1).max()))
        if proxy_mode not in ('enclosing','components'):raise ValueError('Invalid protected geometry representation')
        self.proxy_mode=proxy_mode
        self.tool_capsules_local=[dict(p1=np.array([0.,self.tool_extent_m[0],0.]),p2=np.array([0.,self.tool_extent_m[1],0.]),r=self.tool_radius_m)]
        self.eef_capsules_local=[]
        if proxy_mode=='components':
            from tabletop_tool_proxy import fit_tool_component_capsules,fit_body_component_capsules
            self.prepare(self.current)
            self.tool_capsules_local=fit_tool_component_capsules(arm,self.current)
            for name in ('link6_1','link7','link8','link9'):
                self.eef_capsules_local.extend(fit_body_component_capsules(arm,self.current,arm.model.body(name).id))
        self.last_delta = np.zeros(6)
        self.last_cartesian_delta = np.zeros(3)
        self.rows = []
        self.refresh_pending=False
        self.preview_lift_latched=False
        self.preview_transport_seen=False

        if prediction_mode not in ('legacy','execution','execution_trusted','execution_carry','execution_refresh','execution_lift','execution_lift_hold'):
            raise ValueError('Invalid prediction mode')
        if prediction_mode in ('execution','execution_trusted','execution_carry','execution_refresh','execution_lift','execution_lift_hold') and self.six_dof:
            raise ValueError('Execution predictor currently supports the translation/nullspace controller only')
        if self.preview_enabled and prediction_mode in ('execution_lift','execution_lift_hold') and not (self.safety_only and self.staged_route and self.joint_filter=='off'):
            raise ValueError('Predictive lift requires safety_only, staged_up_near route and joint_filter off')
        self.prediction_mode=prediction_mode
        self.execution_predictor=None
        if self.preview_enabled and prediction_mode in ('execution','execution_trusted','execution_carry','execution_refresh','execution_lift','execution_lift_hold'):
            if tracking_model_path is None:raise ValueError('Execution preview requires tracking model')
            from tabletop_execution_preview import ExecutionReferencePredictor
            self.execution_predictor=ExecutionReferencePredictor(self,tracking_model_path)

    def prepare(self, destination):
        destination.qpos[:] = self.arm.data.qpos
        self.arm.mj.mj_kinematics(self.arm.model, destination)
        self.arm.mj.mj_comPos(self.arm.model, destination)

    def set_preview_action(self, action):
        """Write absolute arm angles and total jaw opening into PRIVATE qpos."""
        arm, scratch = self.arm, self.scratch
        scratch.qpos[arm.qadr] = action[:6]
        scratch.qpos[self.gripper_qadr] = np.clip(action[6] / 2., 0., .04)
        arm.mj.mj_kinematics(arm.model, scratch)

    def carried_tool_local(self,data):
        from tabletop_tool_proxy import world_tool_capsules
        origin=data.xpos[self.wrist_id];R=data.xmat[self.wrist_id].reshape(3,3)
        return [dict(p1=R.T@(c['p1']-origin),p2=R.T@(c['p2']-origin),r=c['r']) for c in world_tool_capsules(data,self.tool_id,self.tool_capsules_local)]

    def capsules(self, data, tool_local=None):
        origin = data.xpos[self.wrist_id]
        if self.proxy_mode=='components':
            from tabletop_tool_proxy import world_body_capsules
            caps=world_body_capsules(data,self.eef_capsules_local)
        else:
            caps=[dict(p1=origin.copy(),p2=data.site_xpos[self.tcp_id].copy(),r=.07)]
        if tool_local is not None:
            rotation=data.xmat[self.wrist_id].reshape(3,3)
            if len(tool_local) and isinstance(tool_local[0],dict):
                caps.extend(dict(p1=origin+rotation@c['p1'],p2=origin+rotation@c['p2'],r=c['r']) for c in tool_local)
            else:
                caps.append(dict(p1=origin+rotation@tool_local[0],p2=origin+rotation@tool_local[1],r=self.tool_radius_m))
        return caps

    def clearance(self, caps):
        if self.proxy_mode=='components':
            from tabletop_tool_proxy import batch_capsule_clearances
            values,normals,_=batch_capsule_clearances(caps,self.obstacle);i=int(np.argmin(values));return float(values[i]),normals[i].copy()
        best = (np.inf, None)
        for cap in caps:
            distance, normal, _, _ = self.distance_fn(cap["p1"], cap["p2"], self.obstacle["p1"], self.obstacle["p2"])
            clearance = float(distance - cap["r"] - self.obstacle["r"])
            if clearance < best[0]:
                best = (clearance, normal.copy())
        return best

    def closest_point(self, data, tool_local=None):
        if self.proxy_mode=='components':
            from tabletop_tool_proxy import batch_capsule_clearances
            values,normals,points=batch_capsule_clearances(self.capsules(data,tool_local),self.obstacle);i=int(np.argmin(values));return float(values[i]),normals[i].copy(),points[i].copy()
        best = None
        for cap in self.capsules(data, tool_local):
            distance, normal, point, _ = self.distance_fn(cap["p1"], cap["p2"], self.obstacle["p1"], self.obstacle["p2"])
            value = float(distance - cap["r"] - self.obstacle["r"])
            if best is None or value < best[0]:
                best = (value, normal.copy(), point.copy())
        return best

    def process(self, chunk, index, *, execution_stop=None, action_index=None, goal_position=None, minimum_margin_only=False):
        started = time.perf_counter_ns()
        arm, cfg = self.arm, self.config
        chunk = np.asarray(chunk, dtype=float)
        if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) == 0 or not np.isfinite(chunk).all():
            raise ValueError("Expected a finite (N,7) action chunk")
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(chunk):
            raise ValueError("Invalid chunk index")
        scheduled_stop = min(len(chunk), len(chunk) if execution_stop is None else execution_stop)
        stop = min(scheduled_stop, index + self.window_steps)
        if stop <= index:
            raise ValueError("Preview index must belong to the executed chunk prefix")
        # Exactly the same clipping applied to executed policy targets.
        clipped = chunk[:stop].copy()
        clipped[:, :6] = np.clip(clipped[:, :6], arm.limits[:, 0], arm.limits[:, 1])
        clipped[:, 6] = np.clip(clipped[:, 6], 0., .08)
        nominal = clipped[index]
        self.prepare(self.current)
        self.prepare(self.scratch)
        tcp = self.current.site_xpos[self.tcp_id]
        wrist = self.current.xpos[self.wrist_id]
        tool = self.current.xpos[self.tool_id]
        tool_local = None
        # Legacy single-finger .02 m gate becomes total-width .04 m.
        tool_attached = bool(nominal[6] <= .04 and np.linalg.norm(tcp - tool) < .05)
        if tool_attached:
            axis = self.current.xmat[self.tool_id].reshape(3, 3)[:, 1]
            rotation = self.current.xmat[self.wrist_id].reshape(3, 3)
            tool_local = self.carried_tool_local(self.current)
        tool_lift_now = bool(tool[2] > self.initial_tool_height + .015)
        self.pickup_ever |= tool_lift_now
        picked_up = self.pickup_ever if self.pickup_mode == "latched" else tool_lift_now
        current_clearance, normal = self.clearance(self.capsules(self.current, tool_local))
        current_normal=normal.copy()
        selected_point_local = None
        if self.six_dof:
            _, normal, point = self.closest_point(self.current, tool_local)
            selected_point_local = self.current.xmat[self.wrist_id].reshape(3, 3).T @ (point-wrist)
        reference_goal = tcp.copy() if self.safety_only else (self.obstacle["placement_goal"] if goal_position is None else np.asarray(goal_position, dtype=float))
        if reference_goal.shape != (3,) or not np.isfinite(reference_goal).all():
            raise ValueError("Invalid phase reference goal")
        goal_distance = float(np.linalg.norm(tcp - reference_goal))
        margin = float(np.clip(.025 + .4 * goal_distance, cfg.margin_min_m, cfg.margin_max_m))
        goal_factor = float(np.clip((goal_distance - .035) / .065, 0., 1.))
        target_clearance = cfg.clearance_buffer_m if (minimum_margin_only or self.safety_only) else max(cfg.clearance_buffer_m, margin * goal_factor)
        risk, selected_clearance, selected_step = np.inf, current_clearance, 1
        predicted_clearance = current_clearance
        predicted_step = 0
        predicted_normal = None
        preview_qualified=bool(picked_up and (self.prediction_mode!='execution_carry' or (tool_attached and tool_lift_now)))
        if preview_qualified and self.preview_enabled:
            if self.execution_predictor is not None:
                for distance,candidate_normal,k in self.execution_predictor.forecast(clipped,index,stop,tool_local):
                    candidate_risk=distance+cfg.slack_m_s*self.dt*(k-1)
                    if candidate_risk<risk:
                        risk,predicted_clearance,predicted_step=candidate_risk,distance,k
                        predicted_normal=candidate_normal
            else:
                for k, action in enumerate(clipped[index:stop], start=1):
                    self.set_preview_action(action)
                    distance, candidate_normal = self.clearance(self.capsules(self.scratch, tool_local))
                    candidate_risk = distance + cfg.slack_m_s * self.dt * (k - 1)
                    if candidate_risk < risk:
                        risk, predicted_clearance, predicted_step = candidate_risk, distance, k
                        predicted_normal = candidate_normal
                        if self.six_dof:
                            _, predicted_normal, point = self.closest_point(self.scratch, tool_local)
                            predicted_point_local = self.scratch.xmat[self.wrist_id].reshape(3, 3).T @ (point-self.scratch.xpos[self.wrist_id])
            if predicted_normal is not None and risk < current_clearance:
                selected_clearance, normal, selected_step = predicted_clearance, predicted_normal, predicted_step
                if self.six_dof:
                    selected_point_local = predicted_point_local
        # Refresh mode uses the forecast to invalidate stale policy prefixes.
        # The physical command is exactly the existing current-state controller.
        if self.prediction_mode in ('execution_refresh','execution_lift','execution_lift_hold'):
            selected_clearance,normal,selected_step=current_clearance,current_normal,1
        deficit = max(0., target_clearance - selected_clearance) if picked_up else 0.
        delta_cartesian = np.zeros(3)
        correction = np.zeros(6)
        angular_correction = np.zeros(3)
        requested_twist = np.zeros(6)
        if deficit > 0:
            direction = normal + cfg.lift_preference * np.array([0., 0., 1.])
            if self.route_preference and (not self.staged_route or self.route_phase!="transport"):
                from tabletop_route_preference import up_near_direction
                direction = up_near_direction(arm.data.body('root').xpos, self.obstacle['p1'])
            direction /= np.linalg.norm(direction) + 1e-9
            delta_cartesian = direction * min(deficit / selected_step, cfg.max_correction_m_s * self.dt)
        unfiltered_delta = delta_cartesian.copy()
        if self.joint_filter == "correction_only":
            self.last_cartesian_delta += self.beta * (delta_cartesian - self.last_cartesian_delta)
            if not picked_up:
                self.last_cartesian_delta[:] = 0.
            delta_cartesian = self.last_cartesian_delta.copy()
        if np.linalg.norm(delta_cartesian) > 1e-12:
            J = np.zeros((3, arm.model.nv))
            Jr = np.zeros((3, arm.model.nv))
            arm.mj.mj_jacSite(arm.model, self.current, J, Jr, self.tcp_id)
            J = J[:, arm.dadr]
            Jr = Jr[:, arm.dadr]
            if self.six_dof:
                point = wrist + self.current.xmat[self.wrist_id].reshape(3, 3) @ selected_point_local
                lever = point-tcp
                direction = delta_cartesian / np.linalg.norm(delta_cartesian)
                torque_axis = np.cross(lever, direction)
                length = self.pose_length_m
                amount = np.linalg.norm(delta_cartesian)
                denominator = 1. + np.dot(torque_axis, torque_axis)/length**2
                requested_twist[:3] = direction * amount/denominator
                requested_twist[3:] = torque_axis * amount/(length**2*denominator)
                pose_J = np.vstack((J, length*Jr))
                desired = np.r_[requested_twist[:3], length*requested_twist[3:]]
                correction = pose_J.T @ np.linalg.solve(pose_J@pose_J.T + cfg.dls_damping**2*np.eye(6), desired)
            elif self.preserve_orientation:
                # Preserve the policy's nominal orientation to first order.
                # The correction is solved only in the angular Jacobian nullspace.
                _, singular, vectors = np.linalg.svd(Jr, full_matrices=True)
                rank = int(np.sum(singular > 1e-9))
                basis = vectors[rank:].T
                reduced = J @ basis
                correction = basis @ reduced.T @ np.linalg.solve(
                    reduced @ reduced.T + cfg.dls_damping**2 * np.eye(3), delta_cartesian)
            else:
                correction = J.T @ np.linalg.solve(J @ J.T + cfg.dls_damping**2 * np.eye(3), delta_cartesian)
            angular_correction = Jr @ correction
        current_q = arm.data.qpos[arm.qadr]
        if self.joint_filter == "correction_only":
            smooth = nominal[:6] - current_q + correction
        else:
            smooth = self.beta * (nominal[:6] - current_q + correction) + (1 - self.beta) * self.last_delta
        self.last_delta = smooth.copy()
        output = nominal.copy()
        if self.safety_only:
            # Constrain only joint legality of the added safety correction. A legal nominal policy
            # target passes through exactly when no correction is required.
            # Uniform scaling retains the angular-nullspace direction.
            scale = 1.
            peak = float(np.max(np.abs(correction)))
            if peak > 0.:
                for j, value in enumerate(correction):
                    if value > 0.:
                        scale = min(scale, (arm.limits[j, 1]-nominal[j])/value)
                    elif value < 0.:
                        scale = min(scale, (arm.limits[j, 0]-nominal[j])/value)
            correction *= max(0., scale)
            angular_correction = Jr @ correction if peak > 0. else np.zeros(3)
            output[:6] = np.clip(nominal[:6] + correction, arm.limits[:, 0], arm.limits[:, 1])
        else:
            output[:6] = np.clip(current_q + np.clip(smooth, -cfg.max_joint_speed_rad_s * self.dt,
                                                    cfg.max_joint_speed_rad_s * self.dt),
                                 arm.limits[:, 0], arm.limits[:, 1])
        debug = dict(action_index=len(self.rows) if action_index is None else int(action_index),
                     chunk_index=int(index), execution_stop=int(scheduled_stop), preview_stop=int(stop if self.preview_enabled else index),
                     preview_samples=stop - index if preview_qualified and self.preview_enabled else 0,
                     preview_qualified=preview_qualified,
                     preview_selected_step=selected_step, prediction_step=predicted_step,
                     pickup_gate=picked_up, tool_lift_now=tool_lift_now, tool_capsule_attached=tool_attached,
                     current_clearance_m=current_clearance, predicted_clearance_m=predicted_clearance,
                     selected_clearance_m=selected_clearance, target_clearance_m=target_clearance,
                     deficit_m=deficit, geometry_active=bool(np.linalg.norm(delta_cartesian) > 1e-12),
                     cartesian_correction_m=delta_cartesian.copy(), joint_correction_rad=correction.copy(),
                     unfiltered_cartesian_correction_m=unfiltered_delta,
                     angular_correction_rad=angular_correction, requested_twist=requested_twist,
                     reference_goal_m=reference_goal.copy(),
                     nominal_target=nominal.copy(), geometry_target=output.copy(),
                     compute_ms=(time.perf_counter_ns() - started) / 1e6)
        debug.update(preview_lift_requested=False,preview_lift_feasible=False,preview_lift_position_error_m=0.,preview_lift_orientation_error_rad=0.)
        if self.preview_enabled and self.prediction_mode in ('execution_lift','execution_lift_hold'):
            risk=predicted_clearance+cfg.slack_m_s*self.dt*max(0,predicted_step-1)
            carrying=bool(tool_attached and tool_lift_now)
            if not tool_attached:self.preview_transport_seen=False
            if self.route_phase=='transport':self.preview_transport_seen=True
            if not carrying or self.preview_transport_seen:self.preview_lift_latched=False
            elif predicted_step>0 and risk<target_clearance:self.preview_lift_latched=True
            if self.preview_lift_latched:
                from tabletop_preview_lift import lift_waypoint_target
                from tabletop_route_preference import up_near_direction
                direction=up_near_direction(arm.data.body('root').xpos,self.obstacle['p1'])
                proposed,projection=lift_waypoint_target(self,nominal,direction,cfg.max_correction_m_s*self.dt,hold_current_orientation=self.prediction_mode=='execution_lift_hold')
                debug.update(preview_lift_requested=True,preview_lift_feasible=projection['feasible'],preview_lift_position_error_m=projection['position_error_m'],preview_lift_orientation_error_rad=projection['orientation_error_rad'])
                if projection['feasible']:
                    output=proposed
                    debug.update(geometry_active=True,geometry_target=output.copy(),joint_correction_rad=output[:6]-nominal[:6],cartesian_correction_m=direction*cfg.max_correction_m_s*self.dt,angular_correction_rad=projection['nominal_orientation_delta_rad'])
        from tabletop_preview_refresh import needs_policy_refresh
        debug['preview_refresh_requested']=bool(self.preview_enabled and self.prediction_mode=='execution_refresh' and needs_policy_refresh(debug,index,self.dt,cfg.slack_m_s))
        self.refresh_pending |= debug['preview_refresh_requested']
        debug.update(preview_proposal=False,preview_accepted=False,preview_forecast_gain_m=0.,preview_current_gain_m=0.)
        if self.preview_enabled and self.prediction_mode=='execution_trusted':
            # Compare to the exact current-state controller, not to the raw policy target.
            self.preview_enabled=False
            try:
                baseline,baseline_debug=self.process(chunk,index,execution_stop=execution_stop,action_index=action_index,goal_position=goal_position,minimum_margin_only=minimum_margin_only)
                self.rows.pop()
            finally:
                self.preview_enabled=True
            proposal=bool(np.max(np.abs(output-baseline))>1e-9)
            if proposal:
                baseline_forecast=self.execution_predictor.forecast(clipped,index,stop,tool_local,record=False)
                proposed_forecast=self.execution_predictor.forecast(clipped,index,stop,tool_local,first_target=output,record=False)
                def score(sequence):
                    return min(distance+cfg.slack_m_s*self.dt*(k-1) for distance,_,k in sequence)
                gain=score(proposed_forecast)-score(baseline_forecast)
                self.prepare(self.scratch);self.set_preview_action(baseline)
                old_clearance=self.clearance(self.capsules(self.scratch,tool_local))[0]
                self.prepare(self.scratch);self.set_preview_action(output)
                new_clearance=self.clearance(self.capsules(self.scratch,tool_local))[0]
                current_gain=new_clearance-old_clearance
                accepted=bool(gain>1e-5 and current_gain>=-1e-6)
                if not accepted:
                    # Preserve all original prediction diagnostics while executing baseline.
                    output=baseline
                    for key in ('geometry_active','cartesian_correction_m','joint_correction_rad','angular_correction_rad','requested_twist','unfiltered_cartesian_correction_m','geometry_target'):
                        debug[key]=baseline_debug[key]
                debug.update(preview_proposal=True,preview_accepted=accepted,preview_forecast_gain_m=float(gain),preview_current_gain_m=float(current_gain))
        debug["compute_ms"]=(time.perf_counter_ns()-started)/1e6
        self.rows.append(debug)
        return output, debug

    def metadata(self):
        return dict(proxy_mode=self.proxy_mode,tool_component_capsules=[dict(p1=c['p1'].tolist(),p2=c['p2'].tolist(),r=c['r']) for c in self.tool_capsules_local],eef_component_capsules=[dict(body_id=c['body_id'],p1=c['p1'].tolist(),p2=c['p2'].tolist(),r=c['r']) for c in self.eef_capsules_local],preview_lift_orientation_reference='current physical TCP orientation during latched lift; nominal policy restored after clear crossing' if self.prediction_mode=='execution_lift_hold' else 'nominal policy target orientation',preview_role='forecast-triggered latched nonlinear lift waypoint; policy grip retained; current-state final guard retained' if self.prediction_mode in ('execution_lift','execution_lift_hold') else 'forecast-triggered policy refresh; current-state geometry command unchanged; at least 10 executed steps between requests' if self.prediction_mode=='execution_refresh' else 'geometric deflection',preview_gate='currently attached carried-tool proxy and current lift >15mm, plus pickup qualification' if self.prediction_mode=='execution_carry' else 'pickup qualification only',preview_acceptance='candidate must improve baseline future minimum risk by >1e-5m and not worsen current endpoint clearance by >1e-6m' if self.prediction_mode=='execution_trusted' else 'legacy direct correction',prediction_mode=self.prediction_mode,execution_prediction=self.execution_predictor.metadata() if self.execution_predictor is not None else None,route_preference=dict(enabled=self.route_preference,staged=self.staged_route,up_weight=3.,near_base_weight=1.,outward_reference='no global outward restriction; preference only during lift' if self.staged_route else 'forbidden after pickup; linear constrained projection plus endpoint FK check'),preview_enabled=self.preview_enabled, geometry_mode="chunk_preview" if self.preview_enabled else "current_state_only", six_dof=self.six_dof, pose_length_m=self.pose_length_m, pose_mapping="closest-point normal wrench to weighted TCP 6D twist" if self.six_dof else "legacy translation with angular nullspace", joint_limit_scope="joint legal ranges only; no joint velocity/delta limit; legal nominal policy targets pass through" if self.safety_only else "whole policy-plus-correction target delta", tool_axial_extent_m=self.tool_extent_m.tolist(), tool_radius_m=self.tool_radius_m, safety_only=self.safety_only, algorithm=self.algorithm if self.preview_enabled else "tabletop_current_state_geometry_v1", pickup_qualification=self.pickup_mode, joint_filter=self.joint_filter,
                    preserve_orientation=self.preserve_orientation and not self.six_dof,
                    current_reference_guard=self.current_guard, guard_failures=self.guard_failures,
                    bounded_projection_events=self.guard_projection_events,
                    guard_scope="post-contact discrete kinematic normal constraint; no contact truth; joint-limited infeasibility aborts, not a continuous safety certificate",
                    orientation_scope="linearized correction before smoothing/joint clipping; not exact nonlinear pose invariance",
                    config=asdict(self.config),
                    action_period_s=self.dt, window_steps=self.window_steps, filter_beta=self.beta,
                    future_slack_per_step_m=self.config.slack_m_s * self.dt,
                    cartesian_correction_limit_per_step_m=self.config.max_correction_m_s * self.dt,
                    joint_delta_limit_per_step_rad=None if self.safety_only else self.config.max_joint_speed_rad_s * self.dt,
                    qpos_indices=self.arm.qadr.tolist(), dof_indices=self.arm.dadr.tolist(),
                    gripper="tabletop total opening=2*per-finger target; private two-slider kinematics; carried-tool proxy held across preview",
                    prediction="kinematic chunk targets, bounded by next policy query; no physics rollout" if self.preview_enabled else "disabled; current-state capsule clearance only; policy chunk cadence unchanged",
                    rows_scope="geometry attempts; join action_index with trace and substep logs to distinguish executed/partial attempts",
                    jacobian="weighted TCP pose Jacobian; closest capsule point gradient for final guard" if self.six_dof else "current TCP site, mapped by arm.dadr", obstacle={k: np.asarray(v).tolist() for k, v in self.obstacle.items()})

    def project_current_target(self, target):
        """Minimal angular-nullspace projection of the FINAL substep reference.

        Uses model/reference displacement at the configured action horizon, not actual
        physical velocity. It deliberately does not use simulation contact truth.
        """
        arm = self.arm
        target = np.asarray(target, dtype=float).copy()
        if self.safety_only:
            target[:6] = np.clip(target[:6], arm.limits[:, 0], arm.limits[:, 1])
        self.prepare(self.current)
        q = arm.data.qpos[arm.qadr].copy()
        tcp = self.current.site_xpos[self.tcp_id]
        tool = self.current.xpos[self.tool_id]
        lift_now = bool(tool[2] > self.initial_tool_height + .015)
        self.pickup_ever |= lift_now
        qualified = self.pickup_ever if self.pickup_mode == "latched" else lift_now
        tool_local = None
        if target[6] <= .04 and np.linalg.norm(tcp - tool) < .05:
            origin = self.current.xpos[self.wrist_id]
            R = self.current.xmat[self.wrist_id].reshape(3, 3)
            axis = self.current.xmat[self.tool_id].reshape(3, 3)[:, 1]
            tool_local = self.carried_tool_local(self.current)
        clearance, normal = self.clearance(self.capsules(self.current, tool_local))
        gamma = 1 - .5 ** (self.dt / .05)
        bound = -gamma * (clearance - self.config.clearance_buffer_m)
        Jp = np.zeros((3, arm.model.nv))
        Jr = Jp.copy()
        arm.mj.mj_jacSite(arm.model, self.current, Jp, Jr, self.tcp_id)
        Jp, Jr = Jp[:, arm.dadr], Jr[:, arm.dadr]
        if self.six_dof:
            _, normal, point = self.closest_point(self.current, tool_local)
            point_J = np.zeros((3, arm.model.nv))
            arm.mj.mj_jac(arm.model, self.current, point_J, None, point, self.wrist_id)
            row = normal @ point_J[:, arm.dadr]
        else:
            row = normal @ Jp
        near_row = None
        if self.route_preference and qualified and not self.staged_route:
            from tabletop_route_preference import near_base_direction
            near = near_base_direction(arm.data.body('root').xpos,self.obstacle['p1'])
            wrist_J=np.zeros((3,arm.model.nv))
            arm.mj.mj_jac(arm.model,self.current,wrist_J,None,self.current.body('link6_1').xpos,self.wrist_id)
            near_row = near @ ((Jp+wrist_J[:,arm.dadr])/2.)
        before = float(row @ (target[:6] - q))
        active = bool(qualified and before < bound)
        output = target.copy()
        feasible = True
        route_active = bool(near_row is not None and near_row @ (target[:6]-q) < -1e-9)
        if active or route_active:
            from agv_reference_projection import bounded_nullspace_projection
            started = time.perf_counter_ns()
            step_limit = self.config.max_joint_speed_rad_s * self.dt
            if self.safety_only:
                lower, upper = arm.limits[:, 0], arm.limits[:, 1]
            else:
                lower = np.maximum(arm.limits[:, 0], q-step_limit)
                upper = np.minimum(arm.limits[:, 1], q+step_limit)
            if near_row is not None:
                from tabletop_route_preference import bounded_multi_projection
                correction,feasible=bounded_multi_projection(np.vstack([row,near_row]),Jr,
                    np.array([bound-before,-near_row@(target[:6]-q)]),lower-target[:6],upper-target[:6])
                info=dict(method='up_near_multi_constraint')
            else:
                correction, feasible, info = bounded_nullspace_projection(
                    row, np.zeros_like(Jr) if self.six_dof else Jr, bound-before, lower-target[:6], upper-target[:6])
            output[:6] = np.clip(target[:6]+correction, lower, upper)
            feasible = bool(feasible and row @ (output[:6]-q) >= bound-1e-9)
            if info['method'] != 'analytic':
                self.guard_projection_events.append(dict(time_s=float(arm.data.time),
                    feasible=feasible, compute_ms=(time.perf_counter_ns()-started)/1e6, **info))
        route_endpoint_m=0.
        if near_row is not None:
            feasible = bool(feasible and near_row @ (output[:6]-q)>=-1e-9)
            # Linear no-outward constraint is insufficient for a distant joint target.
            self.scratch.qpos[:]=arm.data.qpos
            self.scratch.qpos[arm.qadr]=output[:6]
            arm.mj.mj_kinematics(arm.model,self.scratch)
            current_center=(self.current.body('link6_1').xpos+tcp)/2.
            target_center=(self.scratch.body('link6_1').xpos+self.scratch.site('tcp_site').xpos)/2.
            route_endpoint_m=float(near@(target_center-current_center))
            feasible=bool(feasible and route_endpoint_m>=-1e-6)
        after = float(row @ (output[:6] - q))
        debug = dict(post_contact_target=target, reference_guard_active=active,
                     reference_guard_qualified=bool(qualified), reference_guard_feasible=feasible,
                     reference_guard_clearance_m=clearance, reference_guard_bound_m=bound,
                     reference_guard_before_m=before, reference_guard_after_m=after,
                     reference_guard_joint_delta_rad=output[:6] - target[:6],
                     route_preference_qualified=bool(near_row is not None),route_projection_active=route_active,
                     route_near_endpoint_m=route_endpoint_m)
        if not feasible:
            self.guard_failures.append(dict(time_s=float(arm.data.time), clearance_m=clearance,
                                           bound_m=bound, achieved_m=after))
        return output, debug

    def arrays(self):
        shapes = {"preview_lift_requested": (), "preview_lift_feasible": (), "preview_lift_position_error_m": (), "preview_lift_orientation_error_rad": (), "preview_refresh_requested": (), "preview_qualified": (), "preview_proposal": (), "preview_accepted": (), "preview_forecast_gain_m": (), "preview_current_gain_m": (), "action_index": (), "chunk_index": (), "execution_stop": (), "preview_stop": (), "preview_samples": (),
                  "preview_selected_step": (), "prediction_step": (), "pickup_gate": (), "tool_lift_now": (),
                  "tool_capsule_attached": (), "current_clearance_m": (), "predicted_clearance_m": (),
                  "selected_clearance_m": (), "target_clearance_m": (), "deficit_m": (), "geometry_active": (),
                  "cartesian_correction_m": (3,), "joint_correction_rad": (6,), "angular_correction_rad": (3,), "requested_twist": (6,), "nominal_target": (7,),
                  "unfiltered_cartesian_correction_m": (3,), "reference_goal_m": (3,),
                  "geometry_target": (7,), "compute_ms": ()}
        integers = {"action_index", "chunk_index", "execution_stop", "preview_stop", "preview_samples", "preview_selected_step", "prediction_step"}
        booleans = {"preview_lift_requested", "preview_lift_feasible", "preview_refresh_requested", "preview_qualified", "preview_proposal", "preview_accepted", "pickup_gate", "tool_lift_now", "tool_capsule_attached", "geometry_active"}
        return {key: np.asarray([row[key] for row in self.rows], dtype=bool if key in booleans else
                                int if key in integers else float).reshape((-1, *shape)) for key, shape in shapes.items()}


GUARD_LOG_SHAPES = {"post_contact_target": (7,), "reference_guard_active": (),
    "reference_guard_qualified": (), "reference_guard_feasible": (),
    "reference_guard_clearance_m": (), "reference_guard_bound_m": (),
    "reference_guard_before_m": (), "reference_guard_after_m": (),
    "reference_guard_joint_delta_rad": (6,)}
