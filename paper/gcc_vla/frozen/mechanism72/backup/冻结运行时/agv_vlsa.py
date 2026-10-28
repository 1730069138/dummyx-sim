"""Cartesian VLSA adapter using the AGV's shared GroundingDINO capsule."""

import numpy as np


ALGORITHM = "shared_groundingdino_cartesian_cbf_qp_v1"


def capsule_ellipsoid(capsule):
    """Conservatively enclose the perceived capsule, without simulator geometry."""
    p1, p2 = np.asarray(capsule["p1"]), np.asarray(capsule["p2"])
    length = np.linalg.norm(p2 - p1)
    axis = (p2 - p1) / length
    reference = np.array([1., 0., 0.]) if abs(axis[0]) < .9 else np.array([0., 1., 0.])
    transverse = np.cross(reference, axis)
    transverse /= np.linalg.norm(transverse)
    rotation = np.column_stack((transverse, np.cross(axis, transverse), axis))
    radius = float(capsule["r"])
    # The capsule is inside a cylinder of radius r and half-height length/2+r.
    # Scaling those two bounds by sqrt(2) encloses the complete cylinder.
    axes = np.sqrt(2.) * np.array([radius, radius, length / 2. + radius])
    return (p1 + p2) / 2., rotation, axes


class SharedPerceptionVLSA:
    def __init__(self, arm, capsule, action_dt):
        import deploy_screwdriver_client_vlsa as reference
        self.reference = reference
        self.arm = arm
        self.action_dt = action_dt
        self.eef_body = arm.model.body("link7").id
        center, rotation, axes = capsule_ellipsoid(capsule)
        self.layer = reference.AEGISFullActionLayer(
            center, rotation, axes, action_dt,
            eef_axes=reference.Q_EEF_DIAG * .8,
            max_linear_correction=.08, max_angular_correction=.08)
        self.layer.reset(self.eef_pose()[0])
        self.qp_failures = 0
        self.min_h = float("inf")

    def eef_pose(self):
        return self.reference.get_eef_ellipsoid_pose(self.arm.data, self.eef_body)

    def geometry_metadata(self):
        return {"algorithm": ALGORITHM, "action_space": "cartesian",
                "perception": "shared GroundingDINO capsule converted to an enclosing ellipsoid; VLM and MVEE are not used",
                "obstacle_center": self.layer.p_obs.tolist(),
                "obstacle_rotation": self.layer.r_obs.tolist(),
                "obstacle_axes": self.layer.q_obs.tolist(),
                "eef_axes": self.layer.eef_axes.tolist(),
                "eef_local_center": self.reference.EEF_CENTER_IN_LINK7.tolist(),
                "eef_ellipsoid_scale": .8, "max_linear_correction_m_s": .08,
                "max_angular_correction_rad_s": .08, "action_dt_s": self.action_dt,
                "protects_carried_tool": False,
                "qp_failure_behavior": "reference filter returns nominal twist; failure is counted"}

    def process(self, action, timing=None):
        arm, reference = self.arm, self.reference
        q = arm.state()[:6]
        qdot_nom = np.clip(action[:6] - q, -reference.MAX_JOINT_DELTA,
                          reference.MAX_JOINT_DELTA) / self.action_dt
        center, rotation = self.eef_pose()
        jacp, jacr = reference.get_point_jacobians(arm.model, arm.data, center, self.eef_body)
        jacp, jacr = jacp[:, arm.dadr], jacr[:, arm.dadr]
        actual = arm.data.qvel[arm.dadr]
        v_nom, omega_nom = jacp @ qdot_nom, jacr @ qdot_nom
        v_safe, omega_safe, debug = self.layer.filter(
            center, rotation, v_nom, omega_nom, jacp @ actual, jacr @ actual)
        correction = reference.damped_pinv(np.vstack((jacp, jacr)), reference.DLS_RHO) @ np.r_[
            v_safe - v_nom, omega_safe - omega_nom]
        delta = np.clip((qdot_nom + correction) * self.action_dt,
                        -reference.MAX_JOINT_DELTA, reference.MAX_JOINT_DELTA)
        output = action.copy()
        output[:6] = np.clip(q + delta, arm.limits[:, 0], arm.limits[:, 1])
        self.qp_failures += int(not debug["qp_ok"])
        self.min_h = min(self.min_h, float(debug["h"]))
        if timing is not None:
            timing["solver_time_ms"] = float(debug["qp_ms"])
        return output

    def draw(self, viewer):
        with viewer.lock():
            viewer.user_scn.ngeom = 0
            center, rotation = self.eef_pose()
            self.reference._append_debug_ellipsoid(
                viewer.user_scn, center, rotation, self.layer.eef_axes, [0., .5, 1., .3])
            self.reference._append_debug_ellipsoid(
                viewer.user_scn, self.layer.p_obs, self.layer.r_obs, self.layer.q_obs, [1., 1., 0., .3])
