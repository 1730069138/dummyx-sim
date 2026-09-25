import sys
import unittest
from pathlib import Path

import numpy as np


DEPLOY_DIR = Path(__file__).resolve().parents[1] / "scripts" / "deploy"
sys.path.insert(0, str(DEPLOY_DIR))

import deploy_screwdriver_client_vlsa as vlsa


class InterventionTuningTests(unittest.TestCase):
    def make_layer(self, scale=0.8, linear=0.08, angular=0.08):
        layer = vlsa.AEGISFullActionLayer(
            obstacle_center=np.array([0.12, 0.0, 0.0]),
            obstacle_rotation=np.eye(3),
            obstacle_axes=np.array([0.04, 0.04, 0.08]),
            action_dt=0.05,
            eef_axes=vlsa.Q_EEF_DIAG * scale,
            max_linear_correction=linear,
            max_angular_correction=angular,
        )
        layer.reset(np.zeros(3))
        return layer

    def test_smaller_eef_ellipsoid_increases_clearance(self):
        full = self.make_layer(scale=1.0)
        reduced = self.make_layer(scale=0.8)
        full_h = full.current_h(np.zeros(3), np.eye(3))
        reduced_h = reduced.current_h(np.zeros(3), np.eye(3))
        self.assertGreater(reduced_h, full_h)

    def test_qp_limits_velocity_correction_inside_solver(self):
        layer = self.make_layer()
        v_nominal = np.array([0.3, 0.0, 0.0])
        omega_nominal = np.array([0.0, 1.0, 0.0])
        v_safe, omega_safe, debug = layer.filter(
            np.zeros(3),
            np.eye(3),
            v_nominal,
            omega_nominal,
            np.zeros(3),
            np.zeros(3),
        )
        self.assertTrue(debug["qp_ok"])
        self.assertLessEqual(np.max(np.abs(v_safe - v_nominal)), 0.080001)
        self.assertLessEqual(
            np.max(np.abs(omega_safe - omega_nominal)),
            0.08001,
        )

    def test_joint_space_qp_returns_direct_bounded_command(self):
        layer = self.make_layer()
        j_pos = np.eye(3, 6)
        j_rot = np.hstack([np.zeros((3, 3)), np.eye(3)])
        qdot_nom = np.array([0.3, 0.0, 0.0, 0.0, 1.0, 0.0])
        qdot_safe, debug = layer.filter_joint_target(
            np.zeros(3),
            np.eye(3),
            j_pos,
            j_rot,
            qdot_nom,
            np.zeros(6),
            servo_response=0.4,
        )
        self.assertTrue(debug["qp_ok"])
        self.assertEqual(qdot_safe.shape, (6,))
        self.assertLessEqual(
            np.max(np.abs(j_pos @ (qdot_safe - qdot_nom))),
            0.08001,
        )
        self.assertLessEqual(
            np.max(np.abs(j_rot @ (qdot_safe - qdot_nom))),
            0.08001,
        )

    def test_attached_geometry_adds_second_cbf_constraint(self):
        layer = self.make_layer()
        j_pos = np.eye(3, 6)
        j_rot = np.hstack([np.zeros((3, 3)), np.eye(3)])
        _, debug = layer.filter_joint_target(
            np.zeros(3),
            np.eye(3),
            j_pos,
            j_rot,
            np.zeros(6),
            np.zeros(6),
            servo_response=0.4,
            attached_geometry=(
                np.array([0.04, 0.0, 0.0]),
                np.array([0.02, 0.02, 0.08]),
                np.eye(3),
            ),
        )
        self.assertTrue(debug["qp_ok"])
        self.assertTrue(np.isfinite(debug["attached_h"]))
        self.assertIsNotNone(layer.attached_z)


if __name__ == "__main__":
    unittest.main()
