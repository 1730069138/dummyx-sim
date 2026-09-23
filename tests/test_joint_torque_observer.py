import argparse
import ast
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/deploy"))
from joint_torque_observer import (ModelResidualObserver, MujocoTorqueObserver,
    ObserverConfig, ForceEstimate, ForceLog, select_control_force,
    robot_contact_truth, arm_body_ids, add_observer_arguments)
import deploy_screwdriver_client as deploy


class ObserverTests(unittest.TestCase):
    def config(self, **kwargs):
        return ObserverConfig(**dict(dict(damping=1e-6, cutoff_hz=1e5, deadzone=0), **kwargs))

    def update_force(self, observer, force, dt=0.001):
        J = np.column_stack((np.eye(3), np.zeros((3, 3))))
        return observer.update(np.zeros(6), J.T @ force, np.zeros(6), np.zeros(6), J, dt)

    def test_recover_known_tcp_force_and_dynamics_sign(self):
        J = np.array([[0.2, 0.1, 0, 0.03, 0.1, 0],
                      [0, 0.2, 0.1, 0, 0.03, 0.1],
                      [0.1, 0, 0.3, 0.1, 0, 0.03]])
        expected = np.array([2., -3., 8.])
        inertia = np.arange(6) * 0.1
        bias = np.arange(6) * 0.2
        passive = -np.arange(6) * 0.03
        measured = inertia + bias - passive - J.T @ expected
        result = ModelResidualObserver(self.config()).update(inertia, bias, passive, measured, J, .001)
        np.testing.assert_allclose(result.force, expected, atol=1e-7)
        np.testing.assert_allclose(result.tau_external, J.T @ expected)
        zero = ModelResidualObserver(self.config()).update(inertia, bias, passive,
                inertia + bias - passive, J, .001)
        np.testing.assert_allclose(zero.force, 0, atol=1e-10)

    def test_soft_deadzone_and_vector_limit(self):
        obs = ModelResidualObserver(self.config(deadzone=.5, max_force=5))
        np.testing.assert_allclose(self.update_force(obs, [0, 0, .4]).force, 0)
        np.testing.assert_allclose(self.update_force(obs, [0, 0, 2]).force, [0, 0, 1.5], atol=1e-8)
        f = self.update_force(obs, [6, 8, 0]).force
        np.testing.assert_allclose(f, [3, 4, 0], atol=1e-8)

    def test_filter_hysteresis_and_reset(self):
        obs = ModelResidualObserver(self.config(cutoff_hz=10, enter_force=1, exit_force=.5))
        first = self.update_force(obs, [0, 0, 2])
        self.assertAlmostEqual(first.force[2], 2 * (1 - np.exp(-2*np.pi*10*.001)))
        self.assertFalse(first.contact)
        for _ in range(100):
            last = self.update_force(obs, [0, 0, 2])
        self.assertTrue(last.contact)
        for _ in range(100):
            last = self.update_force(obs, [0, 0, .7])
        self.assertTrue(last.contact)
        for _ in range(100):
            last = self.update_force(obs, [0, 0, .2])
        self.assertFalse(last.contact)
        obs.reset()
        self.assertFalse(obs.contact)
        np.testing.assert_array_equal(obs.filtered_force, 0)

    def test_singular_jacobian_and_invalid_parameters(self):
        out = ModelResidualObserver().update(np.ones(6), np.zeros(6), np.zeros(6),
                                            np.zeros(6), np.zeros((3, 6)), .001)
        np.testing.assert_array_equal(out.force, 0)
        for kwargs in ({"damping": 0}, {"cutoff_hz": -1}, {"exit_force": 2}, {"deadzone": float('nan')}):
            with self.assertRaises(ValueError):
                ObserverConfig(**kwargs)

    def test_cli_default_and_logging_schema(self):
        parser = argparse.ArgumentParser()
        add_observer_arguments(parser)
        self.assertEqual(parser.parse_args([]).contact_force_source, "joint_torque")
        log = ForceLog()
        log.append(0., ForceEstimate(np.ones(3), np.ones(6), True), np.zeros(3), np.zeros(3), .8)
        arrays = log.arrays()
        self.assertEqual(arrays['tau_external'].shape, (1, 6))
        np.testing.assert_array_equal(arrays['force_error'], arrays['f_estimated'])
        self.assertFalse(arrays['truth_contact'][0])


class MujocoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(ROOT / 'models/dummyx_apf_scene.xml'))
        cls.tcp = mujoco.mj_name2id(cls.model, mujoco.mjtObj.mjOBJ_SITE, 'tcp_site')

    def test_joint_torque_control_never_reads_contact_for_either_form(self):
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        data.site_xpos[self.tcp, 2] = .22  # exercise active protection branch
        estimate = ForceEstimate(np.array([0., 0., 2.]), np.zeros(6), True)
        with patch.object(mujoco, 'mj_contactForce', side_effect=AssertionError('truth leaked')), \
             patch.object(deploy, 'get_target_table_force', side_effect=AssertionError('legacy input leaked')):
            for mode in ('admittance', 'impedance'):
                force = select_control_force('joint_torque', estimate,
                    lambda: deploy.get_target_table_force(self.model, data, list(range(self.model.nbody))))
                ctrl, active = deploy.apply_virtual_wall_protection(self.model, data, np.zeros(8), self.tcp, force, 0, mode)
                self.assertTrue(active)
                self.assertTrue(np.isfinite(ctrl).all())
        # Explicit legacy mode must use its reader.
        np.testing.assert_array_equal(select_control_force('contact', estimate, lambda: np.ones(3)), 1)
        # Verify the actual deployment call receives only the selected force.
        tree = ast.parse((ROOT / 'scripts/deploy/deploy_screwdriver_client.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == 'apply_virtual_wall_protection']
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args[4].id, 'current_f_xyz')
        assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'current_f_xyz' for t in n.targets)]
        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].value.func.id, 'select_control_force')

    def test_encoder_residual_no_external_force_and_known_applied_force(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT / 'models/dummyx_apf_scene.xml'))
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        model.dof_damping[:] = 0.2  # verify implicit damping correction too
        data = mujoco.MjData(model)
        data.qpos[:6] = [0, .4, .5, 0, 1.57, 0]
        data.ctrl[:6] = data.qpos[:6]
        mujoco.mj_forward(model, data)
        observer = MujocoTorqueObserver(model, self.tcp, ObserverConfig(damping=1e-6, cutoff_hz=1e5, deadzone=0))
        observer.reset(data)
        mujoco.mj_step(model, data)
        zero = observer.sample_after_step(data)
        np.testing.assert_allclose(zero.force, 0, atol=1e-8)
        mujoco.mj_forward(model, data)
        observer.reset(data)
        J = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, J, None, self.tcp)
        expected = np.array([1., -2., 4.])
        data.qfrc_applied[:6] = J[:, :6].T @ expected
        with patch.object(mujoco, 'mj_contactForce', side_effect=AssertionError('truth leaked')):
            mujoco.mj_step(model, data)
            data.qacc[:] = np.nan  # neither solver acceleration nor constraint truth is an input
            data.qfrc_constraint[:] = np.nan
            result = observer.sample_after_step(data)
        np.testing.assert_allclose(result.force, expected, atol=1e-6)
        observer.reset(data)
        np.testing.assert_array_equal(observer.observer.filtered_force, 0)
        np.testing.assert_array_equal(observer.previous_velocity, data.qvel)

    def test_truth_contact_sign(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom type="plane" size="1 1 .1"/><body pos="0 0 .09"><freejoint/>
          <geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        force = robot_contact_truth(model, data, {1})
        self.assertGreater(force[2], 0)
        np.testing.assert_allclose(robot_contact_truth(model, data, {0}), -force)
        np.testing.assert_allclose(robot_contact_truth(model, data, {0, 1}), 0)


if __name__ == '__main__':
    unittest.main()
