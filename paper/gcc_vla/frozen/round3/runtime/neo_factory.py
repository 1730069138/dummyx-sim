"""Frozen-policy external benchmark dispatch, never a PACE/external hybrid."""
from contextlib import contextmanager
import copy
import numpy as np
import tabletop_perception as perception_module
from tabletop_paper_ablation import build_modules as BASELINE_BUILD

ORIGINAL_FIT = perception_module.fit_pillar
METHODS = ('raw_pi0', 'pace_g', 'pace_gc', 'vlsa', 'oscbf', 'neo')


def trimmed_fit_pillar(points, full_cloud):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 30:
        return ORIGINAL_FIT(points, full_cloud)
    lower = float(np.quantile(points[:, 2], .02))
    keep = points[:, 2] >= lower
    geometry, support = ORIGINAL_FIT(points[keep], full_cloud)
    geometry['shared_input_filter'] = dict(rule='lower_world_z_quantile_trim', quantile=.02,
        lower_z_m=lower, input_color_depth_points=len(points), kept_points=int(keep.sum()),
        rejected_points=int((~keep).sum()), simulator_object_pose_used=False,
        scope='Previously authorized input preprocessing, common to all methods in X+40 scene')
    return geometry, support


@contextmanager
def scene_context(config):
    requested = config.get('shared_perception_filter', 'none')
    enabled = requested == 'lower_z_quantile_2pct'
    if requested not in ('none', 'lower_z_quantile_2pct'):
        raise ValueError('Unregistered preprocessing')
    if bool(config.get('depth_filter', False)) != enabled:
        raise ValueError('Inconsistent common preprocessing flags')
    if perception_module.fit_pillar is not ORIGINAL_FIT:
        raise RuntimeError('Unexpected pre-existing perception patch')
    if enabled:
        perception_module.fit_pillar = trimmed_fit_pillar
    try:
        yield
    finally:
        perception_module.fit_pillar = ORIGINAL_FIT


def build_modules(arm, perception, initial_height, args, config):
    method = config['method_id']
    if method not in METHODS:
        raise ValueError('Unregistered independent method')
    if config['preview'] or config['attraction'] or args.box_attraction != 'off' or args.pace_preview != 'off':
        raise ValueError('Preview/attraction forbidden in every arm')
    if config.get('directional_rule', 'original') != 'original':
        raise ValueError('Only original PACE is an eligible PACE reference')
    if method in ('raw_pi0', 'pace_g', 'pace_gc'):
        expected = {'raw_pi0': (False, False), 'pace_g': (True, False), 'pace_gc': (True, True)}[method]
        if (config['geometry'], config['contact']) != expected:
            raise ValueError('PACE reference switches do not match frozen original')
        return BASELINE_BUILD(arm, perception, initial_height, args, config)
    if config['geometry'] or config['contact'] or args.case != 3:
        raise ValueError('External arm must disable PACE G and C at deployment entry')
    if config.get('baseline') == 'vlsa':
        raise ValueError('Old adapter baseline selector forbidden in independent dispatch')
    if method == 'vlsa':
        from vlsa_independent import VLSAIndependent
        obstacle = {k: np.array(perception['pillar'][k], copy=True) for k in ('p1', 'p2', 'r')}
        return VLSAIndependent(arm, obstacle), None
    # Only the frozen geometry/kinematics provider is constructed. Neither of its
    # PACE execution methods is ever invoked by an external controller.
    helper_config = copy.deepcopy(config)
    helper_config['geometry'] = True
    helper, attraction = BASELINE_BUILD(arm, perception, initial_height, args, helper_config)
    if helper is None or attraction is not None:
        raise RuntimeError('Geometry provider construction failed')
    def forbidden_pace_execution(*a, **kw):
        raise RuntimeError('PACE execution reached from independent external arm')
    helper.process = forbidden_pace_execution
    helper.project_current_target = forbidden_pace_execution
    if method == 'oscbf':
        from oscbf_independent import OSCBFSharedGeometry, OSCBFAdapterConfig
        return OSCBFSharedGeometry(helper, OSCBFAdapterConfig(clearance_buffer_m=0.)), None
    from neo_candidate_mesh import NEOIndependent, NEOAdapterConfig
    return NEOIndependent(helper, NEOAdapterConfig()), None
