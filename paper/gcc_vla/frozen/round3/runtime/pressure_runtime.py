"""Process-local adapter for separately registered pressure-scene cohorts.

Loads an admitted zero-policy preflight MJB and restores its settled initial
integration state. Does not perform physics, policy inference, perception, or
controller construction. This reset protocol differs from the original live
reset+settle and must be reported as such, with preflight settling provenance.
"""
from contextlib import contextmanager,ExitStack
from pathlib import Path
from unittest.mock import patch
import hashlib,json,threading
import numpy as np

_CONTEXT_LOCK=threading.Lock()


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for part in iter(lambda:file.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def _physical_signature(model):
    # Fields changed by the original configure_model plus physical morphology,
    # integrator timestep and slip option; not a substitute for source MJB hash.
    names=('body_pos','body_quat','body_mass','body_inertia','geom_pos','geom_quat',
           'geom_size','geom_friction','geom_solref','geom_solimp','jnt_pos','jnt_axis','jnt_range')
    h=hashlib.sha256()
    for key in names:
        x=np.asarray(getattr(model,key));h.update(key.encode());h.update(x.tobytes())
    h.update(repr((float(model.opt.timestep),int(model.opt.noslip_iterations))).encode())
    return h.hexdigest()


@contextmanager
def pressure_scene(deploy,scene_folder):
    """Yield initialization provenance; apply only to owning thread/models.

    Use around the unchanged deploy.main in a dedicated pressure worker. The
    resulting live perception still renders and runs DINO normally. Preflight
    detected geometry is never supplied as a control target.
    """
    folder=Path(scene_folder).resolve(strict=True)
    model_path=folder/'model.mjb';state_path=folder/'initial_state.npz'
    record_path=folder/'preflight_result.json'
    record=json.loads(record_path.read_text())
    if record.get('admitted') is not True:raise ValueError('Pressure scene was not admitted by zero-policy preflight')
    model_hash=_sha(model_path);state_hash=_sha(state_path)
    if model_hash!=record.get('model_sha256'):raise ValueError('Pressure MJB differs from admitted preflight')
    with np.load(state_path,allow_pickle=False) as source:
        state=np.asarray(source['state'],dtype=np.float64).copy();spec=int(source['spec'])
    mj=deploy.mujoco
    if spec!=int(mj.mjtState.mjSTATE_INTEGRATION):raise ValueError('Expected full integration state')
    if state.shape!=(187,) or not np.isfinite(state).all():raise ValueError('Expected original finite 187D integration state')
    owner=threading.get_ident();owned={};models=[]
    original_loader=mj.MjModel.from_xml_path;binary_loader=mj.MjModel.from_binary_path
    original_reset=deploy.reset_scene;original_settle=deploy.settle_pillar
    receipt=dict(scene_id=record['id'],scene_folder=str(folder),model_path=str(model_path),
        model_sha256=model_hash,state_path=str(state_path),state_sha256=state_hash,
        preflight_record_path=str(record_path),preflight_record_sha256=_sha(record_path),
        state_spec=spec,state_dimension=187,initial_time_s=float(state[0]),
        preflight_settling_physics_steps=int(record['settling_physics_steps']),
        additional_settling_physics_steps=0,policy_calls_by_adapter=0,
        initialization='restore preflight settled mjSTATE_INTEGRATION, then mj_forward; original settle skipped',
        scope='pressure cohort only; unchanged main scene reset+settle protocol is distinct',
        control_geometry_source='normal live DINO/RGB-D; no preflight detected geometry is passed to controllers',
        model_loads=0,state_restores=0,settle_calls_skipped=0)

    def load_model(xml_path,*args,**kwargs):
        if threading.get_ident()!=owner:return original_loader(xml_path,*args,**kwargs)
        if args or kwargs:raise ValueError('Pressure adapter expects unchanged deploy.main XML-path-only loader')
        if _sha(model_path)!=model_hash:raise RuntimeError('Frozen pressure MJB changed during context')
        model=binary_loader(str(model_path))
        if mj.mj_stateSize(model,spec)!=187:raise ValueError('MJB/state dimensional mismatch')
        owned[id(model)]=_physical_signature(model);models.append(model)
        receipt['model_loads']+=1
        return model

    def reset(model,data,use_obstacle=False,target_xy=None):
        if id(model) not in owned or threading.get_ident()!=owner:
            return original_reset(model,data,use_obstacle=use_obstacle,target_xy=target_xy)
        if not use_obstacle:raise ValueError('Registered pressure scenes require the pillar condition')
        if target_xy is not None and not np.array_equal(np.asarray(target_xy,float),np.array([.28,-.10])):
            raise ValueError('Pressure initialization has frozen fixed-evaluation tool state')
        if _physical_signature(model)!=owned[id(model)]:raise RuntimeError('Main changed preflight physical model configuration')
        if _sha(state_path)!=state_hash:raise RuntimeError('Frozen pressure state changed during context')
        mj.mj_resetData(model,data)
        mj.mj_setState(model,data,state,spec)
        mj.mj_forward(model,data)
        receipt['state_restores']+=1

    def settle(model,data,*args,**kwargs):
        if id(model) not in owned or threading.get_ident()!=owner:
            return original_settle(model,data,*args,**kwargs)
        receipt['settle_calls_skipped']+=1
        return 0

    if not _CONTEXT_LOCK.acquire(blocking=False):raise RuntimeError('Overlapping pressure initialization contexts in one process are forbidden')
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(mj.MjModel,'from_xml_path',new=staticmethod(load_model)))
            stack.enter_context(patch.object(deploy,'reset_scene',new=reset))
            stack.enter_context(patch.object(deploy,'settle_pillar',new=settle))
            yield receipt
    finally:
        _CONTEXT_LOCK.release()
