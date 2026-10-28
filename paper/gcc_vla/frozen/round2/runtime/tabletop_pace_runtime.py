"""Tabletop interface for AGV-derived F2, admittance and task helpers.

Policy retains the original 8D state and per-finger gripper command. Internal
algorithms use seven targets with TOTAL gripper opening. Static table/bin geometry below is used only by placement evaluation.
Controller support geometry is supplied exclusively by RGB-D perception.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import mujoco
from agv_pace import AGVAdmittance, AGVContactConfig


def segment_to_segment_distance(p1,p2,p3,p4):
    # Shared geometry arithmetic, with no dependency on either deployment main.
    from deploy_screwdriver_client_old import segment_to_segment_distance as distance
    return distance(p1,p2,p3,p4)


class TabletopArm:
    def __init__(self, model, data, steps=50):
        self.mj=mujoco; self.model=model; self.data=data; self.steps=steps
        joints=[model.joint(f'Joint{i}').id for i in range(1,7)]
        self.qadr=model.jnt_qposadr[joints].copy(); self.dadr=model.jnt_dofadr[joints].copy()
        self.limits=model.jnt_range[joints].copy()
        self.gripper_qadr=np.array([model.joint('Joint8').qposadr[0],model.joint('Joint9').qposadr[0]])
        self.target_body=model.body('real_screwdriver').id
        self.target_geoms=np.flatnonzero(model.geom_bodyid==self.target_body)
        self.box=model.body('drop_box').id
        geoms=np.flatnonzero(model.geom_bodyid==self.box)
        bottoms=[g for g in geoms if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX and model.geom_size[g,2]<.01]
        if len(bottoms)!=1: raise ValueError('Expected one tabletop bin bottom')
        self.bin_bottom=int(bottoms[0]); walls=[int(g) for g in geoms if g!=self.bin_bottom]
        self.bin_wall_x=max(walls,key=lambda g:abs(model.geom_pos[g,0]))
        wall_y=max(walls,key=lambda g:abs(model.geom_pos[g,1]))
        self.bin_half=np.array([abs(model.geom_pos[self.bin_wall_x,0])-model.geom_size[self.bin_wall_x,0],abs(model.geom_pos[wall_y,1])-model.geom_size[wall_y,1]])
        self.table=int(np.flatnonzero(model.geom_bodyid==model.body('table').id)[0])
        self.initial_tool_position=data.xpos[self.target_body].copy()
        self.grasp_controller=None

    def state(self):
        return np.r_[self.data.qpos[self.qadr],self.data.qpos[self.gripper_qadr].sum()]

    def policy_chunk(self, chunk):
        chunk=np.asarray(chunk,dtype=float)
        if chunk.ndim!=2 or chunk.shape[1]<7 or not len(chunk) or not np.isfinite(chunk).all():
            raise ValueError('Policy must return finite (N, >=7) targets')
        out=chunk[:,:7].copy(); out[:,:6]=np.clip(out[:,:6],self.limits[:,0],self.limits[:,1])
        # Preserve old policy open/close threshold, convert to total jaw opening.
        out[:,6]=np.where(out[:,6]>.02,.08,0.)
        return out

    def control(self, target):
        target=np.asarray(target,dtype=float)
        if target.shape!=(7,) or not np.isfinite(target).all(): raise ValueError('Invalid internal target')
        if np.any(target[:6]<self.limits[:,0]-1e-8) or np.any(target[:6]>self.limits[:,1]+1e-8):
            raise RuntimeError('safety_abort: target outside joint limits')
        out=np.zeros(8); out[:6]=target[:6]; out[6:]=np.clip(target[6]/2,0,.04)
        return out

    def tool_vertices(self):
        points=[]
        for g in self.target_geoms:
            if not self.model.geom_contype[g]: continue
            mesh=int(self.model.geom_dataid[g]); begin=self.model.mesh_vertadr[mesh]; count=self.model.mesh_vertnum[mesh]
            points.append(self.model.mesh_vert[begin:begin+count]@self.data.geom_xmat[g].reshape(3,3).T+self.data.geom_xpos[g])
        return np.concatenate(points)

    def placement(self):
        d=self.data; points=self.tool_vertices(); box=d.xpos[self.box]; R=d.xmat[self.box].reshape(3,3)
        local=(points-box)@R; lo=local.min(0); hi=local.max(0); center=(lo+hi)/2
        bottom=float(((d.geom_xpos[self.bin_bottom]-box)@R)[2]+self.model.geom_size[self.bin_bottom,2])
        rim=float(((d.geom_xpos[self.bin_wall_x]-box)@R)[2]+self.model.geom_size[self.bin_wall_x,2])
        fingers=set(); bin_contact=False; obstacle_contact=False; table_contact=False; tool_table_contact=False
        finger_bodies={self.model.body('link8').id,self.model.body('link9').id}
        pillar=self.model.geom('pillar_geom').id
        for contact in d.contact:
            if contact.dist>0: continue
            g1,g2=map(int,contact.geom); b1,b2=map(int,self.model.geom_bodyid[[g1,g2]])
            if self.target_body in (b1,b2):
                other=b2 if b1==self.target_body else b1
                if other in finger_bodies: fingers.add(other)
                if other==self.box: bin_contact=True
                if other==self.model.body('table').id: tool_table_contact=True
            if pillar in (g1,g2):
                other=g2 if g1==pillar else g1
                obstacle_contact |= int(self.model.geom_bodyid[other]) in self.robot_bodies or int(self.model.geom_bodyid[other])==self.target_body
            if self.table in (g1,g2):
                other=g2 if g1==self.table else g1
                table_contact |= int(self.model.geom_bodyid[other]) in self.robot_bodies
        speed=float(np.linalg.norm(d.joint('fj_screwdriver').qvel[:3]))
        valid=bool(np.all(np.abs(center[:2])<=self.bin_half-.003) and bottom-.004<=lo[2]<=rim+.02 and bin_contact and speed<.03 and not fingers and self.state()[6]>.07)
        fully_inside=bool(valid and np.all(lo[:2]>=-self.bin_half+.003) and np.all(hi[:2]<=self.bin_half-.003) and hi[2]<=rim+.003)
        nonbox_support=bool(tool_table_contact and not bin_contact and speed<.03 and not fingers and self.state()[6]>.07)
        return dict(valid=valid,finger_contacts=len(fingers),bin_contact=bin_contact,tool_speed_m_s=speed,obstacle_contact=bool(obstacle_contact),robot_table_contact=bool(table_contact),fully_inside=fully_inside,nonbox_support=nonbox_support,tool_center_error_m=float(np.linalg.norm(center[:2])),tool_bounds_local=np.stack((lo,hi)))


class TabletopAdmittance(AGVAdmittance):
    def __init__(self,arm,config=None,*,perception):
        from pace_novelty import ConfirmedSupportEstimator
        self.arm=arm;self.config=config or AGVContactConfig(); self.dt=float(arm.model.opt.timestep);self.action_dt=self.dt*arm.steps
        self.support=ConfirmedSupportEstimator(confirmation_steps=max(1,int(np.ceil(self.config.confirmation_time_s/self.dt-1e-12))))
        self.scratch=mujoco.MjData(arm.model);self.compensation=np.zeros(3)
        from joint_torque_observer import ForceEstimate
        self.latest=ForceEstimate(np.zeros(3),np.zeros(6),False,np.zeros(3));self.latest_time=None
        self.tcp_id=arm.model.site('tcp_site').id;self.surfaces={}
        if not perception or 'box' not in perception:raise ValueError('RGB-D box/support geometry required')
        box=perception['box']; support=box['support']
        self.surfaces['workbench_top']={k:np.array(support[k],copy=True) if k!='top' else float(support[k]) for k in ('center','rotation','half_size','top')}
        self.surfaces['bin_bottom']=dict(center=np.r_[box['opening_center'][:2],box['bottom_top']],
            rotation=np.eye(3),half_size=np.r_[box['interior_half_size'],0.],top=float(box['bottom_top']))
        if any(not np.isfinite(np.asarray(value)).all() for surface in self.surfaces.values() for value in surface.values()):
            raise ValueError('Nonfinite perceived support geometry')



def save_rows(path,rows):
    if rows:
        np.savez_compressed(path,**{k:np.asarray([row[k] for row in rows]) for k in rows[0]})


def module_provenance():
    directory=Path(__file__).parent
    names=['deploy_screwdriver_client_old.py','tabletop_pace_runtime.py','tabletop_pace_preview.py','tabletop_perception.py','tabletop_visuals.py','groundingdino_obstacle.py','tabletop_box_attraction.py','tabletop_route_preference.py','tabletop_execution_preview.py','tabletop_preview_refresh.py','tabletop_preview_lift.py','tabletop_tool_proxy.py','tabletop_recording_storage.py','agv_pace.py','agv_reference_projection.py','pace_novelty.py','joint_torque_observer.py']
    names.append('tabletop_paper_ablation.py')
    return {name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in names}
