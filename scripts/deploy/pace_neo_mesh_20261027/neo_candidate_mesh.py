"""NEO-only exact robot mesh / perceived capsule collision adaptation.

No shared source is changed. A private model is used for distance queries only;
the virtual obstacle geometry and pose are overwritten from frozen RGB-D input.
The live simulation model is never changed by this adapter.
"""
import copy
from dataclasses import replace
import numpy as np
from neo_tracking_parent import NEOIndependent as Parent,NEOAdapterConfig

class NEOIndependent(Parent):
    def __init__(self,helper,config=None):
        # 5 mm is an explicit small-arm adaptation, not the original 50 mm.
        super().__init__(helper,replace(config or NEOAdapterConfig(),obstacle_stop_m=.005))
        mj=self.arm.mj;m=self.arm.model
        self._distance_model=copy.copy(m)
        self._distance_data=mj.MjData(self._distance_model)
        self._virtual_geom=self._distance_model.geom('pillar_geom').id
        p1,p2=np.asarray(self.obstacle['p1']),np.asarray(self.obstacle['p2'])
        axis=p2-p1;length=float(np.linalg.norm(axis))
        if not np.isfinite(length) or length<=1e-8:raise ValueError('Degenerate perceived obstacle')
        axis=axis/length
        reference=np.array([1.,0.,0.]) if abs(axis[0])<.9 else np.array([0.,1.,0.])
        x=reference-axis*(reference@axis);x=x/np.linalg.norm(x);y=np.cross(axis,x)
        self._obstacle_rotation=np.column_stack([x,y,axis]);self._obstacle_center=(p1+p2)/2
        dm=self._distance_model;v=self._virtual_geom
        dm.geom_type[v]=mj.mjtGeom.mjGEOM_CAPSULE
        dm.geom_size[v]=[float(self.obstacle['r']),length/2,0.]
        dm.geom_rbound[v]=float(self.obstacle['r'])+length/2
        bodies={int(c['body_id']) for c in helper.eef_capsules_local}
        self._robot_geoms=[int(g) for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies and m.geom_contype[g]]
        if not self._robot_geoms:raise ValueError('No protected robot mesh geoms')
        if any(m.geom_type[g]!=mj.mjtGeom.mjGEOM_MESH for g in self._robot_geoms):raise ValueError('Protected robot geom must be convex mesh')
        self._last_geom_ids=[]

    def _geometry(self,target):
        h=self.helper;a=self.arm;mj=a.mj;h.prepare(h.current)
        attached=bool(target[6]<=.04 and np.linalg.norm(h.current.site_xpos[h.tcp_id]-h.current.xpos[h.tool_id])<.05)
        d=self._distance_data;m=self._distance_model
        d.qpos[:]=a.data.qpos;mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
        d.geom_xpos[self._virtual_geom]=self._obstacle_center
        d.geom_xmat[self._virtual_geom]=self._obstacle_rotation.ravel()
        geoms=self._robot_geoms+(a.target_geoms.tolist() if attached else [])
        gaps=[];normals=[];points=[];bodies=[]
        for g in geoms:
            segment=np.zeros(6)
            gap=float(mj.mj_geomDistance(m,d,g,self._virtual_geom,1.,segment))
            # This MuJoCo build returns shape-sorted closest points. Identify the
            # capsule endpoint from the perceived capsule surface, not argument order.
            ends=segment.reshape(2,3);axis=self._obstacle_rotation[:,2];half=float(m.geom_size[self._virtual_geom,1])
            projection=np.clip((ends-self._obstacle_center)@axis,-half,half)
            residual=np.linalg.norm(ends-self._obstacle_center-projection[:,None]*axis,axis=1)-float(self.obstacle['r'])
            obstacle_end=int(np.argmin(np.abs(residual)));robot_point=ends[1-obstacle_end]
            delta=robot_point-ends[obstacle_end];length=float(np.linalg.norm(delta))
            if not np.isfinite(gap) or not np.isfinite(segment).all():raise ValueError('Invalid mesh distance')
            if gap>=.999999:
                # Outside the 300 mm influence range; no constraint uses the normal.
                normal=np.array([1.,0.,0.])
            elif abs(gap)>1e-9 and length>1e-9:
                normal=delta/gap;normal=normal/np.linalg.norm(normal)
            else:
                raise ValueError('Unqualified zero-distance mesh normal; terminate without fallback')
            gaps.append(gap);normals.append(normal);points.append(robot_point)
            bodies.append(h.wrist_id if int(a.model.geom_bodyid[g])==h.tool_id else int(a.model.geom_bodyid[g]))
        self._last_geom_ids=geoms
        return np.array(gaps),np.array(normals),np.array(points),bodies,attached

    def project_current_target(self,target):
        output,debug=super().project_current_target(target)
        debug['neo_geometry']='convex_robot_mesh_vs_perceived_capsule'
        debug['neo_protected_geom_ids']=','.join(map(str,self._last_geom_ids))
        return output,debug

    def metadata(self):
        meta=super().metadata()
        meta.update(algorithm='NEO_pi0_tracking_mesh_geometry_5mm_adapter',collision_geometry='Known convex robot collision meshes; perceived RGB-D capsule; private model distance query only',obstacle_stop_m=.005,shared_source_modified=False,shared_geometry='No common geometry source modified; private convex mesh distance adapter')
        meta['differences']+=['Known robot mesh distance replaces enclosing robot component capsule distance; 5mm stop replaces 50mm', 'No obstacle ground-truth geometry or pose is used by the private distance model']
        return meta
