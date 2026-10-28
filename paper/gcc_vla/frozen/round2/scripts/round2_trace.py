"""Side-effect-free 1 ms observation of the existing control/physics loop.

No controller process/project/update, no mj_step/mj_forward, no policy calls.
Call before_step AFTER data.ctrl is written; after_step AFTER mj_step and the
residual observer, BEFORE the existing mj_forward; after_forward AFTER that
existing forward and placement(). Failed attempted intervals remain in the NPZ.
"""
from pathlib import Path
import json, os
import numpy as np

ROLE = {'other':0, 'robot':1, 'tool':2, 'pillar':3, 'table':4, 'bin':5, 'floor':6}
PHASE = {'approach_before_grasp':0, 'gripper_closing_before_lift':1,
         'lift_or_transport':2, 'placement_contact':3, 'unknown':-1}

def _array(x, n, default=np.nan):
    if x is None:return np.full(n,default,float)
    x=np.asarray(x,float).reshape(-1)
    if len(x)<n:raise ValueError('target/vector has insufficient dimensions')
    return x[:n].copy()

def _number(d,k,default=np.nan):
    x=d.get(k,default)
    return float(x) if x is not None else float(default)

class Round2Trace:
    def __init__(self, arm, metadata=None, geometry_helper=None, *, atol=1e-8, record_jacobian=True):
        self.arm=arm;self.metadata=dict(metadata or {});self.geometry_helper=geometry_helper
        self.atol=float(atol);self.record_jacobian=bool(record_jacobian)
        m=arm.model;self.qadr=np.asarray(arm.qadr,int).copy();self.dadr=np.asarray(arm.dadr,int).copy()
        if len(self.qadr)!=6 or len(self.dadr)!=6:raise ValueError('Expected six arm DOFs')
        self.arm_actuator_indices=[]
        for dof in self.dadr:
            ids=[i for i in range(m.nu) if int(m.actuator_trnid[i,0])>=0
                 and int(m.jnt_dofadr[int(m.actuator_trnid[i,0])])==int(dof)]
            if len(ids)!=1:raise ValueError('Trace expects one direct joint actuator per arm DOF')
            self.arm_actuator_indices.append(ids[0])
        self.arm_actuator_indices=np.asarray(self.arm_actuator_indices,int)
        self.robot_bodies=set(map(int,getattr(arm,'robot_bodies',[])))
        if not self.robot_bodies:raise ValueError('arm.robot_bodies required to match paper external-contact protection set')
        self.target_body=int(arm.target_body);self.bin_body=int(arm.box)
        self.tcp_id=int(m.site('tcp_site').id)
        self.initial_tool_height=float(getattr(geometry_helper,'initial_tool_height',
            np.asarray(getattr(arm,'initial_tool_position',arm.data.xpos[self.target_body]))[2]))
        self._lift_ever=False;self.rows=[];self.contacts=[];self.last_metric={}
        self.geom_names=[m.geom(i).name or '' for i in range(m.ngeom)]
        self.body_names=[m.body(i).name or '' for i in range(m.nbody)]
        self.roles=np.array([self._role(i) for i in range(m.ngeom)],np.int16)
        self._affine_valid=(np.asarray(m.actuator_gaintype)==0)&(np.asarray(m.actuator_biastype)==1)&(np.asarray(m.actuator_dyntype)==0)
        self.schema={}
        def fields(names,shape=(),dtype='float64'):
            for name in names.split():self.schema[name]=(tuple(shape),np.dtype(dtype))
        fields('time_before_s time_after_s')
        fields('action_index substep_index',dtype='int64')
        fields('step_completed after_forward_recorded',dtype='bool')
        fields('q_before q_after qvel_before qvel_after target_nominal_a target_geometry_uG delta_geometry target_contact_ubar delta_contact_clipped target_final_u target_post_guard_u target_applied_u u_post_guard u_applied delta_guard limiter_delta delta_u_from_previous position_error_before position_error_after actuator_force_arm qfrc_actuator_arm actuator_demand_arm', (6,))
        fields('previous_applied_valid limiter_debug_valid limiter_active estimated_contact estimate_debug_valid feedback_sample_valid phase_metric_valid_pre phase_metric_valid_post',dtype='bool')
        fields('feedback_sample_time_s')
        fields('tau_external_estimated_Nm',(6,))
        fields('actuator_at_lower_arm actuator_at_upper_arm actuator_at_limit_arm actuator_demand_exceeds_arm', (6,), 'bool')
        fields('qpos_full_before qpos_full_after',(m.nq,))
        fields('qvel_full_before qvel_full_after',(m.nv,))
        fields('ctrl_written ctrl_used_after_ctrlrange actuator_force_all actuator_demand_all',(m.nu,))
        fields('actuator_at_limit_all actuator_demand_exceeds_all ctrl_clipped_all',(m.nu,),'bool')
        fields('tcp_before tcp_after tool_pos_before tool_pos_after',(3,))
        fields('Jp_before Jomega_before',(3,6))
        fields('jacobian_valid geometry_current_valid geometry_gate_actual_valid geometry_gate_actual geometry_pickup_gate_diagnostic geometry_active_action tool_attached_diagnostic contact_debug_valid contact_gate support_before_valid support_after_valid guard_debug_valid guard_active guard_qualified guard_feasible',( ),'bool')
        fields('geometry_gap_m action_geometry_gap_m guard_bound_m guard_before_m guard_after_m support_before_m support_after_m sum_external_contact_force_N max_individual_external_contact_force_N')
        fields('gap_source closest_pair_index contact_count all_contact_count',dtype='int64')
        fields('closest_normal_world closest_point_world geometry_normal_row height_velocity_delta force_velocity_delta feedback_force_N estimated_force_N raw_force_N',(3,))
        # The normal Jacobian row has six joint coefficients, not xyz.
        self.schema['geometry_normal_row']=((6,),np.dtype('float64'))
        fields('phase_pre phase_post finger_contacts_pre finger_contacts_post',dtype='int64')
        fields('gripper_width_measured_pre gripper_width_measured_post tool_z_relative_initial_pre tool_z_relative_initial_post')
        fields('tool_lifted_pre tool_lifted_post lift_ever_pre lift_ever_post gripper_closed_command_pre gripper_closed_command_post bin_contact_pre bin_contact_post placement_valid_pre placement_valid_post tool_in_bin_xy_pre tool_in_bin_xy_post',(), 'bool')
        fields('grasp_seen strict_valid tool_floor_contact pillar_contact_episode pillar_tilt_episode',(),'bool')
        fields('strict_frames',dtype='int64')
        self._contacts_schema={
            'geom1':((), 'int64'),'geom2':((), 'int64'),'body1':((), 'int64'),'body2':((), 'int64'),
            'role1':((), 'int16'),'role2':((), 'int16'),'distance_m':((), 'float64'),
            'dim':((), 'int64'),'efc_address':((), 'int64'),'position_world_m':((3,),'float64'),
            'frame_rows_world':((3,3),'float64'),'wrench_contact_on_geom2':((6,),'float64'),
            'wrench_world_on_geom2':((6,),'float64'),'protected_external':((),'bool'),'protected_internal':((),'bool')}
        self.metadata.update(schema_version='round2_trace_v1',timestep_s=float(m.opt.timestep),
            action_period_s=float(m.opt.timestep*arm.steps),qpos_indices=self.qadr.tolist(),dof_indices=self.dadr.tolist(),
            arm_actuator_indices=self.arm_actuator_indices.tolist(),actuator_names=[m.actuator(i).name for i in range(m.nu)],
            actuator_forcerange=np.asarray(m.actuator_forcerange).tolist(),actuator_forcelimited=np.asarray(m.actuator_forcelimited,bool).tolist(),
            actuator_ctrlrange=np.asarray(m.actuator_ctrlrange).tolist(),actuator_ctrllimited=np.asarray(m.actuator_ctrllimited,bool).tolist(),
            actuator_gear=np.asarray(m.actuator_gear).tolist(),affine_demand_valid=self._affine_valid.tolist(),
            geom_names=self.geom_names,body_names=self.body_names,role_codes=ROLE,phase_codes=PHASE,
            control_semantics='ctrl_written are absolute position-actuator references; arm rad, two gripper slide positions m; not torque commands',
            units={'arm_targets_q':'rad','arm_velocity':'rad/s','arm_actuator_force_and_qfrc':'N*m for verified direct hinge actuators','gripper_actuator_force':'N for slide actuators','world_force':'N','world_contact_moment_about_contact_point':'N*m'},
            actual_gate_semantics='geometry_gate_actual uses guard qualification if provided, else cached action-level pickup gate; diagnostic pickup gate is separately named',
            target_semantics='all target/delta arrays contain six arm rad; final_target API is post-guard; target_final_u/u_applied is actual post-limiter target; total gripper opening reflected in ctrl_written',
            contact_semantics='wrench acts ON geom2; frame rows transform world->contact, world wrench=frame.T applied to force and moment; not a wrench shifted to TCP',
            phase_semantics='observable state predicates with precedence bin-contact/valid, lifted/ever-lifted, closed-command, approach; not a ground-truth task segment or causal label',
            sampling='before: after ctrl write before mj_step; force/contacts: after mj_step+observer before existing mj_forward; after_forward: existing refreshed kinematics/metric',
            attribution='module magnitudes are correlated command contributions, NOT causal attribution; intervention contrast required',
            missing='NaN with explicit validity flag denotes unavailable; disabled contact/guard have zero applied increment, not reconstructed historical signals')

    def _role(self,geom):
        b=int(self.arm.model.geom_bodyid[geom]);gn=self.geom_names[geom];bn=self.body_names[b]
        if b==self.target_body:return ROLE['tool']
        if b in self.robot_bodies:return ROLE['robot']
        if gn=='pillar_geom' or bn=='dynamic_pillar':return ROLE['pillar']
        if bn=='table':return ROLE['table']
        if b==self.bin_body:return ROLE['bin']
        if gn in ('floor','ground') or bn in ('floor','ground'):return ROLE['floor']
        return ROLE['other']

    def _new_row(self):
        return {k:np.full(shape,np.nan if dt.kind=='f' else -1 if dt.kind in 'iu' else False,dtype=dt)
                for k,(shape,dt) in self.schema.items()}

    def _phase(self,arm,target,metric):
        d=arm.data;tool=np.asarray(d.xpos[self.target_body]);zrel=float(tool[2]-self.initial_tool_height)
        lifted=zrel>.015;self._lift_ever|=lifted
        width=float(np.asarray(d.qpos)[np.asarray(arm.gripper_qadr,int)].sum())
        closed=bool(np.asarray(target).reshape(-1)[6]<=.04) if len(np.asarray(target).reshape(-1))>6 else False
        loc=np.asarray(d.xmat[self.bin_body]).reshape(3,3).T@(tool-d.xpos[self.bin_body])
        inxy=bool(np.all(np.abs(loc[:2])<=np.asarray(arm.bin_half)))
        bc=bool(metric.get('bin_contact',False));valid=bool(metric.get('valid',False))
        phase=3 if bc or valid else 2 if self._lift_ever else 1 if closed else 0
        return dict(phase=phase,finger_contacts=int(metric.get('finger_contacts',-1)),gripper_width_measured=width,
            tool_z_relative_initial=zrel,tool_lifted=lifted,lift_ever=self._lift_ever,
            gripper_closed_command=closed,bin_contact=bc,placement_valid=valid,tool_in_bin_xy=inxy)

    def _geometry(self,arm,target,row):
        h=self.geometry_helper
        if h is None:return
        # Only free functions that read existing kinematics and cached geometry.
        from tabletop_tool_proxy import world_body_capsules,world_tool_capsules,batch_capsule_clearances
        d=arm.data;tcp=d.site_xpos[self.tcp_id];tool=d.xpos[self.target_body]
        attached=bool(np.asarray(target)[6]<=.04 and np.linalg.norm(tcp-tool)<.05)
        caps=world_body_capsules(d,h.eef_capsules_local)
        if attached:caps+=world_tool_capsules(d,self.target_body,h.tool_capsules_local)
        values,normals,points=batch_capsule_clearances(caps,h.obstacle);i=int(np.argmin(values))
        row.update(geometry_current_valid=True,geometry_gap_m=float(values[i]),gap_source=2,
            closest_pair_index=i,closest_normal_world=normals[i].copy(),closest_point_world=points[i].copy(),tool_attached_diagnostic=attached)
        if row['jacobian_valid']:row['geometry_normal_row']=np.asarray(normals[i])@row['Jp_before']

    def before_step(self,arm,action_index,substep,nominal=None,geometry_target=None,contact_target=None,final_target=None,
                    geometry_debug=None,contact_debug=None,guard_debug=None,previous_applied=None,
                    *,nominal_target=None,applied_target=None,limiter_debug=None):
        g=geometry_debug or {};c=contact_debug or {};p=guard_debug or {};d=arm.data
        if nominal is None:nominal=nominal_target
        if nominal is None or geometry_target is None or contact_target is None or final_target is None:raise ValueError('four explicit control references required')
        if applied_target is None:applied_target=final_target
        row=self._new_row();a=_array(nominal,6);ug=_array(geometry_target,6);ct=_array(contact_target,6);postguard=_array(final_target,6);u=_array(applied_target,6)
        q=np.asarray(d.qpos)[self.qadr].copy();v=np.asarray(d.qvel)[self.dadr].copy()
        row.update(action_index=int(action_index),substep_index=int(substep),time_before_s=float(d.time),
            q_before=q,qvel_before=v,qpos_full_before=d.qpos.copy(),qvel_full_before=d.qvel.copy(),
            target_nominal_a=a,target_geometry_uG=ug,delta_geometry=ug-a,target_contact_ubar=ct,
            delta_contact_clipped=ct-ug,target_final_u=u,target_post_guard_u=postguard,target_applied_u=u,u_post_guard=postguard,u_applied=u,delta_guard=postguard-ct,limiter_delta=u-postguard,limiter_debug_valid=limiter_debug is not None,limiter_active=bool(np.any(np.abs(u-postguard)>1e-14)),position_error_before=u-q,
            ctrl_written=d.ctrl.copy(),tcp_before=d.site_xpos[self.tcp_id].copy(),tool_pos_before=d.xpos[self.target_body].copy())
        previous_valid=previous_applied is not None
        row['previous_applied_valid']=previous_valid
        if previous_valid:row['delta_u_from_previous']=u-_array(previous_applied,6)
        ctrl=d.ctrl.copy();limited=np.asarray(arm.model.actuator_ctrllimited,bool)
        ctrl[limited]=np.clip(ctrl[limited],arm.model.actuator_ctrlrange[limited,0],arm.model.actuator_ctrlrange[limited,1])
        row.update(ctrl_used_after_ctrlrange=ctrl,ctrl_clipped_all=np.abs(ctrl-d.ctrl)>0,
            contact_debug_valid=contact_debug is not None,contact_gate=bool(c.get('contact_gate',False)),
            support_before_valid=bool(c.get('support_before_valid',False)),support_before_m=_number(c,'support_before_m'),
            geometry_active_action=bool(g.get('geometry_active',False)),action_geometry_gap_m=_number(g,'current_clearance_m'),feedback_sample_valid=bool(c.get('feedback_sample_valid',False)),feedback_sample_time_s=_number(c,'feedback_sample_time_s'),phase_metric_valid_pre=bool(self.last_metric),
            guard_debug_valid=guard_debug is not None,guard_active=bool(p.get('reference_guard_active',False)),
            guard_qualified=bool(p.get('reference_guard_qualified',False)),guard_feasible=bool(p.get('reference_guard_feasible',True)),
            guard_bound_m=_number(p,'reference_guard_bound_m'),guard_before_m=_number(p,'reference_guard_before_m'),guard_after_m=_number(p,'reference_guard_after_m'))
        for name,key in [('height_velocity_delta','height_delta_m_s'),('force_velocity_delta','force_delta_m_s'),('feedback_force_N','feedback_force_N')]:
            row[name]=_array(c.get(key,np.zeros(3)),3)
        if self.record_jacobian:
            jp=np.zeros((3,arm.model.nv));jw=np.zeros_like(jp)
            arm.mj.mj_jacSite(arm.model,d,jp,jw,self.tcp_id)
            row.update(Jp_before=jp[:,self.dadr].copy(),Jomega_before=jw[:,self.dadr].copy(),jacobian_valid=True)
        self._geometry(arm,final_target,row)
        if 'reference_guard_clearance_m' in p:
            row.update(geometry_gap_m=float(p['reference_guard_clearance_m']),gap_source=1,geometry_current_valid=True)
        if 'reference_guard_qualified' in p:row.update(geometry_gate_actual_valid=True,geometry_gate_actual=bool(p['reference_guard_qualified']))
        elif 'pickup_gate' in g:row.update(geometry_gate_actual_valid=True,geometry_gate_actual=bool(g['pickup_gate']))
        for k,val in self._phase(arm,final_target,self.last_metric).items():row[k+'_pre']=val
        row['geometry_pickup_gate_diagnostic']=self._lift_ever
        self.rows.append(row);self.contacts.append([])
        return row

    def after_step(self,arm,row,*,estimate=None,contact_debug=None,metric=None,phase_metric=None):
        """Must run BEFORE the existing mj_forward; does not refresh any state."""
        d=arm.data;m=arm.model
        if row is not self.rows[-1] or row['step_completed']:raise ValueError('sample each pending interval once')
        idx=self.arm_actuator_indices;af=np.asarray(d.actuator_force).copy();limits=np.asarray(m.actuator_forcerange)
        limited=np.asarray(m.actuator_forcelimited,bool)
        lower=limited&(np.abs(af-limits[:,0])<=self.atol);upper=limited&(np.abs(af-limits[:,1])<=self.atol)
        demand=m.actuator_gainprm[:,0]*row['ctrl_used_after_ctrlrange']+m.actuator_biasprm[:,0]+m.actuator_biasprm[:,1]*d.actuator_length+m.actuator_biasprm[:,2]*d.actuator_velocity
        demand=np.asarray(demand,float);demand[~self._affine_valid]=np.nan
        exceeds=limited&((demand<limits[:,0])|(demand>limits[:,1]))
        row.update(time_after_s=float(d.time),step_completed=True,q_after=d.qpos[self.qadr].copy(),qvel_after=d.qvel[self.dadr].copy(),qpos_full_after=d.qpos.copy(),qvel_full_after=d.qvel.copy(),
            position_error_after=row['target_final_u']-d.qpos[self.qadr],actuator_force_all=af,actuator_force_arm=af[idx],qfrc_actuator_arm=d.qfrc_actuator[self.dadr].copy(),
            actuator_demand_all=demand,actuator_demand_arm=demand[idx],actuator_at_lower_arm=lower[idx],actuator_at_upper_arm=upper[idx],actuator_at_limit_arm=(lower|upper)[idx],
            actuator_at_limit_all=lower|upper,actuator_demand_exceeds_arm=exceeds[idx],actuator_demand_exceeds_all=exceeds,
            support_after_valid=bool((contact_debug or {}).get('support_after_valid',False)),support_after_m=_number(contact_debug or {},'support_after_m'))
        if estimate is not None:
            row['estimated_force_N']=_array(estimate.force,3);row['raw_force_N']=_array(estimate.raw_force,3)
            row['estimated_contact']=bool(getattr(estimate,'contact',False));row['estimate_debug_valid']=True
            row['tau_external_estimated_Nm']=_array(getattr(estimate,'tau_external',None),6)
        entries=[];total=0.;peak=0.;external_count=0
        for i in range(int(d.ncon)):
            co=d.contact[i];g1=int(co.geom1);g2=int(co.geom2);b1=int(m.geom_bodyid[g1]);b2=int(m.geom_bodyid[g2])
            w=np.zeros(6);arm.mj.mj_contactForce(m,d,i,w);frame=np.asarray(co.frame).reshape(3,3).copy()
            world=np.r_[frame.T@w[:3],frame.T@w[3:]]
            first=b1 in self.robot_bodies or b1==self.target_body;second=b2 in self.robot_bodies or b2==self.target_body;external=first!=second
            if external:amount=float(np.linalg.norm(w[:3]));total+=amount;peak=max(peak,amount);external_count+=1
            entries.append(dict(geom1=g1,geom2=g2,body1=b1,body2=b2,role1=self.roles[g1],role2=self.roles[g2],distance_m=float(co.dist),dim=int(co.dim),efc_address=int(co.efc_address),position_world_m=np.asarray(co.pos).copy(),frame_rows_world=frame,wrench_contact_on_geom2=w,wrench_world_on_geom2=world,protected_external=external,protected_internal=bool(first and second)))
        self.contacts[-1]=entries;row.update(sum_external_contact_force_N=total,max_individual_external_contact_force_N=peak,contact_count=external_count,all_contact_count=len(entries))
        if metric is not None:self.after_forward(arm,row,metric=metric,phase_metric=phase_metric)
        return row

    def after_forward(self,arm,row,*,metric,phase_metric=None):
        """Call only AFTER caller's existing mj_forward and placement() metric."""
        row.update(tcp_after=arm.data.site_xpos[self.tcp_id].copy(),tool_pos_after=arm.data.xpos[self.target_body].copy(),after_forward_recorded=True,phase_metric_valid_post=metric is not None)
        self.last_metric=dict(metric or {})
        # Direct observable predicates are stored so heuristic phase bins are auditable.
        full_target=np.r_[row['target_final_u'],float(np.asarray(row['ctrl_written'])[-2:].sum())]
        for k,val in self._phase(arm,full_target,self.last_metric).items():row[k+'_post']=val
        pm=phase_metric or {}
        for k in ('grasp_seen','strict_valid','tool_floor_contact','pillar_contact_episode','pillar_tilt_episode'):
            row[k]=bool(pm.get(k,False))
        row['strict_frames']=int(pm.get('strict_frames',-1))
        return row

    def save(self,path):
        path=Path(path)
        if path.exists():raise FileExistsError('never overwrite an existing episode trace: '+str(path))
        path.parent.mkdir(parents=True,exist_ok=True)
        arrays={k:np.asarray([r[k] for r in self.rows],dtype=dt).reshape((len(self.rows),*shape)) for k,(shape,dt) in self.schema.items()}
        offsets=np.r_[0,np.cumsum([len(cs) for cs in self.contacts])].astype(np.int64);arrays['contact_offsets']=offsets
        flat=[c for cs in self.contacts for c in cs]
        for k,(shape,dt) in self._contacts_schema.items():arrays['contact_'+k]=np.asarray([c[k] for c in flat],dtype=dt).reshape((len(flat),*shape))
        meta=dict(self.metadata,episode_first_logged_time_s=float(self.rows[0]['time_before_s']) if self.rows else None,samples=len(self.rows),completed_samples=sum(bool(r['step_completed']) for r in self.rows),ragged_contacts=len(flat),force_limit_tolerance=self.atol)
        arrays['metadata_json']=np.asarray(json.dumps(meta,ensure_ascii=False,sort_keys=True))
        temporary=path.with_name(path.name+'.tmp')
        with temporary.open('xb') as f:np.savez_compressed(f,**arrays)
        os.replace(temporary,path)
        return path


def validate_trace_npz(path,*,require_current_geometry=True,require_after_forward=True):
    """Offline schema/identity validation only. Does not simulate or infer causes."""
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata_json']));n=len(z['time_before_s']);done=z['step_completed'].astype(bool);off=z['contact_offsets']
        assert off.shape==(n+1,) and off[0]==0 and np.all(np.diff(off)>=0)
        assert off[-1]==len(z['contact_geom1']);assert len(z['contact_wrench_contact_on_geom2'])==off[-1]
        assert z['actuator_force_arm'].shape==z['qfrc_actuator_arm'].shape==(n,6)
        counts=np.diff(off);owners=np.repeat(np.arange(n),counts)
        magnitudes=np.linalg.norm(z['contact_wrench_contact_on_geom2'][:,:3],axis=1)*z['contact_protected_external']
        sums=np.bincount(owners,weights=magnitudes,minlength=n)
        assert np.allclose(sums[done],z['sum_external_contact_force_N'][done],atol=1e-10,rtol=1e-12)
        assert np.allclose(z['ctrl_written'][:,meta['arm_actuator_indices']],z['u_applied'],equal_nan=True), 'before_step must run AFTER actual ctrl write'
        assert np.allclose(z['target_geometry_uG']-z['target_nominal_a'],z['delta_geometry'],equal_nan=True)
        assert np.allclose(z['target_contact_ubar']-z['target_geometry_uG'],z['delta_contact_clipped'],equal_nan=True)
        assert np.allclose(z['target_post_guard_u']-z['target_contact_ubar'],z['delta_guard'],equal_nan=True)
        assert np.allclose(z['u_applied']-z['u_post_guard'],z['limiter_delta'],equal_nan=True)
        assert np.allclose(z['target_final_u']-z['q_before'],z['position_error_before'],equal_nan=True)
        assert np.allclose((z['time_after_s']-z['time_before_s'])[done],meta['timestep_s'],atol=1e-9,rtol=0)
        if require_current_geometry:assert np.all(z['geometry_current_valid'][done]),'missing true current geometry; action-level cached gap is insufficient'
        if require_after_forward:assert np.all(z['after_forward_recorded'][done]),'post-forward predicates missing'
        good=z['previous_applied_valid'].astype(bool)
        if n>1:
            adjacent=good[1:]&done[:-1]
            assert np.allclose(z['delta_u_from_previous'][1:][adjacent],(z['target_final_u'][1:]-z['target_final_u'][:-1])[adjacent])
        return dict(samples=n,completed=int(done.sum()),attempted_not_completed=int((~done).sum()),contacts=int(off[-1]),
            all_required_fields=True,reaches_limit_substeps=int(np.any(z['actuator_at_limit_arm'][done],axis=1).sum()),
            first_limit_index=int(np.flatnonzero(done&np.any(z['actuator_at_limit_arm'],axis=1))[0]) if np.any(done&np.any(z['actuator_at_limit_arm'],axis=1)) else None,
            causal_attribution=False)
