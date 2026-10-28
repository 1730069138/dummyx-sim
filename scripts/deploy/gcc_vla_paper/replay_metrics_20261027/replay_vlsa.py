"""Reconstruct archived VLSA controls; no policy, QP, renderer, or live source edits.

All reconstruction uses recorded q-start and recorded safe-minus-nominal twist.
Physical replay is open loop. No state reset is permitted after initialization.
Validation compares all archived substeps/actions before admitting load metrics.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys
sys.dont_write_bytecode=True
import ast, csv, json, hashlib, argparse, time
from pathlib import Path
import numpy as np
import mujoco

ROOT=Path('/home/jun/dummyx-sim')
PACKAGE=ROOT/'outputs/pace_latest_evidence_20261027_1400'
FROZEN=PACKAGE/'脚本与身份/共享运行时'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_def(filename,names):
    p=Path(filename);tree=ast.parse(p.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise ValueError('Missing requested frozen definition')
    env={'np':np,'mujoco':mujoco}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),env)
    return [env[n] for n in names]

contact_evaluation,=load_def(FROZEN/'tabletop_paper_ablation.py',['contact_evaluation'])
arm_body_ids,=load_def(FROZEN/'joint_torque_observer.py',['arm_body_ids'])
tool_ground_contact,=load_def(FROZEN/'deploy_screwdriver_client_old.py',['tool_ground_contact'])
TabletopArm,=load_def(FROZEN/'tabletop_pace_runtime.py',['TabletopArm'])
get_legacy_force,=load_def(PACKAGE/'脚本与身份/VLSA/deploy_vlsa_aligned_v6.py',['get_target_table_force'])

def replay(row,outdir):
    job=Path(row['package_job']);side=Path(row['package_data']);datafile=side.with_name('data_ep_001.npz')
    identity=json.loads((job/'aligned_initialization.json').read_text())
    modelpath=PACKAGE/'模型'/Path(identity['model_path']).name
    if not modelpath.exists():modelpath=Path(identity['model_path'])
    if sha(modelpath)!=identity['model_sha256']:raise ValueError('Model hash mismatch')
    model=mujoco.MjModel.from_binary_path(str(modelpath));data=mujoco.MjData(model);scratch=mujoco.MjData(model)
    with np.load(job/'initial_pre_action_state.npz',allow_pickle=False) as z:initial=z['state'].copy();spec=int(z['spec'])
    if spec!=int(mujoco.mjtState.mjSTATE_INTEGRATION) or len(initial)!=mujoco.mj_stateSize(model,spec):raise ValueError('Incomplete initial state')
    mujoco.mj_setState(model,data,initial,spec);mujoco.mj_forward(model,data)
    with np.load(datafile,allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
    with np.load(side,allow_pickle=False) as z:record={k:z[k].copy() for k in z.files}
    n=len(old['q_pos_vla']);actions=record['actions'];subs=record['substeps'];valid=actions[:,2].astype(int)
    if not (len(actions)==n and len(subs)==int(valid.sum()) and str(old['vlsa_action_space'])=='cartesian' and np.all(old['cbf_applied'])):raise ValueError('Action schema mismatch')
    raw=np.concatenate([np.load(p,allow_pickle=False)['actions'] for p in sorted(job.glob('policy_chunk_*.npz'))])[:n]
    if raw.shape!=(n,7) or not np.array_equal(raw[:,:6],old['q_pos_vla']) or not np.array_equal(raw[:,6],old['gripper_cmd']):raise ValueError('Policy archive mismatch')
    qstarts=old['q_pos_vla']-old['q_dot_vla'];commands=[];nominal_error=0.
    link=model.body('link7').id;tcp=model.site('tcp_site').id;tool=model.body('real_screwdriver').id;pillar=model.body('dynamic_pillar').id
    local=old['eef_center_in_link7'];dt=float(model.opt.timestep);adt=float(old['control_period_s'])
    if abs(dt-.001)>1e-12 or abs(adt-.05)>1e-12:raise ValueError('Nonmatching control periods')
    for k in range(n):
        scratch.qpos[:]=data.qpos;scratch.qpos[:6]=qstarts[k]
        mujoco.mj_kinematics(model,scratch);mujoco.mj_comPos(model,scratch)
        center=scratch.xpos[link]+scratch.xmat[link].reshape(3,3)@local
        jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
        mujoco.mj_jac(model,scratch,jp,jr,center,link);J=np.vstack((jp[:,:6],jr[:,:6]))
        delta=np.clip(old['q_pos_vla'][k]-qstarts[k],-.25,.25);v=delta/adt
        nominal_error=max(nominal_error,float(np.max(np.abs(J@v-np.r_[old['v_nom'][k],old['omega_nom'][k]]))))
        twist=np.r_[old['v_safe'][k]-old['v_nom'][k],old['omega_safe'][k]-old['omega_nom'][k]]
        pinv=J.T@np.linalg.inv(J@J.T+.05**2*np.eye(6))
        safe=np.clip((v+pinv@twist)*adt,-.25,.25)
        u=np.zeros(8);u[:6]=qstarts[k]+safe;u[6:]=.04 if old['gripper_cmd'][k]>.02 else 0.
        commands.append(u)
    commands=np.array(commands)
    arm=TabletopArm(model,data);arm.robot_bodies=arm_body_ids(model);protected=arm.robot_bodies|{tool}
    validbody=[tool,link,model.body('link8').id,model.body('link9').id]
    start=float(data.time);initial_height=float(record['initial_height']);maxheight=initial_height;grasp=False;floor=False;tilt=False;hit=False;strict_frames=0;off=0
    errors=dict(q_start_rad=0.,tcp_end_m=0.,eef_end_m=0.,tool_end_m=0.,tool_z_substep_m=0.,time_s=0.,max_height_m=0.,pillar_up=0.)
    mismatches=dict(pillar_contact=0,finger_count=0,floor_latch=0,placement_valid=0,grasp_latch=0,strict_valid=0,strict_frames=0)
    load=[];legacy_impulse=0.;legacy_peak=0.;taupeak=0.;tick=time.perf_counter()
    for k in range(n):
        errors['q_start_rad']=max(errors['q_start_rad'],float(np.max(np.abs(data.qpos[:6]-qstarts[k]))))
        placement_count=0
        for j in range(valid[k]):
            f=get_legacy_force(model,data,validbody);norm=float(np.linalg.norm(f));legacy_impulse+=norm*dt;legacy_peak=max(legacy_peak,norm)
            data.ctrl[:8]=commands[k]
            mujoco.mj_step(model,data)
            # Same post-integration, pre-forward boundary as archived paper_truth.
            load.append(contact_evaluation(model,data,protected))
            taupeak=max(taupeak,float(np.max(np.abs(data.qfrc_actuator[:6]))))
            mujoco.mj_forward(model,data)
            metric=arm.placement();grasp|=metric['finger_contacts']==2;maxheight=max(maxheight,float(data.xpos[tool,2]));floor|=tool_ground_contact(model,data,arm.target_geoms);hit|=metric['obstacle_contact'];placement_count+=int(metric['valid'])
            original=subs[off];off+=1
            if int(original[1])!=k or int(original[2])!=j:raise ValueError('Substep order mismatch')
            errors['time_s']=max(errors['time_s'],abs(float(data.time)-original[0]));errors['tool_z_substep_m']=max(errors['tool_z_substep_m'],abs(float(data.xpos[tool,2])-original[6]))
            mismatches['pillar_contact']+=int(bool(metric['obstacle_contact'])!=bool(original[3]));mismatches['finger_count']+=int(metric['finger_contacts']!=int(original[4]));mismatches['floor_latch']+=int(floor!=bool(original[5]))
        center=data.xpos[link]+data.xmat[link].reshape(3,3)@local
        for name,value,reference in [('tcp_end_m',data.site_xpos[tcp],old['tcp_pos'][k]),('eef_end_m',center,old['eef_pos'][k]),('tool_end_m',data.xpos[tool],old['screw_pos'][k])]:errors[name]=max(errors[name],float(np.max(np.abs(value-reference))))
        up=float(data.xmat[pillar].reshape(3,3)[2,2]);tilt|=up<.9
        errors['pillar_up']=max(errors['pillar_up'],abs(up-actions[k,8]));errors['max_height_m']=max(errors['max_height_m'],abs(maxheight-actions[k,5]))
        strict=bool(not floor and metric['valid'] and grasp and maxheight>=initial_height+.05)
        strict_frames=strict_frames+1 if strict and valid[k]==50 else 0
        for name,a,b in [('placement_valid',metric['valid'],actions[k,3]),('grasp_latch',grasp,actions[k,4]),('strict_valid',strict,actions[k,6]),('strict_frames',strict_frames,actions[k,7])]:mismatches[name]+=int(a!=b)
    success=strict_frames>=10;clean=bool(success and not hit and not floor and not tilt)
    actual=dict(task_success=success,clean_success=clean,pillar_contact=bool(hit),tool_floor_contact=bool(floor),pillar_tilt=bool(tilt))
    endpoints_match=all(v==bool(record[k]) for k,v in actual.items())
    legacy_error=abs(legacy_impulse-float(old['episode_contact_impulse']))
    # No thresholds tuned from outcomes: strict numeric tolerances and exact events.
    continuous_ok=(errors['q_start_rad']<=1e-7 and max(errors[k] for k in ['tcp_end_m','eef_end_m','tool_end_m','tool_z_substep_m','max_height_m'])<=1e-7 and errors['pillar_up']<=1e-7 and errors['time_s']<=1e-8)
    admitted=bool(continuous_ok and not any(mismatches.values()) and endpoints_match and legacy_error<=1e-5)
    output=Path(outdir)/row['scene']/row['assignment_id'];output.mkdir(parents=True,exist_ok=False)
    arrays={key:np.array([x[key] for x in load]) for key in load[0]}
    np.savez_compressed(output/'replayed_external_loads.npz',**arrays)
    np.savez_compressed(output/'reconstructed_executed_ctrl.npz',ctrl=commands,valid_substeps=valid,q_start=qstarts)
    result=dict(assignment=row['assignment_id'],scene=row['scene'],method='vlsa_native',seed=int(row['seed']),backend=row['backend'],admitted=admitted,source_kind='no-inference open-loop replay from archived safe twists and recorded action starts',new_inference_calls=0,qp_calls=0,physical_substeps=len(load),action_count=n,simulation_duration_s=float(data.time-start),wall_replay_s=time.perf_counter()-tick,nominal_twist_max_error=nominal_error,errors=errors,event_mismatches=mismatches,original_legacy_impulse_Ns=float(old['episode_contact_impulse']),replayed_legacy_impulse_Ns=legacy_impulse,legacy_impulse_abs_error_Ns=legacy_error,original_legacy_peak_N=float(old['episode_peak_contact_force']),replayed_legacy_peak_N=legacy_peak,peak_torque_Nm=taupeak,endpoints=actual,endpoints_match=endpoints_match,model_path=str(modelpath),model_sha256=sha(modelpath),data_path=str(datafile),data_sha256=sha(datafile),aligned_path=str(side),aligned_sha256=sha(side),initial_state_path=str(job/'initial_pre_action_state.npz'),initial_sha256=sha(job/'initial_pre_action_state.npz'),replay_source_sha256=sha(__file__))
    (output/'replay_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(dict(assignment=row['assignment_id'],admitted=admitted,errors=errors,mismatches=mismatches,legacy_error=legacy_error,output=str(output)),ensure_ascii=False),flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--ids',nargs='+',required=True);a=p.parse_args()
    rows=list(csv.DictReader((PACKAGE/'逐回合汇总.csv').open()));chosen=[r for r in rows if r['method']=='vlsa_native' and r['assignment_id'] in a.ids]
    if len(chosen)!=len(set(a.ids)):raise ValueError('Missing or duplicate assignment')
    for r in chosen:replay(r,a.output)

if __name__=='__main__':main()
