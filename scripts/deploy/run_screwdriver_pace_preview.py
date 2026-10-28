"""Old tabletop PACE evaluation profile; reuses an already running policy service.

All CLI overrides are forwarded after the profile, e.g. --pace_preview off.
No policy-server startup, backups, or automatic video recording occur here.
"""
from pathlib import Path
import sys,os
import deploy_screwdriver_client_old as deploy

ROOT=Path(__file__).resolve().parents[2]
PROFILE=[
    '--host','100.64.142.55','--port','8000','--device','cpu',
    '--case','4','--num_episodes','1','--fixed_eval','--max_steps','300','--chunk_steps','50',
    '--pace_preview','on','--pace_prediction_mode','execution_lift',
    '--pace_tracking_model',str(ROOT/'scripts/deploy/tabletop_tracking_model.json'),
    '--pace_tool_proxy','components','--pace_route_preference','staged_up_near',
    '--pace_route_commit','on','--pace_route_descent_check','off',
    '--pace_joint_filter','off','--pace_preserve_orientation','on',
    '--pace_current_guard','on','--pace_safety_only','on',
    '--pace_lift_preference','0','--pace_max_correction_m_s','0.15',
    '--box_attraction','on','--box_attraction_speed_m_s','1.2',
    '--box_attraction_decay_power','1','--box_attraction_target','world_xy_zero',
    '--grip_noslip_iterations','3',
    '--task_prompt','Pick up the screwdriver and place it into the box.',
    '--output_dir',str(ROOT/'inference_results/screwdriver_pace_preview'),
]

def main():
    args=deploy.build_argument_parser().parse_args(PROFILE+sys.argv[1:])
    for key in ('NO_PROXY','no_proxy'):
        hosts=[v for v in os.environ.get(key,'').split(',') if v]
        if args.host not in hosts:hosts.append(args.host)
        os.environ[key]=','.join(hosts)
    deploy.main(args)

if __name__=='__main__':main()
