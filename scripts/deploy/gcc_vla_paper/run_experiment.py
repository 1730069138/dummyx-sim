#!/usr/bin/env python3
"""One entry point for the six-method, two-scene tabletop paper protocol."""
import argparse
import ast
import concurrent.futures
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CONFIG = HERE / 'config'
ASSETS = ROOT / 'models/gcc_vla_paper'
DEPLOY = ROOT / 'scripts/deploy'
LEGACY = DEPLOY / 'pace_neo_mesh_20261027/run_worker.py'
VLSA_SCHEDULER = DEPLOY / 'pace_native_aligned_20261027/scheduler_v3.py'
VLSA_ENTRY = DEPLOY / 'pace_native_aligned_20261027/deploy_vlsa_aligned_v6.py'
DUMMY_PYTHON = Path('/home/jun/anaconda3/envs/dummyx_vla/bin/python')
VLSA_PYTHON = Path('/home/jun/anaconda3/envs/vlsa_env/bin/python')
HOST = '100.64.142.55'
PORTS = {0: 8001, 1: 8002, 2: 8000, 3: 8003, 4: 8004, 5: 8005}
METHODS = {'pi0': 'raw_pi0', 'geometry': 'pace_g', 'gcc-vla': 'pace_gc',
           'oscbf': 'oscbf', 'neo': 'neo', 'vlsa': 'vlsa_native'}
SCENES = {'main': 'main', 'x40': 'pillar_x_plus40mm'}
PROMPT = 'Pick up the screwdriver and place it into the box.'
EXPECTED_SEED_PROTOCOL = 'fold_in(PRNGKey(episode_seed),request_index); reserved keys removed before model transform'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(encoded + '\n')


def verify_sources(include_assets=True):
    manifest = json.loads((CONFIG / 'source_manifest.json').read_text(encoding='utf-8'))
    failures = []
    for rel, expected in manifest['files'].items():
        if not include_assets and rel.startswith('models/gcc_vla_paper/'):
            continue
        path = ROOT / rel
        if not path.is_file():
            failures.append('缺失: ' + str(path))
        elif sha(path) != expected:
            failures.append('内容变化: ' + str(path))
    if failures:
        raise RuntimeError('论文运行依赖校验失败：\n' + '\n'.join(failures))
    return manifest


def parse_gpus(value):
    try:
        items = [int(x) for x in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('GPU 序号应为逗号分隔的 0–5') from exc
    if not items or len(items) != len(set(items)) or any(x not in PORTS for x in items):
        raise argparse.ArgumentTypeError('GPU 序号须为不重复的 0–5')
    return items


def set_flag(flags, key, value):
    result, index = [], 0
    while index < len(flags):
        if flags[index] == key:
            index += 2
        else:
            result.append(flags[index])
            index += 1
    return result + [key, str(value)]


def make_plan(args):
    scene = SCENES[args.scene]
    method = METHODS[args.method]
    start = args.seed_start if args.seed_start is not None else (610000 if args.scene == 'main' else 620000)
    if start < 0 or start + args.episodes - 1 >= 2 ** 31:
        raise ValueError('种子范围应在 0 至 2^31−1 内')
    out = Path(args.output).expanduser().resolve() if args.output else (
        ROOT / 'outputs/gcc_vla_paper_runs' /
        (dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + args.method + '_' + args.scene))
    gpus = args.gpus[:min(len(args.gpus), args.episodes)]
    jobs = []
    for rep in range(args.episodes):
        gpu = gpus[rep % len(gpus)]
        jobs.append({'id': f'{method}_{scene}_r{rep:03d}', 'group': method + '_' + scene,
                     'rep': rep, 'seed': start + rep, 'gpu': gpu, 'port': PORTS[gpu],
                     'scene_id': scene, 'method_id': method})
    plan = {'method': args.method, 'method_id': method, 'scene': args.scene, 'scene_id': scene,
            'episodes': args.episodes, 'seed_start': start, 'gpus': gpus, 'output': str(out),
            'jobs': jobs, 'headless': True, 'real_robot': False, 'cohort': 'new_run',
            'checkpoint': json.loads((CONFIG / 'remote_identity.json').read_text())['checkpoint'],
            'entry_sha256': sha(__file__), 'source_manifest_sha256': sha(CONFIG / 'source_manifest.json')}
    if args.method != 'vlsa':
        profiles = json.loads((CONFIG / 'profiles.json').read_text(encoding='utf-8'))
        template = copy.deepcopy(profiles[method + '/' + scene])
        profile = out / 'profiles' / (method + '_' + scene + '.json')
        flags = template['flags']
        for flag, value in [('--host', HOST), ('--port', 8000), ('--paper_experiment', profile),
                            ('--output_dir', out / 'unused_default_output'), ('--model_cache', out / 'model_cache')]:
            flags = set_flag(flags, flag, value)
        config = template['config']
        config['id'] = method + '_' + scene
        config['phase'] = 'new_run'
        group = {'id': config['id'], 'config': config, 'flags': flags}
        workers = {str(g): [j for j in jobs if j['gpu'] == g] for g in gpus}
        plan['protocol'] = {'run_id': out.name, 'groups': [group], 'workers': workers,
                            'factory': 'neo_factory:build_modules',
                            'entry_sha256': plan['entry_sha256'],
                            'source_manifest_sha256': plan['source_manifest_sha256'],
                            'service_metadata': str(out / 'identity/service_metadata.json'),
                            'pressure_scenes': {'pillar_x_plus40mm': {'folder': str(ASSETS / 'x40')}},
                            'first_batch_gate': bool(args.method == 'neo' and len(jobs) > len(gpus))}
        plan['commands'] = [[str(DUMMY_PYTHON), '-B', '-u', str(HERE / 'run_experiment.py'),
                             '_worker', '--run', str(out), '--gpu', str(g), '--port', str(PORTS[g])]
                            for g in gpus]
    else:
        plan['commands'] = [vlsa_command(j, out) for j in jobs]
    return plan


def vlsa_command(job, out):
    scene = job['scene_id']
    model = ASSETS / ('main' if scene == 'main' else 'x40') / 'model.mjb'
    command = [str(VLSA_PYTHON), '-B', '-u', str(VLSA_SCHEDULER), 'exec-native', '--entry', str(VLSA_ENTRY), '--',
               '--host', HOST, '--port', str(job['port']), '--num_episodes', '1', '--max_steps', '300',
               '--mode', 'vlsa', '--fixed_eval', '--initial_state_protocol', 'full_v1', '--settle_before_start',
               '--vlsa_action_space', 'cartesian', '--task_prompt', PROMPT, '--eef_ellipsoid_scale', '0.8',
               '--max_linear_correction', '0.08', '--max_angular_correction', '0.08', '--servo_response', '0.4',
               '--obstacle_source', 'perception', '--obstacle_text', 'red pillar.', '--device', 'cpu',
               '--groundingdino_config', '/home/jun/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
               '--groundingdino_checkpoint', '/home/jun/GroundingDINO/weights/groundingdino_swint_ogc.pth',
               '--record-root', str(out / 'records' / job['id']), '--episode-seed', str(job['seed']),
               '--aligned-model-mjb', str(model)]
    if scene != 'main':
        command += ['--aligned-scene-folder', str(ASSETS / 'x40'), '--aligned-trim-z']
    return command


def headless_environment():
    env = os.environ.copy()
    env.update(MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', NO_PROXY=HOST + ',localhost,127.0.0.1',
               no_proxy=HOST + ',localhost,127.0.0.1')
    env.pop('DISPLAY', None)
    return env


def check_services(gpus):
    """Read remote process/model identity and local websocket metadata; no inference."""
    expected = json.loads((CONFIG / 'remote_identity.json').read_text(encoding='utf-8'))
    probe_source = (DEPLOY / 'pace_gc_directional/experiments/probe_identity.py').read_text(encoding='utf-8')
    tree = ast.parse(probe_source)
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'remote' for t in n.targets)]
    if len(nodes) != 1:
        raise RuntimeError('无法读取远端身份核对程序')
    remote = ast.literal_eval(nodes[0].value)
    cmd = ['ssh', '-i', '/home/jun/1', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
           'jun@' + HOST, 'python3 -']
    result = subprocess.run(cmd, input=remote, text=True, capture_output=True, timeout=600, check=True)
    current = json.loads(result.stdout)
    if current['checkpoint_files'] != expected['checkpoint_files'] or current['source'] != expected['source']:
        raise RuntimeError('远端模型文件或策略源码与论文记录不一致；未启动实验')
    sys.path.insert(0, str(DEPLOY))
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = HOST + ',localhost,127.0.0.1'
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    services = []
    for gpu in gpus:
        port = PORTS[gpu]
        matched = [p for p in current['services'] if f'--port={port} ' in p['cmdline']
                   and p['environment'].get('CUDA_VISIBLE_DEVICES') == str(gpu)
                   and '--policy.config=pi0_dummyx_lora' in p['cmdline']
                   and '--policy.dir=' + expected['checkpoint'] in p['cmdline']]
        if len(matched) != 1:
            raise RuntimeError(f'GPU {gpu} / 端口 {port} 的策略服务身份不匹配；未启动实验')
        client = WebsocketClientPolicy(HOST, port)
        try:
            metadata = client.get_server_metadata()
        finally:
            client._ws.close()
        if metadata.get('eval_seed_protocol') != EXPECTED_SEED_PROTOCOL:
            raise RuntimeError(f'端口 {port} 未提供论文所需的种子协议；未启动实验')
        services.append({'gpu': gpu, 'port': port, 'metadata': metadata, 'service': matched[0]})
    return current, services


class RootProxy:
    """Redirect only the legacy worker's model cache into this new run."""
    def __init__(self, run):
        self.run = run

    def __truediv__(self, suffix):
        if str(suffix) == 'outputs/pace_external50/benchmark_20261027_062049/experiments/model_cache':
            return self.run / 'model_cache'
        return ROOT / suffix


def worker_main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--port', required=True, type=int)
    args = parser.parse_args(argv)
    run = args.run.resolve(strict=True)
    if PORTS.get(args.gpu) != args.port:
        raise ValueError('GPU 与服务端口不匹配')
    spec = importlib.util.spec_from_file_location('paper_legacy_worker', LEGACY)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    common = sys.modules['common']

    def inside_run(path):
        path = Path(path).resolve()
        if not path.is_relative_to(run):
            raise ValueError('工作进程只能写入本次输出目录: ' + str(path))
        return path

    def verify_protocol(path):
        path = inside_run(path)
        if sha(path / 'protocol.json') != (path / 'protocol.sha256').read_text().strip():
            raise RuntimeError('实验方案内容发生变化')
        protocol = json.loads((path / 'protocol.json').read_text(encoding='utf-8'))
        if sha(__file__) != protocol['entry_sha256'] or sha(CONFIG / 'source_manifest.json') != protocol['source_manifest_sha256']:
            raise RuntimeError('统一入口或源码清单在运行期间发生变化')
        verify_sources(include_assets=False)
        return protocol

    common.new_output = inside_run
    worker.new_output = inside_run
    worker.verify_freeze = verify_protocol
    worker.ROOT = RootProxy(run)
    sys.argv = [str(LEGACY), '--run', str(run), '--gpu', str(args.gpu), '--port', str(args.port),
                '--factory', 'neo_factory:build_modules']
    worker.main()


def shared_run(plan):
    out = Path(plan['output'])
    processes = {}
    for command, gpu in zip(plan['commands'], plan['gpus']):
        with (out / f'worker_gpu{gpu}.log').open('xb') as stream:
            processes[gpu] = subprocess.Popen(command, cwd=ROOT, env=headless_environment(),
                                               stdout=stream, stderr=subprocess.STDOUT)
    gate_pending = plan['protocol']['first_batch_gate']
    gate = '未启用'
    try:
        while any(p.poll() is None for p in processes.values()):
            failed = []
            for gpu, process in processes.items():
                if process.poll() is None:
                    continue
                completion = out / f'worker_gpu{gpu}' / 'completed.json'
                if process.returncode != 0 or not completion.is_file() or not json.loads(completion.read_text()).get('complete'):
                    failed.append(gpu)
            if failed and not (out / 'STOP').exists():
                (out / 'STOP').touch(exist_ok=False)
                print('工作进程异常退出，其他进程将在当前回合结束后停止:', failed, flush=True)
            if gate_pending:
                if (out / 'STOP').exists():
                    gate = '工作进程提前退出，停止'
                    gate_pending = False
                    continue
                ready = {g: out / f'worker_gpu{g}' / 'first_batch_ready.json' for g in processes}
                if all(p.is_file() for p in ready.values()):
                    first = [json.loads(Path(json.loads(p.read_text())['result_path']).read_text()) for p in ready.values()]
                    if any(x.get('error') is not None or x.get('metrics', {}).get('success') is None for x in first):
                        (out / 'STOP').touch(exist_ok=True); gate = '首批技术错误，停止'; gate_pending = False
                    elif not any(x['metrics']['success'] for x in first):
                        (out / 'STOP').touch(exist_ok=True); gate = '首批全部任务失败，停止'; gate_pending = False
                    else:
                        (out / 'RELEASE_AFTER_FIRST_BATCH').touch(exist_ok=False); gate = '首批有成功，继续'; gate_pending = False
                    print(gate, flush=True)
                elif any(p.poll() is not None and not ready[g].exists() for g, p in processes.items()):
                    (out / 'STOP').touch(exist_ok=True); gate = '首批工作进程提前退出，停止'; gate_pending = False
            time.sleep(.5)
    except KeyboardInterrupt:
        (out / 'STOP').touch(exist_ok=True)
        for proc in processes.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in processes.values():
            proc.wait()
        raise
    statuses = {str(g): p.returncode for g, p in processes.items()}
    completions = {}
    for gpu in processes:
        path = out / f'worker_gpu{gpu}' / 'completed.json'
        completions[str(gpu)] = json.loads(path.read_text()) if path.is_file() else None
    complete = all(c is not None and c.get('complete') for c in completions.values())
    write_new(out / 'run_summary.json', {'complete': complete, 'gate': gate,
                                         'worker_exit_codes': statuses, 'worker_completion': completions})
    return complete


def vlsa_run(plan):
    out = Path(plan['output'])
    (out / 'records').mkdir()
    (out / 'logs').mkdir()
    stopped = False
    def run_gpu(gpu):
        nonlocal stopped
        results = []
        for job, command in zip(plan['jobs'], plan['commands']):
            if job['gpu'] != gpu or stopped:
                continue
            with (out / 'logs' / (job['id'] + '.log')).open('xb') as stream:
                completed = subprocess.run(command, cwd=ROOT, env=headless_environment(),
                                           stdout=stream, stderr=subprocess.STDOUT)
            record = out / 'records' / job['id']
            sidecars = list(record.glob('*/aligned_ep_001.npz')) if record.exists() else []
            valid = completed.returncode == 0 and (record / 'summary.txt').is_file() and len(sidecars) == 1
            entry = {'job': job, 'exit_code': completed.returncode, 'completed_episode': valid}
            write_new(out / 'logs' / (job['id'] + '.status.json'), entry)
            results.append(entry)
            if not valid:
                stopped = True
                break
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(plan['gpus'])) as pool:
        groups = list(pool.map(run_gpu, plan['gpus']))
    results = [x for group in groups for x in group]
    complete = len(results) == len(plan['jobs']) and all(x['completed_episode'] for x in results)
    write_new(out / 'run_summary.json', {'complete': complete, 'completed': sum(x['completed_episode'] for x in results),
                                         'planned': len(plan['jobs']), 'results': results})
    return complete


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '_worker':
        worker_main(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description='论文桌面仿真：选择方法和场景；默认只显示计划。')
    parser.add_argument('--method', required=True, choices=tuple(METHODS), help='实验方法')
    parser.add_argument('--scene', required=True, choices=tuple(SCENES), help='main 原场景，x40 柱体 X+40 mm')
    parser.add_argument('--episodes', type=int, default=1, help='该方法该场景的回合数，1–50')
    parser.add_argument('--gpus', type=parse_gpus, default=parse_gpus('0,1,2,3,4,5'), help='远端 GPU 序号，逗号分隔')
    parser.add_argument('--seed-start', type=int, help='首个回合种子；默认原场景 610000，X+40 场景 620000')
    parser.add_argument('--output', help='新的输出目录；不覆盖已有目录')
    parser.add_argument('--execute', action='store_true', help='检查服务身份后执行；省略时只展示计划')
    args = parser.parse_args()
    if not 1 <= args.episodes <= 50:
        parser.error('每组回合数必须为 1–50')
    verify_sources()
    plan = make_plan(args)
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if Path(sys.executable).resolve() != DUMMY_PYTHON.resolve():
        raise RuntimeError('执行请使用 /home/jun/anaconda3/envs/dummyx_vla/bin/python')
    if not VLSA_PYTHON.is_file() or not DUMMY_PYTHON.is_file():
        raise RuntimeError('缺少论文实验环境')
    out = Path(plan['output'])
    if out.exists():
        raise FileExistsError('输出目录已存在，不能覆盖: ' + str(out))
    print('正在核对远端策略服务和模型身份，不发送推理请求。', flush=True)
    remote, services = check_services(plan['gpus'])
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / 'plan.json', plan)
    write_new(out / 'identity/remote_identity.json', remote)
    write_new(out / 'identity/service_metadata.json', services)
    if args.method != 'vlsa':
        protocol = plan['protocol']
        for group in protocol['groups']:
            write_new(out / 'profiles' / (group['id'] + '.json'), group['config'])
        write_new(out / 'protocol.json', protocol)
        (out / 'protocol.sha256').write_text(sha(out / 'protocol.json') + '\n')
        complete = shared_run(plan)
    else:
        complete = vlsa_run(plan)
    print('输出目录:', out, '\n全部完成:', complete, flush=True)
    if not complete:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
