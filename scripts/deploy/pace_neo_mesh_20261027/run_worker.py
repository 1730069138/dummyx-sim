"""One existing service per worker; frozen methods share scene-seed blocks."""
from pathlib import Path
import argparse
import contextlib
import copy
import gc
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(Path('/home/jun/dummyx-sim/scripts/deploy/pace_external50/controllers')), '/home/jun/dummyx-sim/scripts/deploy']
from common import ROOT, WORK, PORTS, METHODS, now, sha, verify_freeze, new_output, write_new_json


def load_function(specification):
    module_name, _, name = specification.partition(':')
    module = importlib.import_module(module_name)
    return module, getattr(module, name or 'build_modules')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--factory', default='independent_dispatch:build_modules')
    args = parser.parse_args()
    assert PORTS[args.gpu] == args.port
    out = new_output(args.run)
    protocol = verify_freeze(out)
    assert args.factory == protocol['factory']
    schedule = protocol['workers'][str(args.gpu)]
    assert schedule
    assert all(job['gpu'] == args.gpu and job['port'] == args.port for job in schedule)
    worker = out / f'worker_gpu{args.gpu}'
    worker.mkdir(exist_ok=False)
    for key in ('NO_PROXY', 'no_proxy'):
        os.environ[key] = '100.64.142.55,localhost,127.0.0.1'
    import numpy as np
    import mujoco
    import deploy_screwdriver_client_old as deploy
    import tabletop_paper_ablation as ablation
    # Bind the original before any patch. Runtime imports of the patched name recurse.
    baseline_build = ablation.build_modules
    module, external_factory = load_function(args.factory)
    scene_context = getattr(module, 'scene_context', lambda config: contextlib.nullcontext())
    metrics_path = ROOT / 'scripts/deploy/gcape_paper/frozen_original/outputs/screwdriver_eval/paper_full50_20261026/metrics.py'
    metric_spec = importlib.util.spec_from_file_location('pace_external50_frozen_metrics', metrics_path)
    metric_module = importlib.util.module_from_spec(metric_spec)
    metric_spec.loader.exec_module(metric_module)
    sys.path.append(str(ROOT / 'scripts/deploy/gcape_paper/experiments'))
    from pressure_runtime import pressure_scene
    base = deploy.websocket_client_policy.WebsocketClientPolicy
    deploy_parser = deploy.build_argument_parser()
    groups = {group['id']: group for group in protocol['groups']}
    initial = deploy_parser.parse_args(protocol['groups'][0]['flags'])

    class MemoDetector:
        def __init__(self):
            self.inner = deploy.GroundingDINOWrapper(initial.groundingdino_config, initial.groundingdino_checkpoint, 'cpu')
            self.cache = {}
            self.hits = self.misses = 0

        def detect_all_boxes(self, rgb, caption, box_threshold, text_threshold):
            key = hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest() + '|' + repr((caption, box_threshold, text_threshold))
            if key not in self.cache:
                self.cache[key] = self.inner.detect_all_boxes(rgb, caption, box_threshold, text_threshold)
                self.misses += 1
                (worker / 'dino_cache.json').write_text(json.dumps(dict(
                    provenance='Actual CPU GroundingDINO output keyed by exact RGB/request; depth geometry refit each episode',
                    entries=self.cache), indent=2, default=lambda item: np.asarray(item).tolist()))
            else:
                self.hits += 1
            return copy.deepcopy(self.cache[key])

    class Viewer:
        def __init__(self, *unused, **kwargs):
            self.user_scn = SimpleNamespace(ngeom=0, geoms=[mujoco.MjvGeom() for _ in range(40)])

        def is_running(self):
            return True

        def sync(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *unused):
            pass

    detector = MemoDetector()
    probe = base('100.64.142.55', args.port)
    try:
        service = probe.get_server_metadata()
    finally:
        probe._ws.close()
    service_entries = json.loads(Path(protocol['service_metadata']).read_text())
    if isinstance(service_entries, dict): service_entries = service_entries['services']
    expected = next(item for item in service_entries if item['port'] == args.port)
    assert service == expected['metadata'] and service.get('eval_seed_protocol')
    write_new_json(worker / 'ready.json', dict(gpu=args.gpu, port=args.port, pid=os.getpid(),
                                               service_metadata=service, policy_calls=0, created_utc=now()))
    state = dict(out=None, seed=None, contacts=[], last_time=None, clients=[], method=None, factory_calls=[])
    original_placement = deploy.TabletopArm.placement

    def observed_placement(arm, *positional, **kwargs):
        metric = original_placement(arm, *positional, **kwargs)
        if state['out'] is not None and state['last_time'] != float(arm.data.time):
            state['last_time'] = float(arm.data.time)
            model, data = arm.model, arm.data
            pairs = []
            for contact in data.contact:
                if contact.dist > 0:
                    continue
                names = [model.body(int(model.geom_bodyid[int(geom)])).name for geom in contact.geom]
                if 'pillar' in ' '.join(names) or bool(metric.get('obstacle_contact', False)):
                    pairs.append(dict(geom_ids=[int(geom) for geom in contact.geom],
                                      geoms=[model.geom(int(geom)).name for geom in contact.geom],
                                      bodies=names, dist_m=float(contact.dist), pos_m=contact.pos.tolist()))
            state['contacts'].append(dict(time_s=float(data.time), pillar_contact=bool(metric.get('obstacle_contact', False)),
                                           finger_contacts=int(metric.get('finger_contacts', 0)), pairs=pairs))
        return metric

    class Observed(base):
        def __init__(self, *positional, **kwargs):
            super().__init__(*positional, **kwargs)
            self.index = 0
            state['clients'].append(self)
            assert self.get_server_metadata() == service, 'Service metadata changed between episodes'

        def infer(self, observation):
            started = time.perf_counter()
            request = dict(observation)
            request.update(_eval_seed=state['seed'], _eval_request_index=self.index)
            result = super().infer(request)
            assert result['eval_rng'] == dict(episode_seed=state['seed'], request_index=self.index)
            hashes = {key: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
                      for key, value in observation.items() if key.startswith('observation/')}
            np.savez_compressed(state['out'] / f'policy_chunk_{self.index:03d}.npz',
                                actions=result['actions'], state=observation['observation/state'])
            receipt = dict(request_index=self.index, episode_seed=state['seed'], observation_hash=hashes,
                           eval_rng=result['eval_rng'], server_timing=result.get('server_timing'),
                           client_elapsed_ms=1000 * (time.perf_counter() - started), captured_utc=now(),
                           gpu=args.gpu, port=args.port, prompt=observation.get('prompt'))
            with (state['out'] / 'policy_lineage.jsonl').open('a') as stream:
                stream.write(json.dumps(receipt, default=str) + '\n')
            self.index += 1
            return result

    def dispatch(arm, perception, initial_height, runtime_args, config):
        method = config['method_id']
        assert method == state['method'] and method in METHODS
        assert config['preview'] is False and config['attraction'] is False
        assert runtime_args.box_attraction == 'off' and runtime_args.pace_preview == 'off'
        if method in ('raw_pi0', 'pace_g', 'pace_gc'):
            preview, attraction = baseline_build(arm, perception, initial_height, runtime_args, config)
        else:
            assert runtime_args.case == 3 and config['contact'] is False and config['geometry'] is False
            preview, attraction = external_factory(arm, perception, initial_height, runtime_args, config)
            assert preview is not None and type(preview).__module__ != 'tabletop_pace_preview', 'External method returned original PACE'
        assert attraction is None
        if method == 'raw_pi0':
            assert preview is None
        state['factory_calls'].append(dict(method=method, case=runtime_args.case, C=config['contact'],
                                            controller=None if preview is None else type(preview).__module__ + '.' + type(preview).__name__,
                                            current_guard=False if preview is None else bool(preview.current_guard),
                                            attraction=False, original_baseline_binding=method in ('raw_pi0', 'pace_g', 'pace_gc')))
        return preview, attraction

    started = time.monotonic()
    completed = 0
    for job in schedule:
        if (out / 'STOP').exists():
            break
        verify_freeze(out)
        current_out = worker / job['id']
        current_out.mkdir()
        group = groups[job['group']]
        config = group['config']
        state.update(out=current_out, seed=job['seed'], contacts=[], last_time=None, method=job['method_id'], factory_calls=[])
        scene_id = config['scene_id']
        scene_folder = None if scene_id == 'main' else protocol['pressure_scenes'][scene_id]['folder']
        flags = group['flags'] + ['--port', str(args.port), '--output_dir', str(current_out / 'episodes'),
                                 '--model_cache', str(ROOT / 'outputs/pace_external50/benchmark_20261027_062049/experiments/model_cache')]
        runtime_args = deploy_parser.parse_args(flags)
        assert runtime_args.case == (8 if job['method_id'] == 'pace_gc' else 4 if job['method_id'] == 'pace_g' else 3)
        assert runtime_args.num_episodes == 1 and runtime_args.chunk_steps == 50
        assert runtime_args.max_steps == 300 and runtime_args.box_attraction == 'off'
        write_new_json(current_out / 'job.json', dict(**job, args=vars(runtime_args), config=config, run_id=protocol['run_id']))
        (worker / 'progress.json').write_text(json.dumps(dict(completed=completed, planned=len(schedule), running=job,
                                                              wall_seconds=time.monotonic() - started)))
        began = time.monotonic()
        error = None
        pressure_receipt = None
        try:
            with contextlib.ExitStack() as stack:
                if scene_folder is not None:
                    pressure_receipt = stack.enter_context(pressure_scene(deploy, scene_folder))
                stack.enter_context(scene_context(config))
                stack.enter_context(patch.object(ablation, 'build_modules', new=dispatch))
                stack.enter_context(patch.object(deploy, 'GroundingDINOWrapper', return_value=detector))
                stack.enter_context(patch.object(deploy.mujoco.viewer, 'launch_passive', side_effect=Viewer))
                stack.enter_context(patch.object(deploy.websocket_client_policy, 'WebsocketClientPolicy', new=Observed))
                stack.enter_context(patch.object(deploy.TabletopArm, 'placement', new=observed_placement))
                log = stack.enter_context((current_out / 'client.log').open('x'))
                stack.enter_context(contextlib.redirect_stdout(log))
                deploy.main(runtime_args)
        except Exception as failure:
            error = repr(failure)
            write_new_json(current_out / 'error.json', dict(error=error, traceback=traceback.format_exc()))
        with (current_out / 'contact_object_samples.jsonl').open('x') as stream:
            for sample in state['contacts']:
                stream.write(json.dumps(sample) + '\n')
        write_new_json(current_out / 'factory_receipt.json', state['factory_calls'])
        if pressure_receipt is not None:
            write_new_json(current_out / 'pressure_initialization_receipt.json', pressure_receipt)
        for client in state['clients']:
            try:
                client._ws.close()
            except Exception:
                pass
        state['clients'].clear()
        gc.collect()
        result = None
        runs = list((current_out / 'episodes').glob('run_*'))
        if len(runs) == 1 and (runs[0] / 'episodes.json').exists():
            values = json.loads((runs[0] / 'episodes.json').read_text())
            result = values[0] if values else None
        try:
            metrics = metric_module.extract(current_out, result)
        except Exception as failure:
            metrics = dict(metric_error=repr(failure))
            (current_out / 'metric_error.txt').write_text(traceback.format_exc())
        entry = dict(**job, result=result, error=error, metrics=metrics, wall_seconds=time.monotonic() - began,
                     path=str(current_out), finished_utc=now(), run_id=protocol['run_id'])
        write_new_json(current_out / 'result.json', entry)
        with (worker / 'ledger.jsonl').open('a') as stream:
            stream.write(json.dumps(entry) + '\n')
        if error is not None or metrics.get('metric_error'):
            print('TECHNICAL ERROR: worker stops, trial retained as unknown', flush=True)
            break
        completed += 1
        (worker / 'progress.json').write_text(json.dumps(dict(completed=completed, planned=len(schedule), last=job,
                                                              wall_seconds=time.monotonic() - started,
                                                              dino_hits=detector.hits, dino_misses=detector.misses)))
        print(json.dumps(dict(job=job['id'], completed=completed, planned=len(schedule), result=result,
                              error=error, wall_seconds=entry['wall_seconds'])), flush=True)
        if completed == 1 and protocol.get('first_batch_gate'):
            write_new_json(worker / 'first_batch_ready.json', dict(job=job['id'], result_path=str(current_out/'result.json'), finished_utc=now()))
            print('FIRST BATCH COMPLETE: waiting for explicit release or STOP', flush=True)
            while not (out/'STOP').exists() and not (out/'RELEASE_AFTER_FIRST_BATCH').exists():
                time.sleep(1.)
            if (out/'STOP').exists():
                break
            # A release is valid only after all six first results are durable.
            assert all((out/f'worker_gpu{g}'/'first_batch_ready.json').is_file() for g in protocol['workers']), 'Premature batch release'

    write_new_json(worker / 'completed.json', dict(trials=completed, planned=len(schedule),
                                                  complete=completed == len(schedule), service_left_running=True))
    print('Worker complete', flush=True)


if __name__ == '__main__':
    main()
