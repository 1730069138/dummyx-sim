#!/usr/bin/env python3
"""Audit saved Exp2 fixed/normal pairs, without physics or policy calls.

Recomputes J4 from every trace; checks job/path/direction identities and original
paper_truth arrays; compares each of the first 4000 scalar external-load samples;
records first actual reference/state divergence and last nonzero load times.
Existing JSON or Markdown output is refused.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def first_time(mask, time):
    indices = np.flatnonzero(mask)
    return float(time[indices[0]]) if len(indices) else None


def last_time(mask, time):
    indices = np.flatnonzero(mask)
    return float(time[indices[-1]]) if len(indices) else None


def load_episode(row):
    path = Path(row['trace_path']).resolve(strict=True)
    result_path = Path(row['result_path']).resolve(strict=True)
    result = json.loads(result_path.read_text())
    assert result['id'] == row['id']
    assert result['seed'] == row['seed']
    assert result['config_name'] == row['config']
    assert result['scene_id'] == row['scene']
    assert Path(result['trace_path']).resolve() == path
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata_json']))
        job = meta['job']
        assert job['id'] == row['id'] and job['seed'] == row['seed']
        assert job['arm'] == row['config'] and job['scene_id'] == row['scene']
        expected = {'fixed': 'fixed_up_base', 'normal': 'closest_normal'}[row['config']]
        assert meta['geometry_direction'] == expected
        assert meta['geometry_enabled'] and meta['contact_enabled'] and meta['current_guard']
        done = z['step_completed'].astype(bool)
        assert np.all(done), 'Incomplete logged physical interval'
        assert len(done) >= 4000
        start = float(z['time_before_s'][0])
        before = z['time_before_s'][done] - start
        after = z['time_after_s'][done] - start
        assert np.allclose(after - before, .001, atol=1e-8, rtol=0)
        assert np.allclose(np.diff(after), .001, atol=1e-8, rtol=0)
        arrays = {name: z[name][done] for name in [
            'sum_external_contact_force_N', 'target_nominal_a', 'target_geometry_uG',
            'delta_geometry', 'target_contact_ubar', 'u_applied', 'q_before', 'delta_guard']}
        assert all(np.isfinite(a).all() for a in arrays.values())
        force = arrays['sum_external_contact_force_N']
        j4 = float(force[:4000].sum() * .001)
        assert abs(j4 - row['J4']) < 1e-12, 'Analyzer J4 does not match this trace'
        g = np.max(abs(arrays['delta_geometry']), axis=1)
        guard = np.max(abs(arrays['delta_guard']), axis=1)
        data_path = Path(row['data_path']).resolve(strict=True)
        assert str(data_path).startswith(str(result_path.parent) + '/'), 'Data path outside own job'
        truth_path = data_path.with_name('paper_truth_ep_001.npz')
        with np.load(truth_path, allow_pickle=False) as truth:
            recorded_truth = truth['all_sum_force_N']
            truth_time = truth['time_s']
            assert len(recorded_truth) == len(force)
            max_truth_difference = float(np.max(abs(recorded_truth - force)))
            assert max_truth_difference < 1e-9
        preview_path = data_path.with_name('preview_ep_001.npz')
        with np.load(preview_path, allow_pickle=False) as preview:
            preview_directions = sorted(set(preview['geometry_direction'].tolist()))
            assert preview_directions == [expected]
            chosen = preview['chosen_direction']
            assert np.isfinite(chosen).all()
            chosen_active = np.linalg.norm(chosen, axis=1) > 1e-12
            chosen_nonzero_count = int(chosen_active.sum())
            assert chosen_nonzero_count > 0, 'Direction never actually selected'
            chosen_first = chosen[np.flatnonzero(chosen_active)[0]].tolist()
        summary = dict(
            id=row['id'], trace_path=str(path), result_path=str(result_path),
            data_path=str(data_path), truth_path=str(truth_path), preview_path=str(preview_path),
            seed=row['seed'], gpu=row['gpu'], geometry_direction=meta['geometry_direction'],
            physical_samples=len(force), J4_recomputed=j4, J4_analyzer=row['J4'],
            J4_exact_matches_analyzer=j4 == row['J4'],
            first4000_force_sha256=array_sha(force[:4000]),
            full_geometry_target_sha256=array_sha(arrays['target_geometry_uG']),
            truth_force_max_abs_difference_N=max_truth_difference,
            first_geometry_increment_before_s=first_time(g > 1e-12, before),
            geometry_increment_nonzero_steps=int(np.count_nonzero(g > 1e-12)),
            first_guard_increment_before_s=first_time(guard > 1e-12, before),
            last_positive_force_first4_after_s=last_time(force[:4000] > 0, after[:4000]),
            last_force_gt_1e9below_first4_after_s=last_time(force[:4000] > 1e-9, after[:4000]),
            last_positive_force_entire_episode_after_s=last_time(force > 0, after),
            preview_directions=preview_directions,
            preview_chosen_nonzero_count=chosen_nonzero_count,
            preview_first_nonzero_chosen_direction=chosen_first,
            truth_time_first_s=float(truth_time[0]), truth_time_last_s=float(truth_time[-1]))
        assert summary['first_geometry_increment_before_s'] is not None
        return summary, arrays, before, after


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir', type=Path, default=root / 'audit/analysis_exp2_complete')
    parser.add_argument('--output', type=Path, default=root / 'audit/exp2_J4_identity_diagnostic.json')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.suffix.lower() != '.json':
        raise ValueError('--output must be a new .json file')
    markdown = output.with_suffix('.md')
    if output.exists() or markdown.exists():
        raise FileExistsError('Refusing existing JSON/Markdown output')
    analysis_path = args.analysis_dir.resolve() / 'analysis_round2.json'
    data = json.loads(analysis_path.read_text())
    rows = [r for r in data['per_episode'] if r['experiment'] == 'exp2']
    assert len(rows) == 200
    allocated = {}
    for row in rows:
        key = (row['scene'], row['seed'])
        cell = allocated.setdefault(key, {})
        assert row['config'] not in cell
        cell[row['config']] = row
    assert len(allocated) == 100
    assert all(set(cell) == {'fixed', 'normal'} for cell in allocated.values())
    pairs = []
    for number, ((scene, seed), cell) in enumerate(sorted(allocated.items()), 1):
        fixed, fa, ft, fta = load_episode(cell['fixed'])
        normal, na, nt, nta = load_episode(cell['normal'])
        assert fixed['trace_path'] != normal['trace_path']
        assert fixed['data_path'] != normal['data_path']
        length = min(len(ft), len(nt))
        assert np.allclose(ft[:length], nt[:length], atol=1e-12, rtol=0)
        force_difference = abs(fa['sum_external_contact_force_N'][:4000] - na['sum_external_contact_force_N'][:4000])
        divergence = {}
        for field in ['target_nominal_a', 'delta_geometry', 'target_geometry_uG',
                      'target_contact_ubar', 'u_applied', 'q_before']:
            difference = np.max(abs(fa[field][:length] - na[field][:length]), axis=1)
            divergence[field] = dict(
                first_exact_difference_before_s=first_time(difference > 0, ft[:length]),
                first_gt_1e12below_difference_before_s=first_time(difference > 1e-12, ft[:length]),
                max_abs_difference_rad=float(difference.max()))
        target_time = divergence['u_applied']['first_gt_1e12below_difference_before_s']
        last_load = max(fixed['last_positive_force_first4_after_s'], normal['last_positive_force_first4_after_s'])
        pairs.append(dict(
            scene=scene, seed=seed, fixed=fixed, normal=normal,
            first4_force_exact_equal=bool(np.array_equal(fa['sum_external_contact_force_N'][:4000], na['sum_external_contact_force_N'][:4000])),
            first4_force_max_abs_difference_N=float(force_difference.max()),
            J4_difference_Ns=normal['J4_recomputed'] - fixed['J4_recomputed'],
            reference_and_state_divergence=divergence,
            last_first4_load_precedes_first_applied_target_divergence=target_time is not None and last_load < target_time,
            first_applied_target_divergence_minus_last_first4_load_s=target_time - last_load if target_time is not None else None,
            normal_has_actual_geometry_increment=normal['geometry_increment_nonzero_steps'] > 0,
            geometry_targets_ever_differ=divergence['target_geometry_uG']['max_abs_difference_rad'] > 1e-12))
        if number % 25 == 0:
            print('Read pairs', number, '/100', flush=True)
    scenes = []
    for scene in ['main', 'pillar_x_plus40mm']:
        selected = [p for p in pairs if p['scene'] == scene]
        def range_of(values):
            values = [v for v in values if v is not None]
            return [min(values), max(values)] if values else None
        scenes.append(dict(
            scene=scene, n_pairs=len(selected),
            J4_fixed_mean=float(np.mean([p['fixed']['J4_recomputed'] for p in selected])),
            J4_normal_mean=float(np.mean([p['normal']['J4_recomputed'] for p in selected])),
            exact_equal_first4_force_pairs=sum(p['first4_force_exact_equal'] for p in selected),
            actual_geometry_target_difference_pairs=sum(p['geometry_targets_ever_differ'] for p in selected),
            load_before_first_applied_target_divergence_pairs=sum(p['last_first4_load_precedes_first_applied_target_divergence'] for p in selected),
            last_first4_load_after_s_range=range_of([p[c]['last_positive_force_first4_after_s'] for p in selected for c in ['fixed', 'normal']]),
            first_geometry_increment_before_s_range=range_of([p[c]['first_geometry_increment_before_s'] for p in selected for c in ['fixed', 'normal']]),
            first_applied_target_divergence_before_s_range=range_of([p['reference_and_state_divergence']['u_applied']['first_gt_1e12below_difference_before_s'] for p in selected]),
            target_divergence_minus_last_first4_load_s_range=range_of([p['first_applied_target_divergence_minus_last_first4_load_s'] for p in selected])))
    checks = dict(
        all_100_pairs_force_first4000_exact_equal=all(p['first4_force_exact_equal'] for p in pairs),
        all_100_pairs_geometry_targets_actually_diverge=all(p['geometry_targets_ever_differ'] for p in pairs),
        all_100_normal_configs_actually_intervened=all(p['normal_has_actual_geometry_increment'] for p in pairs),
        all_100_pairs_last_first4_load_before_applied_target_divergence=all(p['last_first4_load_precedes_first_applied_target_divergence'] for p in pairs),
        all_200_J4_recomputations_exact_match_analyzer=all(p[c]['J4_exact_matches_analyzer'] for p in pairs for c in ['fixed', 'normal']))
    report = dict(passed=all(checks.values()), checks=checks, scenes=scenes, pairs=pairs,
                  source_analysis=str(analysis_path), source_analysis_sha256=sha(analysis_path),
                  script=str(Path(__file__).resolve()), script_sha256=sha(__file__),
                  code_sources={str(root / path): sha(root / path) for path in [
                      'runtime/tabletop_pace_preview.py', 'scripts/run_round2_worker.py']},
                  physics_steps=0, policy_calls=0,
                  notes=['Times are relative to first logged physical interval.',
                         'Force time uses interval end; reference divergence uses interval start.',
                         'Fsum is sum of protected external contact force norms, not net force.',
                         'J4 identity does not imply equal later closed-loop trajectories or outcomes.'])
    lines = ['# Exp2前4秒载荷积分相同的只读时序核查', '',
             f'来源：`{analysis_path}`。实际读取200份独立trace、200份原paper_truth、200份preview；按场景与随机种子组成100对fixed/normal。没有物理推进、控制器重算或策略推理。', '',
             f'总体验证：**{"通过" if report["passed"] else "存在待解释项"}**；逐项检查：`{json.dumps(checks, ensure_ascii=False)}`。', '',
             '## 直接结果', '',
             '|场景|同种子对数|fixed/normal J4均值(N·s)|逐样本完全相同对数|真实几何目标分叉对数|末次前4s载荷早于首次实际目标分叉|',
             '|---|---|---|---|---|---|']
    for s in scenes:
        lines.append(f"|{s['scene']}|{s['n_pairs']}|{s['J4_fixed_mean']:.15g} / {s['J4_normal_mean']:.15g}|{s['exact_equal_first4_force_pairs']}|{s['actual_geometry_target_difference_pairs']}|{s['load_before_first_applied_target_divergence_pairs']}|")
    lines += ['', '|场景|前4s末次非零载荷时刻(s，步末)|首次非零几何增量(s，步初)|首次实际下发目标分叉(s，步初)|分叉减末次载荷的时间差(s)|', '|---|---|---|---|---|']
    for s in scenes:
        def show(field):
            values = s[field]
            return f'{values[0]:.6f}–{values[1]:.6f}' if values else '无'
        lines.append(f"|{s['scene']}|{show('last_first4_load_after_s_range')}|{show('first_geometry_increment_before_s_range')}|{show('first_applied_target_divergence_before_s_range')}|{show('target_divergence_minus_last_first4_load_s_range')}|")
    lines += ['', '## 实现与路径核对', '',
              '- 每个统计行均核对其独立result的id、seed、场景、配置及trace路径，并核对trace.metadata.job；同对fixed/normal的trace与原记录data路径均不同。J4从各自trace前4000个连续、完成的1ms采样独立重算，并与analyzer原值比较。',
              '- metadata及保存的preview记录分别实际标明`fixed_up_base`与`closest_normal`。每个normal回合都存在非零chosen_direction及非零delta_geometry，且与fixed的几何目标实际分叉，不存在仅换标签、方向分支未介入的现象。JSON保留每回合首次chosen_direction、非零次数及几何目标数组hash。',
              '- trace的FΣ逐步与该回合原paper_truth的all_sum_force_N对齐（最大差<1e−9 N）；在100对的前4000采样内，fixed与normal的FΣ逐样本精确相同，而不是仅四舍五入后的均值相等。',
              f'- 分支设置见`{root}/scripts/run_round2_worker.py`第175–176、185、195行；实际方向选择见`{root}/runtime/tabletop_pace_preview.py`第307–317行。代码及分析源SHA256均记录于JSON。', '',
              '## 可支持的解释与边界', '',
              '前4秒的受保护外接触载荷已经在两个方向产生实际下发参考分叉之前结束，之后即使前4秒内已出现不同几何目标，该窗口剩余FΣ仍为零。因此同种子配对的前4秒载荷积分完全相同有真实采样和执行时序依据。它不支持“normal未执行”或“analyzer误用了fixed路径”的解释。',
              '该窗口主要落在方向差异尚未影响外接触载荷的阶段，不适合检验抬升避障方向的独立收益。仍需使用整回合任务/安全结果、真实柱接触、后续轨迹及几何诊断比较方向设计。这里的相同不能推广为整个闭环轨迹相同、两个方向等效或没有其他性能差异。',
              '末次非零载荷按严格>0记录；JSON同时提供>1e−9 N敏感性时刻。首目标分叉同时给逐位非零和>1e−12 rad口径。所有时间均相对首条物理区间；载荷用步末时刻、参考用步初时刻，避免把同一1ms区间误作先后因果证明。', '',
              '## 复跑', '', '输出JSON与同名Markdown必须均不存在，脚本拒绝覆盖：', '', '```bash',
              f'/home/jun/anaconda3/envs/dummyx_vla/bin/python {Path(__file__).resolve()} --output /tmp/exp2_J4_new_audit.json',
              '```', '', f'脚本SHA256：`{report["script_sha256"]}`。逐回合时间、路径、J4、方向及分叉证据见同名JSON。', '']
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    with markdown.open('x') as stream:
        stream.write('\n'.join(lines))
    print('PASSED', report['passed'], 'JSON', output, 'Markdown', markdown, flush=True)


if __name__ == '__main__':
    main()
