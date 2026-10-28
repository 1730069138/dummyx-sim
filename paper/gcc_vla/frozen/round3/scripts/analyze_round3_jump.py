#!/usr/bin/env python3
"""Read-only analysis of frozen round3 F/F_limited allocations.

No MuJoCo, worker, client or controller imports. Only manifest-listed attempts in
the current run are considered. Existing output directories are refused.
"""
from pathlib import Path
import argparse
import collections
import datetime
import importlib.util
import json
import math

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / 'scripts/analyze_round2.py'
spec = importlib.util.spec_from_file_location('round3_readonly_helpers', HELPER_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

# Avoid decompressing unrelated images/kinematic arrays. These are exactly the
# fields used by validate_trace, trace_summary, and current-run limiter checks.
TRACE_FIELDS = '''metadata_json step_completed contact_offsets contact_geom1
contact_geom2 contact_role1 contact_role2 contact_wrench_world_on_geom2
contact_protected_external contact_protected_internal time_before_s time_after_s
target_geometry_uG target_nominal_a delta_geometry target_contact_ubar
delta_contact_clipped u_post_guard delta_guard u_applied limiter_delta
ctrl_written actuator_force_arm actuator_at_limit_arm actuator_demand_exceeds_arm
sum_external_contact_force_N phase_pre phase_metric_valid_pre
delta_u_from_previous previous_applied_valid'''.split()


def within(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Source outside explicitly allowed current run: ' + str(path))
    return path


def load_trace(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in TRACE_FIELDS}


def summarize_saved_trace(trace_path, data_path, result, ident, threshold=None,
                          check_limiter=True):
    """Shared pure saved-array path; also callable for explicitly labelled regression."""
    z = load_trace(trace_path)
    meta, done = base.validate_trace(z)
    assert done.all(), 'Nontechnical completion has an unfinished physical interval'
    assert done.any(), 'No completed physical intervals'
    row = base.data_endpoints(data_path, result)
    row.update(base.trace_summary(z, meta, done, ident))
    row['failure_category'] = (None if row['SR'] else 'tool_floor'
                               if row['termination_reason'] in ('tool_floor', 'tool_floor_contact')
                               else base.failure_category(row['termination_reason']))
    row['endpoint_source'] = 'original_data_npz' if data_path else 'explicit_result_metrics'
    if data_path:
        truth = Path(data_path).with_name('paper_truth_ep_001.npz')
        assert truth.exists(), 'Missing original paper_truth in current attempt'
        with np.load(truth, allow_pickle=False) as recorded:
            force = z['sum_external_contact_force_N'][done]
            assert len(recorded['time_s']) == len(force)
            difference = abs(recorded['all_sum_force_N'] - force)
            assert np.allclose(recorded['all_sum_force_N'], force, atol=1e-9, rtol=1e-11)
            row['truth_force_max_abs_difference_N'] = float(difference.max())
        row['paper_truth_path'] = str(truth)
    arm = meta['arm_actuator_indices']
    ranges = np.asarray(meta['actuator_forcerange'])[arm]
    assert np.all(np.asarray(meta['actuator_forcelimited'])[arm])
    assert np.array_equal(ranges, np.tile([-8., 8.], (6, 1))), 'Unexpected actual six-arm force bounds'
    if check_limiter:
        previous = z['previous_applied_valid'].astype(bool) & done
        if threshold is None:
            assert meta.get('target_jump_limit_rad') is None
            assert np.allclose(z['limiter_delta'][done], 0, atol=1e-12, rtol=0)
        else:
            limit = np.asarray(threshold, float)
            assert np.array_equal(np.asarray(meta['target_jump_limit_rad']), limit)
            assert np.all(abs(z['delta_u_from_previous'][previous]) <= limit[None, :] + 1e-12)
            assert np.allclose(z['limiter_delta'][done & ~previous], 0, atol=1e-12, rtol=0)
    row['limiter_changed_substeps'] = int(np.count_nonzero(np.max(abs(z['limiter_delta'][done]), axis=1) > 1e-12))
    return row, meta


def verify_identity(job, result, meta, expected, threshold):
    assert job['experiment'] == 'jump_expansion'
    assert job['stage'] == 'formal'
    assert job['arm'] == job['config_name'] == expected['config_name']
    for key in ['id', 'seed', 'gpu', 'rep', 'config_name']:
        assert job[key] == expected[key], ('manifest versus plan', key)
        assert result[key] == job[key], ('result versus manifest', key)
    assert job['scene_id'] == expected['scene'] == result['scene_id']
    assert result['experiment'] == 'jump_expansion' and result['stage'] == 'formal'
    mjob = meta.get('job') or {}
    for key in ['id', 'seed', 'gpu', 'config_name', 'experiment', 'stage', 'scene_id']:
        assert mjob.get(key) == job[key], ('trace identity', key)
    assert meta['geometry_direction'] == 'fixed_up_base'
    assert meta['geometry_enabled'] and meta['contact_enabled'] and meta['current_guard']
    if job['config_name'] == 'F_limited':
        assert np.array_equal(np.asarray(meta['target_jump_limit_rad']), np.asarray(threshold))
    else:
        assert meta.get('target_jump_limit_rad') is None


def build_summaries(rows, planned_n):
    summary = []
    loads = []
    groups = []
    for scene in ['main', 'pillar_x_plus40mm']:
        for config in ['F', 'F_limited']:
            rr = [r for r in rows if r['scene'] == scene and r['config'] == config]
            group = dict(scene=scene, config=config, n_planned=planned_n, n_completed=len(rr),
                         complete_cohort=len(rr) == planned_n)
            flat = dict(scene=scene, config=config, n_planned=planned_n, n_completed=len(rr))
            for metric in ['SR', 'STSR', 'PCR', 'tau_limit_episode', 'pillar_tilt']:
                k = sum(r[metric] for r in rr)
                low, high = base.wilson(k, len(rr))
                group[metric] = dict(count=k, n=len(rr), percent=100*k/len(rr) if rr else None,
                                     wilson95_percent=[low, high])
                flat.update({metric+'_count':k, metric+'_n':len(rr),
                             metric+'_percent':group[metric]['percent'],
                             metric+'_wilson95_low_percent':low, metric+'_wilson95_high_percent':high})
            for metric in ['J', 'J4', 'Fpk', 'T', 'tau_limit_substeps']:
                value = base.stat([r[metric] for r in rr])
                group[metric] = value
                loads.append(dict(scene=scene, config=config, metric=metric,
                                  n_planned=planned_n, n_completed=len(rr), n_valid=value['n'],
                                  mean=value['mean'], sample_SD=value['sd'], min=value['min'], max=value['max']))
            group['failure_categories'] = dict(collections.Counter(r['failure_category'] for r in rr if not r['SR']))
            groups.append(group)
            summary.append(flat)
    return groups, summary, loads


def build_tests(rows, ready):
    tests = []
    for family, metrics, size in [('binary6', ['SR', 'STSR', 'PCR'], 6), ('load4', ['J', 'J4'], 4)]:
        items = []
        for scene in ['main', 'pillar_x_plus40mm']:
            limited = [r for r in rows if r['scene'] == scene and r['config'] == 'F_limited']
            full = [r for r in rows if r['scene'] == scene and r['config'] == 'F']
            for metric in metrics:
                one = dict(family=family, scene=scene, outcome=metric,
                           config_A='F_limited', config_B='F', effect_direction='F_limited minus F',
                           planned_cohort_complete=ready, cohort_n_limited=len(limited), cohort_n_F=len(full),
                           p_raw=None, p_holm=None)
                if family == 'binary6':
                    ka, kb = sum(r[metric] for r in limited), sum(r[metric] for r in full)
                    one.update(test='two-sided Fisher exact', n_limited=len(limited), n_F=len(full),
                               count_limited=ka, count_F=kb,
                               risk_difference_pp=100*(ka/len(limited)-kb/len(full)) if limited and full else None)
                    if ready:
                        result = stats.fisher_exact([[ka, len(limited)-ka], [kb, len(full)-kb]], alternative='two-sided')
                        one.update(p_raw=float(result.pvalue),
                                   odds_ratio=float(result.statistic) if np.isfinite(result.statistic) else None,
                                   odds_ratio_nonfinite=not bool(np.isfinite(result.statistic)))
                else:
                    aa = [r[metric] for r in limited if r[metric] is not None and np.isfinite(r[metric])]
                    bb = [r[metric] for r in full if r[metric] is not None and np.isfinite(r[metric])]
                    one.update(test='two-sided Mann-Whitney U; asymptotic tie correction and continuity correction',
                               n_limited=len(aa), n_F=len(bb),
                               mean_difference=float(np.mean(aa)-np.mean(bb)) if aa and bb else None,
                               coverage_note='J4: observed complete4s only; early termination missing, no zero fill; available subset may be selected by termination')
                    if ready and aa and bb:
                        result = stats.mannwhitneyu(aa, bb, alternative='two-sided', method='asymptotic', use_continuity=True)
                        one.update(p_raw=float(result.pvalue), U_limited=float(result.statistic),
                                   cliffs_delta_limited_minus_F=2*float(result.statistic)/(len(aa)*len(bb))-1)
                items.append(one)
        base.holm(items, size)
        tests.extend(items)
    return tests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--attempt-dir', type=Path, action='append', default=[],
                        help='Additional explicit recovery directory with protocol.json and allocations.json')
    parser.add_argument('--technical-journal', type=Path,
                        help='Explicit current-run technical interruptions JSON list; default audit/technical_interruptions.json if present')
    parser.add_argument('--require-complete', action='store_true', help='Exit2 after saving audit if200 valid allocations not complete or issues exist')
    args = parser.parse_args()
    root = args.run_root.resolve(strict=True)
    output = within(args.output_dir, root)
    if output.exists():
        raise FileExistsError('Refusing existing output directory')
    plan_path = root / 'statistics_plan.json'
    plan = base.readj(plan_path)
    plan_sha = base.sha(plan_path)
    assert plan_sha == (root/'statistics_plan.sha256').read_text().strip()
    assert plan['design']['formal_total'] == 200
    assert plan['design']['n_per_scene_config'] == 50
    planned = {r['id']:r for r in plan['allocations']}
    assert len(planned) == 200
    thresholds = plan['jump_limiter']['threshold_rad']
    sources = {str(plan_path):plan_sha, str(HELPER_PATH):base.sha(HELPER_PATH),
               str(Path(__file__).resolve()):base.sha(__file__)}
    attempts = []
    issues = []
    candidates = collections.defaultdict(list)
    run_dirs = [root/'experiments/formal'] + [within(p, root) for p in args.attempt_dir]
    assert len(set(p.resolve() for p in run_dirs)) == len(run_dirs), 'Duplicate attempt directory'
    for directory in run_dirs:
        if not directory.exists():
            issues.append(dict(status='missing_attempt_directory', path=str(directory)))
            continue
        protocol_path, manifest_path = directory/'protocol.json', directory/'allocations.json'
        protocol, jobs = base.readj(protocol_path), base.readj(manifest_path)
        assert base.sha(protocol_path) == (directory/'protocol.sha256').read_text().strip()
        assert protocol['statistics_plan_sha256'] == plan_sha
        for path in [protocol_path, manifest_path]:sources[str(path)] = base.sha(path)
        seen = set()
        for job in jobs:
            identity = job['id']
            assert identity not in seen, 'Duplicate ID within manifest'
            seen.add(identity)
            assert identity in planned, 'Manifest allocation not in frozen plan'
            expected = planned[identity]
            assert job['experiment'] == 'jump_expansion' and job['stage'] == 'formal'
            assert job['arm'] == job['config_name'] == expected['config_name']
            assert job['scene_id'] == expected['scene']
            for field in ['seed', 'gpu', 'rep']:
                assert job[field] == expected[field], ('manifest versus frozen plan', identity, field)
            path = directory/f"worker_gpu{job['gpu']}"/identity
            rp = path/'result.json'
            attempt = dict(id=identity, experiment='jump_expansion', stage='formal',
                           config=job['config_name'], scene=job['scene_id'], seed=job['seed'], gpu=job['gpu'],
                           attempt_dir=str(path), result_path=str(rp), technical=None,
                           endpoint_status='unknown', included_in_formal=False)
            attempts.append(attempt)
            if not rp.exists():
                attempt.update(status='missing_result', reason='pending_or_missing_without_result')
                continue
            sources[str(rp)] = base.sha(rp)
            result = base.readj(rp)
            reason = str((result.get('result') or {}).get('reason', (result.get('metrics') or {}).get('reason','')))
            technical = result.get('error') is not None or reason.startswith(('policy_error:', 'simulation_error:', 'perception_error:')) or bool((result.get('metrics') or {}).get('metric_error'))
            attempt.update(technical=technical, reason=reason, error=result.get('error'))
            if technical:
                attempt.update(status='technical_attempt', endpoint_status='unknown technical interruption, not task failure')
                continue
            try:
                tp = within(result.get('trace_path', path/'substep_trace.npz'), root)
                assert tp.is_relative_to(path.resolve()), 'Trace outside own attempt'
                assert tp.exists(), 'Missing required current-run trace'
                dp = base.find_data(path, result)
                if dp:
                    dp = within(dp, root)
                    assert dp.is_relative_to(path.resolve()), 'Data outside own attempt'
                ident = dict(experiment='jump_expansion', stage='formal', config=job['config_name'],
                             scene=job['scene_id'], seed=job['seed'], gpu=job['gpu'])
                row, meta = summarize_saved_trace(tp, dp, result, ident,
                    thresholds if job['config_name'] == 'F_limited' else None)
                verify_identity(job, result, meta, expected, thresholds)
                row.update(ident, id=identity, attempt_dir=str(path), result_path=str(rp),
                           trace_path=str(tp), data_path=str(dp) if dp else None)
                candidates[identity].append(row)
                attempt.update(status='valid_task_completion', endpoint_status='observed original frozen predicate',
                               SR=row['SR'], STSR=row['STSR'], PCR=row['PCR'], pillar_tilt=row['pillar_tilt'],
                               termination_reason=row['termination_reason'], failure_category=row['failure_category'])
                for item in [tp,dp,Path(row['paper_truth_path']) if row.get('paper_truth_path') else None]:
                    if item:sources[str(item)] = base.sha(item)
            except Exception as exc:
                attempt.update(status='validation_failed_unknown', endpoint_status='unknown, not counted task failure', error=repr(exc))
                issues.append(dict(id=identity, path=str(rp), status='trace_identity_or_endpoint_validation_failed', error=repr(exc)))
    journal_path = args.technical_journal or root/'audit/technical_interruptions.json'
    if Path(journal_path).exists():
        journal_path = within(journal_path, root)
        records = base.readj(journal_path)
        assert isinstance(records, list), 'Technical journal must be an explicit list'
        sources[str(journal_path)] = base.sha(journal_path)
        for record in records:
            identity = record['id']
            assert identity in planned
            original = within(record['original_path'], root)
            existing = next((a for a in attempts if a['attempt_dir'] == str(original)),None)
            if existing and existing['status'] == 'valid_task_completion':
                issues.append(dict(id=identity, status='journal_conflicts_with_valid_task_endpoint', path=str(original)))
                continue
            if existing:
                existing.update(technical=True, status='technical_attempt', reason=record.get('reason','technical_interruption_without_result'),
                                endpoint_status='unknown technical interruption, not task failure', source_journal=str(journal_path))
            else:
                expected=planned[identity]
                attempts.append(dict(id=identity, experiment='jump_expansion', stage='formal',
                    config=expected['config_name'], scene=expected['scene'], seed=expected['seed'], gpu=expected['gpu'],
                    attempt_dir=str(original), result_path=None, technical=True, included_in_formal=False,
                    status='technical_attempt', reason=record.get('reason','technical_interruption_without_result'),
                    endpoint_status='unknown technical interruption, not task failure', source_journal=str(journal_path)))
    rows=[]
    for attempt in attempts:
        if attempt['status']=='missing_result' and candidates[attempt['id']]:
            issues.append(dict(id=attempt['id'],status='unexplained_missing_prior_attempt_with_recovery',
                               path=attempt['attempt_dir'],
                               requirement='Explicit current-run technical journal must explain original attempt'))
    for identity in planned:
        valid=candidates[identity]
        if len(valid)>1:
            issues.append(dict(id=identity,status='multiple_valid_completions_no_automatic_selection',paths=[r['result_path'] for r in valid]))
        elif len(valid)==1:
            rows.append(valid[0])
            next(a for a in attempts if a['result_path']==valid[0]['result_path'])['included_in_formal']=True
    groups, summary, loads=build_summaries(rows,50)
    ready=len(rows)==200 and not issues and all(g['complete_cohort'] for g in groups)
    tests=build_tests(rows,ready)
    missing=[identity for identity in planned if identity not in {r['id'] for r in rows}]
    outcome=dict(generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        run_id=plan['run_id'], planned_formal=200, completed_formal=len(rows), complete_cohort=ready,
        technical_attempts=sum(a['technical'] is True for a in attempts), unresolved_allocation_ids=missing,
        validation_issues=issues, groups=groups, statistical_tests=tests, per_episode=rows, attempts=attempts,
        sources=[dict(path=k,sha256=v) for k,v in sorted(sources.items())], statistics_plan_sha256=plan_sha,
        script_sha256=base.sha(__file__), helper_sha256=base.sha(HELPER_PATH),
        software=dict(numpy=np.__version__,scipy=__import__('scipy').__version__),
        physics_steps=0,policy_calls=0,
        notes=['Only explicit current-run formal/recovery manifests; smoke and old probes are excluded.',
               'Actual task failures retained; technical/missing/validation-unknown attempts are not task failures.',
               'Multiple valid completions for one allocation require manual resolution; never select by outcome.',
               'J4 requires4000completed contiguous1ms samples; early termination yields null, no imputation.',
               'Fisher6 and MWU4 are separate Holm families; effects F_limited minus F.',
               'No confirmatory pvalues until all200 frozen allocations are valid and cohort audit passes.',
               'Peak/time/saturation/tilt descriptive; observed associations do not establish module causality.'])
    output.mkdir(parents=True,exist_ok=False)
    for name, values in [('summary.csv',summary),('load_summary.csv',loads),('per_episode.csv',rows),
                         ('failure_classification.csv',attempts),('statistical_tests.csv',tests),
                         ('technical_attempts.csv',[a for a in attempts if a['technical'] is True])]:
        if values:base.writecsv(output/name,values)
        else:(output/name).write_text('')
    base.writej(output/'analysis_round3_jump.json',outcome)
    lines=['# 第三轮冻结限跳对照统计','',f'正式有效{len(rows)}/200；技术尝试{outcome["technical_attempts"]}；验证问题{len(issues)}；完整队列={ready}。不纳入2个冒烟或任何第二轮回合。','',
           '|场景|配置|n|SR|STSR|PCR|触限|倾倒|','|---|---|---|---|---|---|---|---|']
    for g in groups:
        fmt=lambda k:f"{g[k]['count']}/{g[k]['n']}"
        lines.append(f"|{g['scene']}|{g['config']}|{g['n_completed']}/50|{fmt('SR')}|{fmt('STSR')}|{fmt('PCR')}|{fmt('tau_limit_episode')}|{fmt('pillar_tilt')}|")
    lines+=['','连续量均值、样本SD与有效n见load_summary.csv；各比例Wilson95%区间见summary.csv。J为受保护体对外接触力模之和的积分，J4缺少完整4s时为null；T含成功及失败的实际终止时长。','',
            '6项Fisher与4项MWU分别Holm校正；效应方向均为F_limited−F。MWU采用双侧渐近、结校正和连续性校正；Cliff’s δ=2U/(n_limited×n_F)−1。未完成全部冻结分配或存在审计问题时不生成确认性p值。','',
            '所有任务失败保留；技术中断、缺失或验证未知单列，不算任务失败。恢复目录必须通过--attempt-dir明确提供并具有protocol/allocations；多个有效完成不自动选择。分析不修改计划、阈值、算法或原记录。','']
    (output/'results.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(output),completed=len(rows),planned=200,complete=ready,
                          technical=outcome['technical_attempts'],issues=len(issues)),ensure_ascii=False))
    if args.require_complete and not ready:
        raise SystemExit(2)


if __name__=='__main__':
    main()
