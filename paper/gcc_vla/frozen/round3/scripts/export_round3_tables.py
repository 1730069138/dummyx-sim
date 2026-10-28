#!/usr/bin/env python3
"""Export four current-round rows in the third manuscript's Table7/D2 style.

Requires the completed current200-allocation analysis. Reads no old outcomes and
does not edit Word. Every output file must be new.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import statistics


ROOT = Path(__file__).resolve().parents[1]
TABLE7_HEADERS = ['场景／配置', 'SR[95% CI]', 'STSR[95% CI]', 'PCR[95% CI]', '触限回合', '柱体倾倒']
D2_HEADERS = ['场景／配置', 'J / N·s', 'J₄ / N·s', 'FΣ峰值 / N', 'T / s', '触限子步／回合']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(count, n):
    z = 1.959963984540054
    denominator = 1 + z*z/n
    center = (count/n + z*z/(2*n)) / denominator
    error = z/denominator * math.sqrt(count/n*(1-count/n)/n + z*z/(4*n*n))
    return [100*(center-error), 100*(center+error)]


def rate(group, key):
    value = group[key]
    n, count = value['n'], value['count']
    assert n == 50 and 0 <= count <= n
    calculated = wilson(count, n)
    assert all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(calculated, value['wilson95_percent']))
    low, high = calculated
    return f'{count}/50[{low:.1f}, {high:.1f}]'


def continuous(group, key, digits):
    value = group[key]
    n = value['n']
    if n == 0:
        return '缺失（n=0）'
    assert value['mean'] is not None and math.isfinite(value['mean'])
    sd = f'{value["sd"]:.{digits}f}' if value['sd'] is not None else '—'
    text = f'{value["mean"]:.{digits}f}±{sd}'
    # Full cohorts may still lack a complete4s window after early task termination.
    # Preserve true valid_n rather than presenting a subset as n=50.
    return text if n == 50 else text + f'（n={n}）'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=ROOT/'analysis_jump_final/analysis_round3_jump.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'task3')
    args = parser.parse_args()
    analysis = args.analysis.resolve(strict=True)
    output = args.output_dir.resolve()
    if not analysis.is_relative_to(ROOT) or not output.is_relative_to(ROOT):
        raise ValueError('Read analysis and write exports only within this round3 root')
    files = [output/'table7_jump_expansion.csv', output/'tableD2_jump_expansion.csv',
             output/'table_export_source.json', output/'table_export_notes.md']
    if any(path.exists() for path in files):
        raise FileExistsError('Refusing existing table/export metadata files')
    data = json.loads(analysis.read_text())
    plan = json.loads((ROOT/'statistics_plan.json').read_text())
    assert data['run_id'] == plan['run_id'] == ROOT.name
    assert data['statistics_plan_sha256'] == sha(ROOT/'statistics_plan.json')
    assert data['planned_formal'] == data['completed_formal'] == 200
    assert data['complete_cohort'] and not data['validation_issues']
    assert len(data['per_episode']) == 200
    groups = data['groups']
    table7, tableD2, bindings = [], [], []
    for scene in ['main', 'pillar_x_plus40mm']:
        for config in ['F', 'F_limited']:
            matches = [(i, group) for i, group in enumerate(groups)
                       if group['scene'] == scene and group['config'] == config]
            assert len(matches) == 1
            index, group = matches[0]
            rows = [r for r in data['per_episode'] if r['scene'] == scene and r['config'] == config]
            assert len(rows) == group['n_completed'] == group['n_planned'] == 50
            assert group['complete_cohort']
            assert all(row['experiment'] == 'jump_expansion' and row['stage'] == 'formal' for row in rows)
            for key in ['SR', 'STSR', 'PCR', 'tau_limit_episode', 'pillar_tilt']:
                assert sum(row[key] for row in rows) == group[key]['count']
            for key in ['J', 'J4', 'Fpk', 'T', 'tau_limit_substeps']:
                values = [row[key] for row in rows if row[key] is not None and math.isfinite(row[key])]
                assert len(values) == group[key]['n']
                if values:
                    assert math.isclose(statistics.mean(values), group[key]['mean'], abs_tol=1e-10, rel_tol=1e-10)
                    if len(values) > 1:
                        assert math.isclose(statistics.stdev(values), group[key]['sd'], abs_tol=1e-10, rel_tol=1e-10)
            label = ('原场景' if scene == 'main' else 'X＋40 mm') + '／' + ('F' if config == 'F' else 'F＋跳变限幅')
            table7.append(dict(zip(TABLE7_HEADERS, [label, rate(group,'SR'), rate(group,'STSR'), rate(group,'PCR'),
                f"{group['tau_limit_episode']['count']}/50", f"{group['pillar_tilt']['count']}/50"])))
            tableD2.append(dict(zip(D2_HEADERS, [label, continuous(group,'J',2), continuous(group,'J4',2),
                continuous(group,'Fpk',2), continuous(group,'T',2), continuous(group,'tau_limit_substeps',1)])))
            bindings.append(dict(label=label, scene=scene, config=config,
                source_group_pointer=f'/groups/{index}', n=50,
                continuous_valid_n={key:group[key]['n'] for key in ['J','J4','Fpk','T','tau_limit_substeps']},
                source_episode_ids=[row['id'] for row in rows]))
    assert len(table7) == len(tableD2) == 4
    output.mkdir(parents=True, exist_ok=True)
    for path, headers, rows in [(files[0],TABLE7_HEADERS,table7),(files[1],D2_HEADERS,tableD2)]:
        with path.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
    record = dict(status='PASS_CURRENT200_ONLY', analysis=str(analysis), analysis_sha256=sha(analysis),
        statistics_plan_sha256=data['statistics_plan_sha256'], script=str(Path(__file__).resolve()),
        script_sha256=sha(__file__), source_analyzer_sha256=data['script_sha256'], bindings=bindings,
        output_sha256={str(path):sha(path) for path in files[:2]},
        formatting=dict(rate='count/50[Wilson lower, upper]; limits in percent with1decimal',
                        load='mean±sampleSD,2decimals', saturation_substeps='mean±sampleSD,1decimal',
                        missing_window='No imputation; valid_n annotation if less than50'),
        old_outcomes_read=False, word_modified=False, physics_steps=0, policy_calls=0)
    with files[2].open('x') as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    notes = ['# 第三轮限跳扩样表格导出', '',
             '本次仅从本轮正式200回合最终统计导出4行；不拷贝第二轮5例探针、旧表7或D2的数值。', '',
             '表7格式：计数/50及Wilson95%百分区间（1位小数），触限与倾倒为计数/50。',
             'D2格式：J、J₄、峰值与T为均值±样本SD（2位小数）；触限子步/回合为均值±样本SD（1位小数）。',
             'J₄不足完整4s时不补零；有效n<50则在该数值后注明n，其余有效n见table_export_source.json。',
             'T包含成功完成与失败终止时长。J为受保护体对外接触力模之和的积分，不是矢量净冲量。',
             '仅生成独立CSV，不编辑Word。目标文件已存在时拒绝覆盖。', '']
    files[3].write_text('\n'.join(notes))
    print(json.dumps(dict(status=record['status'], rows=4, outputs=[str(p) for p in files]), ensure_ascii=False))


if __name__ == '__main__':
    main()
