#!/usr/bin/env python3
"""Verify final Exp3 tables/windows and write a new interpretation report only."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir', type=Path, default=root / 'analysis_final')
    parser.add_argument('--matched', type=Path, default=root / 'audit/exp3_matched_failure_windows/comparison.json')
    parser.add_argument('--recheck', type=Path, default=root / 'audit/exp3_matched_failure_windows_recheck/comparison.json')
    parser.add_argument('--output', type=Path, default=root / 'audit/exp3_guard_interpretation.md')
    args = parser.parse_args()
    output = args.output.resolve()
    evidence = output.with_suffix('.json')
    if output.exists() or evidence.exists():
        raise FileExistsError('Refusing existing output report/evidence')
    analysis = args.analysis_dir.resolve() / 'analysis_round2.json'
    data = json.loads(analysis.read_text())
    comparison = json.loads(args.matched.read_text())
    recheck = json.loads(args.recheck.read_text())
    assert comparison == recheck, 'Supplementary script rerun differs'
    assert comparison['analysis_sha256'] == sha(analysis)
    assert data['completed_formal'] == 440 and not data['validation_issues']
    technical = [r for r in data['attempts'] if r['technical']]
    rows = [r for r in data['per_episode'] if r['experiment'] == 'exp3']
    assert len(rows) == 200
    windows_path = analysis.parent / 'exp3_failure_guard_windows.csv'
    with windows_path.open() as stream:
        windows = list(csv.DictReader(stream))
    checks = {'comparison_reproduction_exact_equal': True,
              'matched_analysis_identity_correct': True, 'main_completed_440': True,
              'final_validation_issues_zero': True}
    sources = [analysis, windows_path, args.matched, args.recheck,
               root / 'scripts/analyze_round2.py',
               root / 'scripts/compare_guard_failure_windows.py',
               Path(__file__).resolve()]
    main_table = []
    for scene in ['main', 'pillar_x_plus40mm']:
        for config in ['F', 'B']:
            selected = [r for r in rows if r['scene'] == scene and r['config'] == config]
            assert len(selected) == 50
            group = next(g for g in data['groups'] if g['experiment'] == 'exp3'
                         and g['scene'] == scene and g['config'] == config)
            for field in ['SR', 'STSR', 'PCR', 'pillar_tilt', 'tau_limit_episode']:
                assert sum(r[field] for r in selected) == group[field]['count']
            failed = [r for r in selected if not r['SR']]
            failure_reasons = dict(collections.Counter(r['termination_reason'] for r in failed))
            window = next(w for w in windows if w['scene'] == scene and w['config'] == config)
            den = sum(r['guard_window_substeps'] for r in failed)
            num = sum(r['guard_window_active'] for r in failed)
            assert den == int(window['action_cached_pickup_window_substeps'])
            assert num == int(window['action_cached_guard_active'])
            assert len(failed) == int(window['failed_episodes'])
            assert abs(num / den - float(window['action_cached_guard_fraction'])) < 1e-15
            main_table.append(dict(scene=scene, config=config, n=50,
                SR=group['SR']['count'], STSR=group['STSR']['count'], PCR=group['PCR']['count'],
                pillar_tilt=group['pillar_tilt']['count'], J=group['J'], J4=group['J4'],
                Fpk=group['Fpk'], T=group['T'], tau_limit_episode=group['tau_limit_episode']['count'],
                failure_reasons=failure_reasons, failure_window=window))
    checks['all_4_main_groups_and_failure_windows_match_per_episode'] = True
    checks['all_100_B_pillar_contact_and_tilt'] = all(r['PCR'] and r['pillar_tilt'] for r in rows if r['config'] == 'B')
    checks['all_37_F_failures_are_step_limit'] = all(r['termination_reason'] == 'step_limit' for r in rows if r['config'] == 'F' and not r['SR'])
    for record in comparison['summary']:
        scene = record['scene']
        failed = [r for r in rows if r['scene'] == scene and r['config'] == 'F' and not r['SR']]
        seeds = {r['seed'] for r in failed}
        selected = failed if record['subset'] == 'F_failed' else [r for r in rows if r['scene'] == scene and r['config'] == 'B' and r['seed'] in seeds]
        assert len(selected) == record['n']
        for field in ['SR', 'STSR', 'PCR']:
            assert sum(r[field] for r in selected) == record[field]
        for prefix in ['guard_window', 'diagnostic_lift_window']:
            assert sum(r[prefix + '_substeps'] for r in selected) == record[prefix + '_substeps']
            assert sum(r[prefix + '_active'] for r in selected) == record[prefix + '_active']
    checks['matched_F_failed_seed_subsets_recomputed_match'] = True
    lines = ['# Exp3：最终几何复核的整体效果与失败窗口解释', '',
             f'只读依据：`{analysis}`、其`exp3_failure_guard_windows.csv`及`{args.matched.resolve()}`。主计划440/440已完成，技术中断记录{len(technical)}项单独保留，最终validation_issues为0。本文只讨论Exp3的200回合，不与Exp1/Exp2、probe、历史600或72回合合并。', '',
             '## 完整F/B主比较', '',
             'F为完整当前方法；B保留动作级几何与接触顺应，关闭逐1ms子步执行的最终几何复核。每个场景每组n=50，成功口径不变。STSR仅覆盖论文定义的柱接触、工具触地及柱体倾倒事件。', '',
             '|场景|配置|SR|STSR|真实柱接触PCR|柱体倾倒|全程积分(N·s)|前4s积分(N·s)|峰值力(N)|完成/终止时间(s)|',
             '|---|---|---|---|---|---|---|---|---|---|']
    for g in main_table:
        name = '原场景' if g['scene'] == 'main' else 'X＋40 mm'
        fmt = lambda field: f"{g[field]['mean']:.2f}±{g[field]['sd']:.2f}"
        lines.append(f"|{name}|{g['config']}|{g['SR']}/50|{g['STSR']}/50|{g['PCR']}/50|{g['pillar_tilt']}/50|{fmt('J')}|{fmt('J4')}|{fmt('Fpk')}|{fmt('T')}|")
    lines += ['',
              '关闭复核后两场景全部100回合发生真实柱接触和柱体倾倒，STSR均为0；保留复核的F分别为9/50、33/50发生柱接触，均未倾倒。结果支持该层在当前执行链中具有可检验的整体几何约束作用，但F在X＋40 mm仍有较高柱接触率，不能写成严格安全保证。',
              '代价必须并列保留：原场景F/B任务成功为46/50与49/50，X＋40 mm为17/50与44/50。最终JSON预注册六项二元检验族中，原场景SR的Holm校正p=0.36218；X＋40 mm SR的校正p=1.4862e−7；两场景STSR/PCR的校正p分别为1.4946e−18和8.8847e−6。这是既定整层开关的结果差异，不是特定触发时刻导致失败的因果判定。',
              '全程载荷积分F较低，但它是受保护体外接触力范数之和的积分，受接触来源和整个执行过程共同影响，不等于净冲量；前4s积分没有方向性改善。峰值亦不宜宣称已普遍消除。必须保留整层约束收益与任务推进代价，不能只取成功率或全程载荷其中一项。', '',
              '## 各自失败回合的预定窗口', '',
              '主窗口按各回合保存的动作级pickup_gate为真的已完成子步，截到拾起后首次直接placement.valid谓词；无该谓词时截到终止。首次直接placement.valid不是持续0.5s的严格任务成功。主窗口合并触发率=全部有效子步内guard_active总数/有效子步总数，不是回合触发率，也不是各回合百分比的算术平均。', '',
              '|场景/配置|本组失败回合|实际终止原因|主窗口触发子步/有效子步|合并比例|统一1ms抬升锁存敏感性比例|',
              '|---|---|---|---|---|---|']
    for g in main_table:
        w = g['failure_window']; names = '原场景' if g['scene'] == 'main' else 'X＋40 mm'
        lines.append(f"|{names}/{g['config']}|{w['failed_episodes']}|{json.dumps(g['failure_reasons'], ensure_ascii=False)}|{w['action_cached_guard_active']}/{w['action_cached_pickup_window_substeps']}|{100*float(w['action_cached_guard_fraction']):.3f}%|{100*float(w['diagnostic1ms_guard_fraction']):.3f}%|")
    lines += ['',
              'F原场景4次、X＋40 mm33次失败均为step_limit，不能改写为复核不可行或安全中止。F失败窗口内有持续复核干预，但“失败且存在触发”不能证明触发是失败原因；障碍、策略参考和既往轨迹同时影响干预与任务进展。B零触发是关闭配置强制产生的结构性零，不是无几何风险、无触发必要或关闭复核改善安全的实测证明。',
              'F和B各自失败回合分别为4对1、33对6，种子集合和窗口长度不同。主表不能作为同条件严格配对窗口的比较。', '',
              '## 固定F失败种子的补充描述', '',
              '另选择F失败种子，列出同场景同种子的B，不论B成功与否。这是按F结果选择的事后条件子集，不在主检验族，不附p值，不能解释为随机化的失败恢复率。', '',
              '|场景/子集|n|SR|STSR|PCR|主窗口触发/有效子步|主窗口合并比例|回合比例算术平均|',
              '|---|---|---|---|---|---|---|---|']
    for s in comparison['summary']:
        names = '原场景' if s['scene'] == 'main' else 'X＋40 mm'
        label = 'F失败种子/F' if s['subset'] == 'F_failed' else '相同F失败种子/B'
        lines.append(f"|{names}/{label}|{s['n']}|{s['SR']}/{s['n']}|{s['STSR']}/{s['n']}|{s['PCR']}/{s['n']}|{s['guard_window_active']}/{s['guard_window_substeps']}|{100*s['guard_window_pooled_fraction']:.3f}%|{100*s['guard_window_mean_episode_fraction']:.3f}%|")
    lines += ['',
              '对应B在原场景4/4、X＋40 mm29/33完成任务，但相应4/4和33/33均发生柱接触，STSR为0，且最终全组记录表明这些B回合亦全部倾倒。此观察提示当前复核存在任务推进与接触约束的权衡，不证明可以通过撞柱安全完成，也不支持隐藏F失败。',
              '两个子集的窗口均由各自轨迹的首次放置谓词或终止截断，B成功往往更早结束，暴露时长不同；即使种子相同也不是固定状态的局部反事实。X＋40 mm F的合并比例25.610%与回合比例均值29.650%不同，原因是子步加权与回合等权不同，应保持原预定合并口径。', '',
              '## 协调机制能解释到哪一层', '',
              '实际执行顺序是在动作级几何参考上计算接触参考，再由最终几何复核限制候选目标；但这层逐1ms运行、作用于完整候选绝对关节位置参考，既可能限制接触增量，也可能修正名义/动作级几何目标的当前局部不满足。F/B关闭的是这一整层，包括其频率和门控，不能把结果专属归因为“接触顺应引起几何退化后的协同修复”。',
              '因此可写“逐子步最终几何复核在当前整条执行链中降低柱接触与倾倒，伴随任务推进限制”；不可写“已证明接触专属协同收益”“guard触发导致这37次失败”“触发越多风险越大”或“严格非线性安全投影”。若要隔离接触专属协调，仍需保留几何复核和参考路径一致，仅改变接触增量协调方式的独立对照。', '',
              '## 来源及复跑', '',
              f'- 主统计：`{root}/scripts/analyze_round2.py`，窗口定义见153–177行，成功和安全真值来源见179–190行。',
              f'- 执行顺序：`{root}/runtime/deploy_screwdriver_client_old.py`第541–549行；最终复核：`{root}/runtime/tabletop_pace_preview.py`第466–559行。',
              f'- 补充窗口脚本：`{root}/scripts/compare_guard_failure_windows.py`。已实际复跑至`{args.recheck.resolve().parent}`，与原comparison逐项完全一致；不覆盖原结果。',
              f'- 本报告校验脚本：`{Path(__file__).resolve()}`。可用--output指定新Markdown路径；报告及同名JSON若存在均拒绝覆盖。',
              '- 逐来源SHA256、完整主表、窗口及一致性检查保存在同名JSON；原始数据、主统计、冻结算法均未修改。', '']
    out = dict(passed=all(checks.values()), checks=checks, main_table=main_table,
               matched_summary=comparison['summary'], technical_records=len(technical),
               source_sha256={str(p.resolve()): sha(p) for p in sources},
               physics_steps=0, policy_calls=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write('\n'.join(lines))
    with evidence.open('x') as stream:
        json.dump(out, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('PASSED', out['passed'], output, evidence)


if __name__ == '__main__':
    main()
