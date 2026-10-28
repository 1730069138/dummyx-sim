#!/usr/bin/env python3
"""Reproduce the frozen 237 first-request contrasts, using saved arrays only.

Replays the actual stage/list/group/reference loops in audit_final_allocation.py.
All outputs stay inside this round3 work root; existing outputs are refused.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from lxml import etree
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/home/jun/dummyx-sim/outputs/gcc_vla_experiments_round2_20261028_190400')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=OLD)
    parser.add_argument('--word', type=Path, default=Path('/media/jun/JUN/JUN/GCC-VLA_第三轮修改稿.docx'))
    parser.add_argument('--output-root', type=Path, default=ROOT)
    args = parser.parse_args()
    source = args.source_root.resolve(strict=True)
    target = args.output_root.resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError('All outputs must stay under current round3 root')
    paths = dict(csv=target / 'initial_block_contrasts.csv',
                 json=target / 'task2/initial_block_contrasts.json',
                 aggregate=target / 'task2/initial_block_contrasts_aggregate.csv',
                 sources=target / 'task2/source_sha256.csv', report=target / 'task2/report.md')
    if any(p.exists() for p in paths.values()):
        raise FileExistsError('Refusing existing audit outputs')
    source_hashes = {}

    def digest(path):
        path = Path(path).resolve(strict=True)
        name = str(path)
        if name not in source_hashes:
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(2 ** 20), b''):
                    h.update(block)
            source_hashes[name] = h.hexdigest()
        return source_hashes[name]

    def readj(path):
        digest(path)
        return json.loads(Path(path).read_text())

    def array_digest(array):
        # Includes shape/dtype to avoid conflating arrays with equal raw bytes.
        h = hashlib.sha256()
        h.update(str(array.dtype).encode())
        h.update(json.dumps(list(array.shape)).encode())
        h.update(np.ascontiguousarray(array).tobytes())
        return h.hexdigest()

    def saved_start(job, stage, job_path):
        chunk = job_path / 'policy_chunk_000.npz'
        lineage = job_path / 'policy_lineage.jsonl'
        if not chunk.exists() or not lineage.exists():
            raise FileNotFoundError('Initial block missing: ' + str(job_path))
        digest(chunk)
        digest(lineage)
        with np.load(chunk, allow_pickle=False) as z:
            actions = z['actions'].copy()
            state = z['state'].copy()
        first = json.loads(lineage.read_text().splitlines()[0])
        return dict(job=job, stage=stage, path=job_path, actions=actions, state=state,
                    obs=first['observation_hash'], chunk=chunk, lineage=lineage)

    starts = collections.defaultdict(list)
    allocations = []
    missing_recovered = []
    for stage in ['smoke', 'exp1', 'jump', 'exp23', 'exp23_recovery']:
        directory = source / 'experiments' / stage
        if not directory.exists():
            continue
        jobs = readj(directory / 'allocations.json')
        for job in jobs:
            path = directory / f"worker_gpu{job['gpu']}" / job['id']
            result_path = path / 'result.json'
            record = dict(job=job, stage=stage, path=path, result_path=result_path,
                          exists=result_path.exists())
            allocations.append(record)
            if not result_path.exists():
                recovered = source / 'experiments/exp23_recovery' / f"worker_gpu{job['gpu']}" / job['id'] / 'result.json'
                if stage == 'exp23' and job['seed'] == 860049 and job['gpu'] == 4 and recovered.exists():
                    missing_recovered.append(dict(stage=stage, id=job['id'], recovery=str(recovered)))
                    continue
                raise FileNotFoundError('Unresolved allocation: ' + str(result_path))
            result = readj(result_path)
            assert all(result[k] == v for k, v in job.items())
            reason = (result.get('result') or {}).get('reason', '')
            technical = bool(result.get('error') or (result.get('metrics') or {}).get('metric_error')
                             or reason.startswith(('policy_error:', 'simulation_error:', 'perception_error:')))
            record['technical'] = technical
            if technical:
                continue
            start = saved_start(job, stage, path)
            record['start'] = start
            # Exact normalization from frozen generator line36, retaining list order.
            key = ('exp23' if stage == 'exp23_recovery' else stage,
                   job['experiment'], job['scene_id'], job['seed'])
            starts[key].append(start)

    pairs = []
    legacy_pairs = []

    def compare(a, b, stage, experiment, scene, seed, route):
        aj, bj = a['job'], b['job']
        obs_equal = a['obs'] == b['obs']
        row = dict(audit_index=len(pairs) + 1, stage=stage, experiment=experiment,
                   scene=scene, seed=seed, config_A=aj['config_name'], config_B=bj['config_name'],
                   id_A=aj['id'], id_B=bj['id'], source_stage_A=a['stage'], source_stage_B=b['stage'],
                   scope='smoke' if stage == 'smoke' else 'optional_probe' if stage == 'jump_vs_exp1' else 'formal',
                   gpu_A=aj['gpu'], gpu_B=bj['gpu'], same_gpu=aj['gpu'] == bj['gpu'],
                   state_equal=bool(np.array_equal(a['state'], b['state'])),
                   observation_hash_equal=bool(obs_equal),
                   first_action_block_equal=bool(np.array_equal(a['actions'], b['actions'])),
                   first_action_block_max_abs_diff=float(np.max(abs(a['actions'] - b['actions']))),
                   config_B_report='F_jump_limited' if stage == 'jump_vs_exp1' else bj['config_name'],
                   generation_route=route,
                   chunk_A=str(a['chunk']), chunk_A_sha256=digest(a['chunk']),
                   chunk_B=str(b['chunk']), chunk_B_sha256=digest(b['chunk']),
                   lineage_A=str(a['lineage']), lineage_A_sha256=digest(a['lineage']),
                   lineage_B=str(b['lineage']), lineage_B_sha256=digest(b['lineage']),
                   state_A_array_sha256=array_digest(a['state']), state_B_array_sha256=array_digest(b['state']),
                   action_A_array_sha256=array_digest(a['actions']), action_B_array_sha256=array_digest(b['actions']),
                   observation_A_sha256=hashlib.sha256(json.dumps(a['obs'], sort_keys=True).encode()).hexdigest(),
                   observation_B_sha256=hashlib.sha256(json.dumps(b['obs'], sort_keys=True).encode()).hexdigest(),
                   state_A_shape=list(a['state'].shape), state_B_shape=list(b['state'].shape),
                   actions_A_shape=list(a['actions'].shape), actions_B_shape=list(b['actions'].shape),
                   observation_hash_keys=sorted(a['obs']))
        pairs.append(row)
        legacy = dict(stage=stage, experiment=experiment, scene=scene, seed=seed,
                      state_equal=row['state_equal'], first_action_block_equal=row['first_action_block_equal'],
                      first_action_block_max_abs_diff=row['first_action_block_max_abs_diff'])
        if stage != 'jump_vs_exp1':
            legacy.update(reference=aj['id'], candidate=bj['id'], same_gpu=row['same_gpu'],
                          observation_hash_equal=row['observation_hash_equal'])
        legacy_pairs.append(legacy)

    group_audit = []
    for key, rows in starts.items():
        reference = rows[0]
        group_audit.append(dict(stage=key[0], experiment=key[1], scene=key[2], seed=key[3],
                                configurations=[r['job']['config_name'] for r in rows],
                                reference=reference['job']['config_name'], contrasts=len(rows) - 1))
        for candidate in rows[1:]:
            compare(reference, candidate, *key, route='frozen lines39-41: first reference versus remaining')
    # Exact extra loop from frozen lines43-45, plus explicit observation digest audit.
    for record in [r for r in allocations if r['stage'] == 'jump' and r['exists'] and not r.get('technical')]:
        job = record['job']
        ref_id = f"exp1_F_{job['scene_id']}_r{job['rep']:03d}"
        reference = next(r['start'] for r in allocations if r['stage'] == 'exp1' and r['job']['id'] == ref_id)
        compare(reference, record['start'], 'jump_vs_exp1', 'exp1_jump', job['scene_id'], job['seed'],
                route='frozen lines43-45: probe versus original Exp1 F; observation hash newly verified')

    frozen = readj(source / 'audit/final_allocation_integrity.json')
    execution = readj(source / 'execution_summary.json')
    assert legacy_pairs == frozen['initial_block_contrasts'], 'Reproduced ordered list differs from frozen generator output'
    assert len(pairs) == execution['initial_block_contrasts'] == 237
    assert all(r['same_gpu'] and r['state_equal'] and r['observation_hash_equal']
               and r['first_action_block_equal'] for r in pairs)
    agg = collections.defaultdict(list)
    for row in pairs:
        agg[(row['scope'], row['stage'], row['experiment'], row['scene'], row['config_A'], row['config_B'])].append(row)
    aggregate = [dict(scope=k[0], stage=k[1], experiment=k[2], scene=k[3], config_A=k[4], config_B=k[5],
                      contrasts=len(v), same_gpu_all=all(r['same_gpu'] for r in v),
                      state_equal_all=all(r['state_equal'] for r in v),
                      observation_hash_equal_all=all(r['observation_hash_equal'] for r in v),
                      first_action_block_equal_all=all(r['first_action_block_equal'] for r in v),
                      seeds=[r['seed'] for r in v]) for k, v in sorted(agg.items())]
    scope_counts = dict(collections.Counter(r['scope'] for r in pairs))
    experiment_counts = dict(collections.Counter(r['experiment'] for r in pairs if r['scope'] == 'formal'))
    generator = source / 'scripts/audit_final_allocation.py'
    digest(generator)
    digest(__file__)
    word = args.word.resolve(strict=True)
    digest(word)
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
          'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
    with zipfile.ZipFile(word) as archive:
        body = etree.fromstring(archive.read('word/document.xml')).find('w:body', ns)
    mentions = []
    for i, element in enumerate(body):
        text = ''.join(element.xpath('.//w:t/text()|.//m:t/text()', namespaces=ns))
        if '237' in text:
            mentions.append(dict(body_index=i, text=text))
    assert len(mentions) == 1
    appendix = ('首请求一致性共核查237项配置对照，包括正式实验230项（触限诊断30项、方向对照100项、F/B消融100项）、冒烟2项，以及限跳探针对原F配置的5项；组内按首配置对其余配置逐项比较，限跳探针另与原F比较。'
                '保存的本体状态数组、首观测摘要与首动作块均一致，但237不是独立回合数，也不代表后续闭环轨迹严格配对。')
    result = dict(status='PASS_REPRODUCED_237_WITH_SCOPE_CORRECTION', source_root=str(source),
                  generator=dict(path=str(generator), function='module-level loops; no named comparison function',
                                 collection_lines=[16, 36], group_pair_lines=[38, 41],
                                 optional_probe_pair_lines=[42, 45], output_lines=[46, 52]),
                  count=len(pairs), scope_counts=scope_counts, formal_experiment_counts=experiment_counts,
                  count_formula='230 formal (30 Exp1 +100 Exp2 +100 Exp3) +2 smoke +5 probe =237',
                  frozen_ordered_pair_list_exact_match=True, all237_state_observation_and_block_equal=True,
                  historical_observation_hash_field_present=232, historical_probe_observation_hash_field_missing=5,
                  current_probe_observation_hashes_newly_verified=5,
                  all_same_gpu=True, groups=group_audit, aggregate=aggregate,
                  excluded_missing_original_resolved_by_recovery=missing_recovered,
                  third_round_word=str(word), third_round_word_237_mentions=mentions,
                  word_scope_assessment='237 claim is true after current observation audit, but includes2smoke and5probe and counts comparisons not independent groups/episodes; clarify scope.',
                  appendix_replacement=appendix, pairs=pairs, source_sha256=source_hashes,
                  physics_steps=0, policy_calls=0, frozen_files_modified=False,
                  notes=['Four configurations in Exp1 contribute3 reference comparisons per seed, not all6 unordered combinations.',
                         'Recovery normalization puts seed860049 back into exp23; B is first reference and recoveredF candidate for that one pair.',
                         'The original5 probe pair records did not contain observation_hash_equal; this audit adds saved-lineage verification.',
                         'execution_summary wrapper assignment script was not located among packaged Python files; actual list generator and exact237 reproduction are recorded.'])
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    def writecsv(path, rows):
        with path.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()} for row in rows)
    writecsv(paths['csv'], pairs)
    writecsv(paths['aggregate'], aggregate)
    writecsv(paths['sources'], [dict(path=k, sha256=v) for k, v in sorted(source_hashes.items())])
    with paths['json'].open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    lines = ['# 第三轮任务2：237项首次输入／动作块对照的实际来源', '',
             '**结论：实际复算237项，与冻结清单顺序和各历史字段完全一致；237包含230项正式配置对照、2项冒烟对照、5项限跳探针对原F对照。它不是237个独立回合，也不是仅正式440回合的对照数。**', '',
             '## 真实生成代码与逐层计算', '',
             f'生成清单的冻结代码为`{generator}`，SHA256 `{digest(generator)}`；比较循环位于模块顶层，没有命名函数。16–36行按smoke、exp1、jump、exp23、exp23_recovery读取分配、实际result、首动作块和policy_lineage；36行把恢复批重新归入exp23。38–41行以分组首配置为参考，比较其余配置；42–45行额外把5个限跳探针对应到原Exp1 F；46行保存清单，52行输出其长度。',
             'execution_summary.json字段237与冻结final_allocation_integrity.json清单长度一致。本次按该代码的原始阶段和分配顺序复算，逐条匹配了原清单；在封包Python中未发现另一个将此长度写入execution_summary的包装脚本，因此不虚构其函数或行号。', '',
             '|范围|实验／场景|实际比较方式|对照项数|', '|---|---|---|---|']
    by_exp_scene = collections.Counter((r['scope'], r['experiment'], r['scene']) for r in pairs)
    for key, count in by_exp_scene.items():
        route = '首配置对其余配置' if key[0] != 'optional_probe' else '限跳对原Exp1 F'
        lines.append(f'|{key[0]}|{key[1]}／{key[2]}|{route}|{count}|')
    lines += ['', '正式Exp1每场景5个种子，每种子4配置，按实际分配顺序的首配置对其余3配置逐项比较；领先配置随预定执行顺序轮换，而非固定raw。因此每场景15项、两场景30项，而不是四配置全部两两组合的60项。Exp2固定／法向每场景50项，共100项；Exp3 F/B每场景50项，共100项。',
              '6个冒烟回合中，Exp1的raw与F分属不同场景／种子，不形成对照；只有Exp2与Exp3各1项，均在X＋40 mm场景。限跳5回合在自身阶段每组只有一个配置，不在组内循环产生项数，而是由额外循环与原Exp1 F构成5项。',
              '原技术中断分配seed860049的缺失旧result不进入首块列表；恢复结果被归回exp23，仍只形成一个Exp3 X＋40 mm对照。该种子由于B先进入分组，比较方向为B→恢复F，其余99个Exp3对照为F→B；不新增实验样本。', '',
              '## 数组与摘要的一致性核查', '',
              '237项的本体state数组、首块actions数组及已保存的首观测摘要字典均逐项相同，动作块最大绝对差均为0，GPU也逐项相同。CSV提供每项实验、场景、种子、配置A/B、三项一致标记、原NPZ与lineage路径和SHA256；机器JSON还保留数组shape及含dtype/shape的摘要。摘要只针对保存的首请求，不能据此推断后续状态、动作或整个闭环严格配对。',
              '重要范围修正：原冻结清单前232项包含observation_hash_equal字段；5项jump_vs_exp1只保存state和首块比较，未保存首观测比较字段。本次实际读取这5项双方policy_lineage第一行补核，首观测摘要也全部相同；这是本轮新增的只读证据，不能倒写成旧清单原已保存该字段。', '',
              '## 第三轮Word附录C核对', '',
              f'只读稿件：`{word}`，SHA256 `{digest(word)}`。实际body_index {mentions[0]["body_index"]}写作：', '', mentions[0]['text'], '',
              '这句话没有注明237包含2项冒烟和5项探针，并把计数写为“组”，容易被理解为237组独立正式实验。建议改为以下两句；不改动Word：', '', appendix, '',
              '## 输出与复跑', '',
              f'- 逐项CSV：`{paths["csv"]}`', f'- 聚合CSV：`{paths["aggregate"]}`',
              f'- 机器证据：`{paths["json"]}`', f'- 来源SHA256：`{paths["sources"]}`',
              f'- 本轮只读脚本：`{Path(__file__).resolve()}`；指定一个本轮根目录内尚无这些输出的子目录可复跑，现有输出均拒绝覆盖。',
              '- 未运行闭环、MuJoCo或策略推理，未修改第二轮冻结包或第三轮Word。', '']
    with paths['report'].open('x') as stream:
        stream.write('\n'.join(lines))
    print(json.dumps(dict(status=result['status'], count=len(pairs), scope_counts=scope_counts,
                          formal_experiment_counts=experiment_counts, outputs={k:str(v) for k,v in paths.items()}), ensure_ascii=False))


if __name__ == '__main__':
    main()
