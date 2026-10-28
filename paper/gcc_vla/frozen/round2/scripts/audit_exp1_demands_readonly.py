#!/usr/bin/env python3
"""Read recorded Exp1 arrays and event CSV only; never run MuJoCo or inference.

Writes a new JSON file or a JSON in a new directory. Existing output paths are
rejected. The demand/first-limit records reproduce the original ad-hoc audit;
the extra aggregates make every interpretation table independently inspectable.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counter_sum(rows, field):
    result = collections.Counter()
    for row in rows:
        result.update(row[field])
    return dict(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'audit/analysis_exp1')
    parser.add_argument('--output', type=Path, required=True,
                        help='New .json file, or new directory; never overwrite')
    args = parser.parse_args()
    requested = args.output.resolve()
    if requested.exists():
        raise FileExistsError('Refusing existing output path: ' + str(requested))
    output = requested if requested.suffix.lower() == '.json' else requested / 'exp1_demands_and_events.json'
    if output.exists():
        raise FileExistsError('Refusing overwrite: ' + str(output))
    analysis = args.analysis_dir.resolve()
    data = json.loads((analysis / 'analysis_round2.json').read_text())
    rows = data['per_episode'] + data['optional_probe_per_episode']
    out = {'n': len(rows), 'files': []}
    for row in rows:
        with np.load(row['trace_path'], allow_pickle=False) as z:
            meta = json.loads(str(z['metadata_json']))
            idx = meta['arm_actuator_indices']
            valid = np.asarray(meta['affine_demand_valid'])[idx]
            done = z['step_completed']
            demand = z['actuator_demand_arm']
            force = z['actuator_force_arm']
            sat = z['actuator_at_limit_arm'] & done[:, None]
            ranges = np.asarray(meta['actuator_forcerange'])[idx]
            clipped = np.clip(demand, ranges[None, :, 0], ranges[None, :, 1])
            difference = np.abs(clipped - force)[done]
            where = np.flatnonzero(sat.any(axis=1))
            first = int(where[0]) if len(where) else None
            result = dict(
                id=row['id'], config=row['config'], scene=row['scene'],
                affine_valid_all6=bool(valid.all()), forcerange_arm=ranges.tolist(),
                max_abs_clip_demand_force_difference_Nm=float(difference.max()),
                limited_joint_substeps=int(sat.sum()),
                demand_exceeds_and_limited_joint_substeps=int((sat & z['actuator_demand_exceeds_arm']).sum()),
                first=None)
            if first is not None:
                limited_joints = np.flatnonzero(sat[first])
                err = z['position_error_before'][first]
                jump = z['delta_u_from_previous'][first]
                dem = demand[first]
                result['first'] = dict(
                    relative_s=float(z['time_before_s'][first] - row['trace_start_s']),
                    joints=(limited_joints + 1).tolist(),
                    max_abs_position_error_limited_joint_rad=float(np.max(abs(err[limited_joints]))),
                    error_max_limited_joint=int(limited_joints[np.argmax(abs(err[limited_joints]))] + 1),
                    max_abs_target_jump_limited_joint_rad=float(np.max(abs(jump[limited_joints]))),
                    jump_max_limited_joint=int(limited_joints[np.argmax(abs(jump[limited_joints]))] + 1),
                    max_abs_position_error_all6_rad=float(np.max(abs(err))),
                    error_max_all6_joint=int(np.argmax(abs(err)) + 1),
                    max_abs_target_jump_all6_rad=float(np.max(abs(jump))),
                    jump_max_all6_joint=int(np.argmax(abs(jump)) + 1),
                    max_abs_demand_limited_joint_Nm=float(np.max(abs(dem[limited_joints]))),
                    demand_max_limited_joint=int(limited_joints[np.argmax(abs(dem[limited_joints]))] + 1))
            out['files'].append(result)
    out['summary'] = dict(
        affine_valid_all45_sixarm=all(r['affine_valid_all6'] for r in out['files']),
        max_abs_clip_demand_force_difference_Nm=max(r['max_abs_clip_demand_force_difference_Nm'] for r in out['files']),
        all_limited_demands_exceed_limit=all(r['limited_joint_substeps'] == r['demand_exceeds_and_limited_joint_substeps'] for r in out['files']))

    events = collections.defaultdict(collections.Counter)
    with gzip.open(analysis / 'exp1_saturation_events.csv.gz', 'rt') as stream:
        for row in csv.DictReader(stream):
            key = (row['stage'], row['config'], row['scene'])
            count = events[key]
            count['saturated_substeps'] += 1
            for field in ['geometry_delta_rad', 'contact_delta_rad', 'guard_delta_rad',
                          'target_jump_rad', 'position_error_rad']:
                values = json.loads(row[field])
                count[field + '_nonzero_substeps'] += any(abs(x) > 1e-12 for x in values)
            contacts = json.loads(row['contacts'])
            count['protected_external_substeps'] += any(c['protected_external'] for c in contacts)
            count['pillar_protected_external_substeps'] += any(
                c['protected_external'] and 'pillar' in c['roles'] for c in contacts)
    out['event_aggregates'] = [dict(stage=k[0], config=k[1], scene=k[2], **v)
                               for k, v in sorted(events.items())]

    groups = []
    for scene in ['main', 'pillar_x_plus40mm']:
        for config in ['raw', 'G', 'F', 'oscbf', 'F_jump_limited']:
            selected = [r for r in rows if r['config'] == config and r['scene'] == scene]
            if not selected:
                continue
            first_rows = [r for r in selected if r['tau_limit_episode']]
            peaks = collections.Counter()
            for row in selected:
                pair = row['peak_max_contribution_external_pair']
                peaks.update(['+'.join(pair['roles']) if pair else 'none'])
            first_stats = [r['first'] for r in out['files'] if r['config'] == config
                           and r['scene'] == scene and r['first']]
            ranges = {}
            for field in ['max_abs_position_error_limited_joint_rad',
                          'max_abs_target_jump_limited_joint_rad',
                          'max_abs_demand_limited_joint_Nm']:
                values = [r[field] for r in first_stats]
                ranges[field] = [min(values), max(values)] if values else None
            groups.append(dict(
                scene=scene, config=config, n=len(selected),
                SR=sum(r['SR'] for r in selected), STSR=sum(r['STSR'] for r in selected),
                PCR=sum(r['PCR'] for r in selected), limit_episodes=len(first_rows),
                first_limit_phase_counts=dict(collections.Counter(r['first_limit_phase'] for r in first_rows)),
                first_limit_relative_s_range=[min(r['first_limit_relative_s'] for r in first_rows),
                                              max(r['first_limit_relative_s'] for r in first_rows)] if first_rows else None,
                limited_per_joint=[sum(r['tau_limit_per_joint'][i] for r in selected) for i in range(6)],
                limited_joint_substeps=sum(r['tau_limit_joint_substeps'] for r in selected),
                dominant_associations=counter_sum(selected, 'saturated_joint_dominant_association_counts'),
                saturated_substep_phase_counts=counter_sum(selected, 'saturated_substep_phase_counts'),
                saturated_substep_contact_category_counts=counter_sum(selected, 'saturated_substep_contact_category_counts'),
                first_limit_external_contact_episodes=sum(any(c['protected_external'] for c in r['first_limit_contacts']) for r in first_rows),
                peak_max_contribution_pair_episode_counts=dict(peaks), first_limit_ranges=ranges,
                first_error_max_joint_counts=dict(collections.Counter(r['error_max_limited_joint'] for r in first_stats)),
                first_jump_max_joint_counts=dict(collections.Counter(r['jump_max_limited_joint'] for r in first_stats))))
    out['group_aggregates'] = groups
    out['optional_probe_descriptive_groups'] = data['optional_probe_descriptive_groups']
    out['sources'] = {str(analysis / name): sha(analysis / name) for name in [
        'analysis_round2.json', 'exp1_saturation_events.csv.gz', 'exp1_diagnostics_flat.csv',
        'optional_probe_descriptive_groups.csv']}
    out['script_sha256'] = sha(__file__)
    out['notes'] = [
        'Only completed recorded 1ms steps; no simulator import or physical replay.',
        'Dominant module counts use saturated joint times substep, not causal attribution.',
        'Current zero contact delta does not exclude historical controller effects.',
        'First-limit errors/jumps are maximized among currently limited arm joints.',
        'Probe is separate from formal samples.']
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(out, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(output)


if __name__ == '__main__':
    main()
