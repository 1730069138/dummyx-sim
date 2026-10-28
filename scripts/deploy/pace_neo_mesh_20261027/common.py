"""Paths, immutable evidence and write boundaries for the independent benchmark."""
from pathlib import Path
import datetime
import hashlib
import json
import math
import numbers
import shutil

ROOT = Path('/home/jun/dummyx-sim')
SCRIPTS = ROOT / 'scripts/deploy/pace_external50'
WORK = ROOT / 'outputs/pace_neo_mesh_20261027'
BENCHMARK = ROOT / 'outputs/pace_external50/benchmark_20261027_062049'
SEALED = ROOT / 'outputs/pace_gc_directional/research_20261026_214604'
ORIGINAL_PROTOCOL = ROOT / 'outputs/screwdriver_eval/paper_full50_20261026/protocol.json'
PORTS = {0: 8001, 1: 8002, 2: 8000, 3: 8003, 4: 8004, 5: 8005}
METHODS = ('raw_pi0', 'pace_g', 'pace_gc', 'vlsa', 'oscbf', 'neo')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def new_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(WORK.resolve()):
        raise ValueError('Output must stay inside the new independent benchmark: ' + str(path))
    return path


def normalize_nonfinite(value, path='$'):
    """Return JSON-compatible values plus paths of missing nonfinite numbers.

    Missing timing estimates remain null, never numerical zero. This also
    handles NumPy numeric scalars/arrays without importing NumPy into helpers.
    """
    missing = []
    if value is None or isinstance(value, (str, bool)):
        return value, missing
    if isinstance(value, numbers.Integral):
        return int(value), missing
    if isinstance(value, numbers.Real):
        number = float(value)
        return (number, missing) if math.isfinite(number) else (None, [path])
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            clean[key], paths = normalize_nonfinite(item, path + '.' + str(key))
            missing.extend(paths)
        return clean, missing
    if isinstance(value, (list, tuple)):
        clean = []
        for index, item in enumerate(value):
            result, paths = normalize_nonfinite(item, f'{path}[{index}]')
            clean.append(result)
            missing.extend(paths)
        return clean, missing
    if type(value).__module__.startswith('numpy') and hasattr(value, 'tolist'):
        return normalize_nonfinite(value.tolist(), path)
    return value, missing


def write_new_json(path, value):
    path = new_output(path)
    clean, missing = normalize_nonfinite(value)
    if missing and isinstance(clean, dict):
        clean['serialization_nonfinite_fields'] = sorted(set(clean.get('serialization_nonfinite_fields', []) + missing))
    # Serialization must finish before exclusive creation: an encoder failure
    # cannot leave a partial result.json blocking the immutable output path.
    encoded = json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False, default=str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(encoded)
    # Workers append the same entry to their ledger immediately afterward. Keep
    # that in-memory entry consistent with the strict JSON receipt just written.
    if missing and isinstance(value, dict):
        value.clear()
        value.update(clean)
    elif missing and isinstance(value, list):
        value[:] = clean
    return clean


def backup_before_edit(path):
    """Save current bytes and an independent verification copy before revision."""
    path = Path(path).resolve(strict=True)
    if not (path.is_relative_to(SCRIPTS.resolve()) or path.is_relative_to(Path(__file__).resolve().parent)):
        new_output(path)
    stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S_%f')
    base = WORK / 'evidence/revisions' / stamp
    relative = path.relative_to(ROOT)
    backup = base / 'backup' / relative
    working = base / 'verification_copy' / relative
    for destination in (backup, working):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    hashes = [sha(p) for p in (path, backup, working)]
    assert len(set(hashes)) == 1
    write_new_json(base / 'three_way_sha256.json', dict(
        source=str(path), backup=str(backup), verification_copy=str(working),
        source_sha256=hashes[0], backup_sha256=hashes[1], working_sha256=hashes[2], captured_utc=now()))
    return backup


def protect_original():
    manifest = json.loads((BENCHMARK / 'STAGE_MANIFEST.json').read_text())
    assert sha(ORIGINAL_PROTOCOL) == manifest['source_protocol_sha256'], 'Original protocol changed'
    receipt = Path(manifest['backup_receipt'])
    assert sha(receipt) == manifest['backup_receipt_sha256']
    for item in json.loads(receipt.read_text()):
        assert sha(item['source']) == item['source_sha256'], ('Original changed', item['source'])
    return True


def verify_freeze(run):
    run = new_output(run)
    protocol = json.loads((run / 'protocol.json').read_text())
    freeze = json.loads((run / 'freeze.json').read_text())
    assert sha(run / 'protocol.json') == freeze['protocol_sha256'], 'Protocol changed'
    for path, expected in protocol['new_source_hashes'].items():
        assert sha(path) == expected, ('Frozen source changed', path)
    for key in ('checkpoint_identity', 'service_metadata', 'statistics_plan', 'probe_receipt',
                'scene_protocol', 'original_protocol', 'pilot_eligibility'):
        if protocol.get(key):
            assert sha(protocol[key]) == protocol[key + '_sha256'], ('Frozen evidence changed', key)
    protect_original()
    return protocol
