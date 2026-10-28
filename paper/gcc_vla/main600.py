#!/usr/bin/env python3
"""Original six-method entry; optional fixed-count scheduling, unchanged controllers."""
from pathlib import Path
import hashlib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manage import entry_module

def main():
    entry = entry_module()
    fixed = '--fixed-count' in sys.argv
    if fixed:
        sys.argv.remove('--fixed-count')
        original = entry.make_plan
        def make_plan(args):
            plan = original(args)
            if 'protocol' in plan:
                plan['protocol']['first_batch_gate'] = False
            plan['fixed_episode_count'] = True
            plan['paper_dispatch_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            return plan
        entry.make_plan = make_plan
    entry.main()

if __name__ == '__main__':
    main()
