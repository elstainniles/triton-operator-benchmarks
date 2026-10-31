"""Serial subprocess dispatch: each operator gets a fresh audited runtime/cache."""
from collections import Counter
import csv
from datetime import datetime, timezone
import subprocess
import sys

from .catalog import ROOT, select, plans, read_json
from .runner import save


def run(args):
    if args.case_id:
        raise ValueError('--case-id requires a single generated operator')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(completed=False, all_passed=False,
                   started_utc=datetime.now(timezone.utc).isoformat(), operators=[])
    rows = []
    try:
        for entry in select(args.operator):
            schedule = plans(entry['id'], args.suite, args.variants)
            if not schedule:
                continue
            folder = output / entry['id']
            cmd = [sys.executable, '-I', str(ROOT/'tools/run.py'), 'run', '--operator', entry['id'],
                   '--suite', args.suite, '--variants', args.variants,
                   '--output', str(folder)]
            cmd.extend(['--timing', getattr(args, 'timing', 'profiler')])
            if args.physical_card is not None:
                cmd.extend(['--physical-card', str(args.physical_card)])
            if args.environment is not None:
                cmd.extend(['--environment', str(args.environment.resolve())])
            if args.correctness_only:
                cmd.append('--correctness-only')
            with (output / (entry['id']+'.log')).open('w', encoding='utf-8') as log:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            data = read_json(folder/'result.json') if (folder/'result.json').exists() else {}
            first_row = len(rows)
            for record in data.get('cases', []):
                case = record.get('case') or {}
                for variant in record.get('variants', [record]):
                    rows.append(dict(operator=entry['name'], case=case.get('id'),
                                     shape=str(case['shape']) if 'shape' in case else None, dtype=case.get('dtype'),
                                     type=entry['type'], variant=variant.get('variant_id', 'baseline'),
                                     status=variant['status'], kernel_us=variant.get('median_us')))
            counts = dict(Counter(row['status'] for row in rows[first_row:]))
            summary['operators'].append(dict(id=entry['id'], exit_code=proc.returncode,
                completed=data.get('completed', False), counts=counts,
                all_passed=bool(counts) and set(counts) <= {'correct','measured'}))
            summary['counts'] = dict(Counter(r['status'] for r in rows))
            save(output/'summary.json', summary)
            print(entry['id'], summary['operators'][-1], flush=True)
            # Runtime/device faults must be inspected before dispatching another NPU process.
            if proc.returncode:
                raise RuntimeError(f'{entry["id"]} stopped; see its log and result.json')
        summary['completed'] = True
        summary['all_passed'] = bool(summary['operators']) and all(item['all_passed'] for item in summary['operators'])
    finally:
        summary['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(output/'summary.json', summary)
        with (output/'performance.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['operator','case','shape','dtype','type','variant','status','kernel_us'])
            writer.writeheader()
            writer.writerows(rows)
