import argparse
import json
from pathlib import Path

from .catalog import plans, resolve, select


def main():
    parser = argparse.ArgumentParser(description='Independent Triton operator benchmarks')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list').add_argument('--operator', default='all')
    for name in ('plan', 'run'):
        command = commands.add_parser(name)
        command.add_argument('--operator', required=True)
        command.add_argument('--suite', choices=['smoke', 'fp32', 'bf16', 'all'], default='all')
        command.add_argument('--variants', choices=['baseline', 'manual', 'all'], default='all')
        if name == 'run':
            command.add_argument('--output', required=True, type=Path)
            command.add_argument('--physical-card', type=int, choices=range(8))
            command.add_argument('--environment', type=Path)
            command.add_argument('--correctness-only', action='store_true')
            command.add_argument('--case-id', action='append', default=[], help='Generated graph or FLA case ID; repeatable')
    args = parser.parse_args()
    if args.command == 'list':
        print(json.dumps(select(args.operator), ensure_ascii=False, indent=2))
    elif args.command == 'plan':
        print(json.dumps(plans(args.operator, args.suite, args.variants), ensure_ascii=False, indent=2))
    else:
        selected = select(args.operator)
        executable = any(e.get('runner') != 'fla' or e.get('entry_mode') != 'source_archive' for e in selected)
        if executable and (args.physical_card is None or args.environment is None):
            parser.error('Executable entries require --physical-card and --environment')
        if args.operator in ('all', 'graph', 'flaggems', 'liger', 'fla') or ',' in args.operator:
            from .batch import run
            return run(args)
        entry, _ = resolve(args.operator)
        if entry.get('runner') == 'fla':
            from .fla_runner import run
        elif entry.get('runner') == 'liger':
            from .liger_runner import run
        elif entry.get('runner') == 'graph_tests':
            from .graph_runner import run
        else:
            if args.case_id:
                parser.error('--case-id is available for generated graph cases only')
            from .runner import run
        return run(args)
