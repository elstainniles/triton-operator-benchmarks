import argparse
import json
from pathlib import Path

from .catalog import operators, plans, resolve


def main():
    parser = argparse.ArgumentParser(description='Independent Triton operator benchmarks')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list')
    for name in ('plan', 'run'):
        command = commands.add_parser(name)
        command.add_argument('--operator', required=True)
        command.add_argument('--suite', choices=['fp32', 'bf16', 'all'], default='all')
        command.add_argument('--variants', choices=['baseline', 'manual', 'all'], default='all')
        if name == 'run':
            command.add_argument('--output', required=True, type=Path)
            command.add_argument('--physical-card', required=True, type=int, choices=range(8))
            command.add_argument('--environment', required=True, type=Path)
            command.add_argument('--correctness-only', action='store_true')
    args = parser.parse_args()
    if args.command == 'list':
        print(json.dumps(operators(), ensure_ascii=False, indent=2))
    elif args.command == 'plan':
        print(json.dumps(plans(args.operator, args.suite, args.variants), ensure_ascii=False, indent=2))
    else:
        entry, _ = resolve(args.operator)
        if entry.get('runner') == 'graph_tests':
            parser.error('TG suites use tools/run_graph_tests.py; see docs/GRAPH_OPTIMIZATION_TESTS.md')
        from .runner import run
        run(args)
