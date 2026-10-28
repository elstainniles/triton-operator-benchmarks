"""Attribute exact NPU profiler rows to direct Triton calls; reject ambiguity."""
from collections import defaultdict
import csv
from decimal import Decimal
import json
from pathlib import Path
import statistics


def correlate(events, rows, expected_labels, kernel_name, allow_unlabelled_kernels=False):
    expected_labels = set(expected_labels)
    labels = [e for e in events if e.get('ph') == 'X' and e.get('name', '').startswith('opbench::')]
    if len(labels) != len(expected_labels) or {e['name'] for e in labels} != expected_labels:
        raise ValueError('Missing, duplicate or unexpected benchmark labels')
    flows = {}
    for phase in ('s', 'f'):
        selected = [e for e in events if e.get('cat') == 'async_npu' and e.get('ph') == phase]
        flows[phase] = {e['id']: e for e in selected}
        if len(flows[phase]) != len(selected):
            raise ValueError('Duplicate flow IDs')
    if flows['s'].keys() != flows['f'].keys():
        raise ValueError('Unpaired CPU/NPU flow')
    by_time = {Decimal(r['Start Time(us)'].strip()): r for r in rows}
    if len(by_time) != len(rows):
        raise ValueError('Duplicate kernel start timestamps')
    seen, matched, nonkernel, excluded = set(), defaultdict(list), [], []
    for flow_id, start in flows['s'].items():
        end_time = Decimal(flows['f'][flow_id]['ts'])
        t = Decimal(start['ts'])
        containing = [label for label in labels
                      if (label['pid'], label['tid']) == (start['pid'], start['tid'])
                      and Decimal(label['ts']) <= t < Decimal(label['ts']) + Decimal(label['dur'])]
        if end_time not in by_time:
            if containing:
                raise ValueError('Non-kernel device task inside a direct Triton call')
            nonkernel.append(str(flow_id))
            continue
        if not containing and allow_unlabelled_kernels:
            excluded.append(dict(name=by_time[end_time]['Name'], start_us=str(end_time)))
            seen.add(end_time)
            continue
        if len(containing) != 1:
            raise ValueError('Kernel flow has ambiguous or missing benchmark label')
        if end_time in seen:
            raise ValueError('Kernel assigned twice')
        seen.add(end_time)
        row = by_time[end_time]
        if row['Name'] != kernel_name:
            raise ValueError(f'Unexpected kernel: {row["Name"]}; expected {kernel_name}')
        duration = Decimal(row['Duration(us)'].strip())
        if not duration.is_finite() or duration <= 0:
            raise ValueError('Invalid kernel duration')
        matched[containing[0]['name']].append(float(duration))
    if seen != set(by_time):
        raise ValueError('Unattributed kernel rows')
    if set(matched) != expected_labels or any(len(v) != 1 for v in matched.values()):
        raise ValueError('Each expected call must map to exactly one Triton kernel')
    grouped = defaultdict(lambda: defaultdict(list))
    for label, durations in matched.items():
        _, vid, round_id, repeat = label.split('::')
        grouped[vid][int(round_id)].append(dict(repeat=int(repeat), duration_us=durations[0]))
    variants = {}
    for vid, rounds in grouped.items():
        medians = [dict(round=r, median_us=statistics.median(c['duration_us'] for c in calls))
                   for r, calls in sorted(rounds.items())]
        variants[vid] = dict(kernel_name=kernel_name,
                             samples=[dict(round=r, **call) for r, calls in sorted(rounds.items())
                                      for call in sorted(calls, key=lambda c: c['repeat'])],
                             rounds=medians,
                             median_us=statistics.median(r['median_us'] for r in medians))
    return dict(variants=variants, excluded_unlabelled_nonkernel_flows=nonkernel,
                excluded_unlabelled_kernels=excluded,
                correlation_verified=True)


def analyze(directory, expected_labels, kernel_name, allow_unlabelled_kernels=False):
    traces = list(Path(directory).rglob('trace_view.json'))
    if len(traces) != 1:
        raise ValueError(f'Expected one profiler trace, found {len(traces)}')
    trace = json.loads(traces[0].read_text(encoding='utf-8'), parse_float=Decimal)
    events = trace['traceEvents'] if isinstance(trace, dict) else trace
    with traces[0].with_name('kernel_details.csv').open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    return correlate(events, rows, expected_labels, kernel_name, allow_unlabelled_kernels)
