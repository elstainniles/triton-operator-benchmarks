"""Strict per-launch MSPTI device timing for direct Triton calls."""
from collections import defaultdict
import math
import random
import statistics


def measure(calls, synchronize, monitor_factory, prepare=None):
    """Measure (variant, callable, exact device kernel name) entries in three rounds.

    Each synchronized launch gets its own monitor window, so callbacks can be
    attributed without assuming that different launches produce the same number
    of device tasks. Reject extra, missing, or unexpected kernels.
    """
    samples = defaultdict(lambda: defaultdict(list))
    rng = random.Random(20261026)
    for round_id in range(3):
        order = list(calls)
        rng.shuffle(order)
        for variant, launch, kernel_name in order:
            for repeat in range(5):
                if prepare is not None:
                    prepare()
                synchronize()
                records = []
                monitor = monitor_factory()
                def callback(data):
                    records.append((data.name, data.start, data.end))
                monitor.start(callback)
                try:
                    launch()
                    synchronize()
                finally:
                    monitor.stop()
                if len(records) != 1 or records[0][0] != kernel_name:
                    raise ValueError(f'Expected exactly one {kernel_name} kernel, observed {records!r}')
                duration = (records[0][2] - records[0][1]) / 1000.0
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError('Invalid MSPTI kernel duration')
                samples[variant][round_id].append(dict(repeat=repeat, duration_us=duration))
    variants = {}
    for variant, _, kernel_name in calls:
        rounds = samples[variant]
        medians = [dict(round=r, median_us=statistics.median(x['duration_us'] for x in values))
                   for r, values in sorted(rounds.items())]
        variants[variant] = dict(kernel_name=kernel_name,
            samples=[dict(round=r, **x) for r, values in sorted(rounds.items()) for x in values],
            rounds=medians, median_us=statistics.median(x['median_us'] for x in medians))
    return dict(method='mspti', variants=variants, correlation_verified=True)


def measure_npu(calls, prepare=None):
    import torch
    try:
        from mspti import KernelMonitor
    except ImportError as exc:
        raise RuntimeError('MSPTI timing requires the mspti Python package in the runtime') from exc
    return measure(calls, torch.npu.synchronize, KernelMonitor, prepare)
