"""Two 15-sample profiler rounds, attributed to exact compiled Triton kernels."""
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import statistics
import threading

from .device import identity, Monitor


def measure(variants, output, round_index):
    import torch
    import torch_npu
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    selected=list(variants)
    if round_index % 2:selected.reverse()
    labels={}
    owner=identity()
    with Monitor(int(__import__('os').environ['ASCEND_RT_VISIBLE_DEVICES']), owner,
                 output/'device.jsonl') as monitor:
        monitor.require_clear()
        stop=threading.Event()
        thread=threading.Thread(target=lambda: _sample_frequently(monitor,stop),daemon=True)
        thread.start()
        exp=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone)
        try:
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0,warmup=1,active=3,repeat=1),
                record_shapes=False,profile_memory=False,with_stack=False,
                experimental_config=exp,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output/'profile'))) as profiler:
                for step in range(4):
                    for index in range(5):
                        for variant in selected:
                            variant['prepare']()
                            torch.npu.synchronize()
                            if step==0:
                                variant['launch']()
                            else:
                                label=f'fla_cv::{variant["id"]}::{(step-1)*5+index}'
                                labels[label]=dict(id=variant['id'],name=variant['name'],
                                                   hash=variant['hash'],sample=(step-1)*5+index)
                                with torch.profiler.record_function(label):
                                    got=variant['launch']()
                                    if got.hash!=variant['hash']:
                                        raise RuntimeError('Compiled kernel hash changed during timing')
                    torch.npu.synchronize()
                    profiler.step()
        finally:
            stop.set()
            thread.join(timeout=25)
            if thread.is_alive():
                raise RuntimeError('Fast occupancy sampler did not stop')
        monitor.sample()
        if not monitor.eligible_since(0):
            raise RuntimeError('Device occupancy interfered with timing')
    traces=list((output/'profile').rglob('trace_view.json'))
    if len(traces)!=1:
        raise RuntimeError('Expected one profiler trace')
    trace=traces[0]
    raw=json.loads(trace.read_text(),parse_float=Decimal)
    events=raw['traceEvents'] if isinstance(raw,dict) else raw
    spans=[e for e in events if e.get('ph')=='X' and e.get('name','').startswith('fla_cv::')]
    if len(spans)!=len(labels) or {e['name'] for e in spans}!=set(labels):
        raise RuntimeError('Incomplete or duplicate CPU sample labels')
    with trace.with_name('kernel_details.csv').open(newline='') as stream:
        kernels=list(csv.DictReader(stream))
    kernel_by_start={Decimal(k['Start Time(us)'].strip()):k for k in kernels}
    if len(kernel_by_start)!=len(kernels):
        raise RuntimeError('Ambiguous device kernel timestamps')
    starts={e['id']:e for e in events if e.get('cat')=='async_npu' and e.get('ph')=='s'}
    ends={e['id']:e for e in events if e.get('cat')=='async_npu' and e.get('ph')=='f'}
    observed={key:[] for key in labels}
    for flow,start in starts.items():
        matches=[e for e in spans if e['pid']==start['pid'] and
                 Decimal(e['ts'])<=Decimal(start['ts'])<Decimal(e['ts'])+Decimal(e['dur'])]
        if not matches:
            continue
        if len(matches)!=1 or flow not in ends:
            raise RuntimeError('Ambiguous profiler flow')
        timestamp=Decimal(ends[flow]['ts'])
        if timestamp not in kernel_by_start:
            raise RuntimeError('Missing device kernel detail')
        observed[matches[0]['name']].append(kernel_by_start[timestamp])
    samples=[]
    for label,expected in labels.items():
        found=observed[label]
        if len(found)!=1:
            raise RuntimeError('Expected one device kernel for '+label)
        name=found[0]['Name']
        intended=expected['name']
        if name!=intended and not (len(intended)>47 and name==intended[-47:]):
            raise RuntimeError('Unexpected device kernel for '+label)
        samples.append(dict(variant=expected['id'],sample=expected['sample'],name=name,
                            us=float(Decimal(found[0]['Duration(us)']))))
    times={}
    for variant in variants:
        data=[x['us'] for x in samples if x['variant']==variant['id']]
        if len(data)!=15 or any(value<=0 for value in data):
            raise RuntimeError('Incomplete or invalid device samples')
        times[variant['id']]=dict(us=statistics.median(data),samples_us=data)
    return dict(round=round_index,variants=times,samples=samples,trace=str(trace),
                trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                metric='attributed Triton device kernel us')


def _sample_frequently(monitor,stop):
    while not stop.wait(.25):
        monitor.sample()
