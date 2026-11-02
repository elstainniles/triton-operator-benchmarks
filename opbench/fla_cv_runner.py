"""Serial FLA CV workbook batch runner with independent CPU references."""
import csv
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

from .catalog import ROOT, read_json
from .environment import audit, configure
from .fla_cv import GROUP, load_case, load_kernel, selected, sha, verify_case
from .runner import compile_error, save


def reference(case, seed, folder, timeout):
    """Run CPU semantics in a fresh interpreter, never in the NPU JIT process."""
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / 'cpu-reference'
    env = dict(os.environ, TRITON_INTERPRET='1', ASCEND_RT_VISIBLE_DEVICES='-1',
               OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4')
    command = [sys.executable,'-I',str(ROOT/'tools/run_fla_cv.py'),'_reference',
               '--cases',str(case['row']),'--seed',str(seed),'--output',str(dest)]
    proc = subprocess.run(command,env=env,capture_output=True,text=True,timeout=timeout)
    (folder/'cpu-reference.log').write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8')
    if proc.returncode:
        raise RuntimeError('CPU reference failed: '+proc.stderr[-1600:])
    info = read_json(dest/'reference.json')
    if info['module_sha256'] != case['module_sha256'] or info['source_sha256'] != case['source_sha256']:
        raise RuntimeError('CPU reference source changed')
    if hashlib.sha256((dest/'reference.pt').read_bytes()).hexdigest() != info['data_sha256']:
        raise RuntimeError('CPU reference payload changed')
    return dest,info


def run_one(args):
    """Run exactly one row in a fresh process, cache and card-lock scope."""
    import fcntl
    rows=selected(args.cases,args.operator)
    if len(rows)!=1 or not rows[0].get('operator_id'):
        raise ValueError('Worker requires one source-backed CV row')
    row=rows[0]
    path,case=load_case(row)
    verify_case(path,case)
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    result=dict(schema_version=1,row=row['row'],case=row['id'],operator_id=row['operator_id'],
                status='STARTING',baseline_us=None,candidate_us=None,
                source_sha256=case['source_sha256'],module_sha256=case['module_sha256'],
                seeds=list(case['seeds']),factor=args.factor,axis=args.axis,physical_card=args.physical_card,
                accuracy=[],candidates=[],timing=[],started_epoch=time.time())
    save(output/'result.json',result)
    try:
        expected=read_json(args.environment)
        configure(expected,output,args.physical_card)
        os.environ['TRITON_KERNEL_DUMP']='1'
        for key,name in (('TRITON_HOME','triton-home'),('XDG_CACHE_HOME','xdg'),
                         ('TORCH_EXTENSIONS_DIR','torch-extensions')):
            target=output/'cache'/name
            target.mkdir()
            os.environ[key]=str(target)
        lockroot=Path('/tmp')/f'triton-operator-benchmarks-{os.getuid()}'
        lockroot.mkdir(mode=0o700,exist_ok=True)
        if lockroot.is_symlink() or lockroot.stat().st_uid!=os.getuid():
            raise RuntimeError('Invalid per-user lock root')
        lock_paths=[p for p in (args.task_lock,args.card_lock) if p]
        lock_paths.append(lockroot/f'card-{args.physical_card}.lock')
        if len({p.resolve() for p in lock_paths})!=len(lock_paths):
            raise ValueError('Task and card lock paths must differ')
        with ExitStack() as stack:
            for lock_path in lock_paths:
                lock=stack.enter_context(lock_path.open('a'))
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            import torch
            import torch_npu
            from .device import Monitor,identity
            from .fla_cv_reference import arguments,compare,grid,make_inputs
            from .fla_cv_timing import measure
            from triton.backends.ascend.block_batching_guard_v3 import validate_plan_binding
            result['environment']=audit(expected)
            torch.set_num_threads(4)
            if torch.npu.device_count()!=1:
                raise RuntimeError('Expected exactly one explicitly visible NPU')
            torch.npu.set_device(0)
            kernel=load_kernel(path,case)
            fn=getattr(kernel,case['symbol'])
            launch_grid=grid(case,kernel)
            result['grid']=launch_grid
            owner=identity()
            with Monitor(args.physical_card,owner,output/'device.jsonl') as monitor:
                monitor.require_clear()
                references={seed:reference(case,seed,output/f'reference-{seed}',args.reference_timeout)
                            for seed in case['seeds']}
                baseline_options=dict(case['config'],enable_hivm_batch_matmul=True)
                variants=[]
                for seed in case['seeds']:
                    folder,meta=references[seed]
                    cpu=make_inputs(case,seed)
                    if {k:hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest() for k,v in cpu.items()}!=meta['input_sha256']:
                        raise RuntimeError('Reference inputs do not match NPU inputs')
                    ref=torch.load(folder/'reference.pt',weights_only=True)
                    if seed==case['seeds'][0]:
                        device={}
                        guards={}
                        for name,tensor in cpu.items():
                            # torch.bool guards are initialized to True and checked as booleans.
                            pad=torch.full((tensor.numel()+64,),True if tensor.dtype==torch.bool else 13,
                                           dtype=tensor.dtype,device='npu')
                            view=pad[32:-32].view(tensor.shape)
                            view.copy_(tensor)
                            device[name]=view
                            guards[name]=pad
                        initial={k:v.clone() for k,v in device.items()}
                        first_cpu=cpu
                    else:
                        for name,tensor in cpu.items():
                            device[name].copy_(tensor)
                            initial[name].copy_(tensor)
                    def prepare(device=device,initial=initial,writes=ref['writes']):
                        for name in writes:
                            device[name].copy_(initial[name])
                    def check(device=device,expected=ref['expected'],writes=ref['writes'],guards=guards):
                        torch.npu.synchronize()
                        checks=compare(device,expected,writes)
                        for name,pad in guards.items():
                            value=True if pad.dtype==torch.bool else 13
                            checks.append(dict(name=name+'_guard',passed=bool((pad[:32]==value).all().item()
                                                                      and (pad[-32:]==value).all().item())))
                        return checks
                    values=arguments(case,device)
                    if seed==case['seeds'][0]:
                        result['status']='BASELINE_COMPILE'
                        save(output/'result.json',result)
                        compiled=fn.warmup(**values,grid=launch_grid,**baseline_options)
                        result['baseline_binary_sha256']=hashlib.sha256(compiled.kernel).hexdigest()
                        result['baseline_kernel_name']=compiled.name
                        result['baseline_compiler_hash']=compiled.hash
                    def launch_base(values=values):return fn[launch_grid](**values,**baseline_options)
                    result['status']='BASELINE_RUNTIME'
                    prepare();launch_base()
                    checks=check()
                    result['accuracy'].append(dict(seed=seed,variant='baseline',checks=checks))
                    save(output/'result.json',result)
                    if not all(c['passed'] for c in checks):
                        result['status']='BASELINE_ACCURACY_FAILED'
                        return result
                    if seed==case['seeds'][0]:
                        variants=[dict(id='baseline',name=compiled.name,hash=compiled.hash,
                                       prepare=prepare,launch=launch_base)]
                        if args.factor>1:
                            for axis,extent in enumerate(launch_grid):
                                if extent<args.factor or (args.axis!='all' and int(args.axis)!=axis):
                                    continue
                                options=dict(baseline_options,block_batching_cv=True,
                                             block_batching_factor=args.factor,block_batching_axis=axis)
                                candidate=dict(factor=args.factor,axis=axis,status='COMPILING')
                                result['candidates'].append(candidate)
                                try:
                                    choice=fn.warmup(**values,grid=launch_grid,**options)
                                    candidate['report']=choice.block_batching
                                    prepare();got=fn[launch_grid](**values,**options)
                                    candidate['launch']=dict(got.last_launch)
                                    if got.last_launch.get('runtime_status')!='APPLIED':
                                        candidate['status']='FALLBACK'
                                        continue
                                    artifact=got.candidate
                                    guard=got.block_batching['guard']
                                    if not validate_plan_binding(guard) or artifact.metadata.block_batching_plan_sha256!=guard['plan_fingerprint']:
                                        raise RuntimeError('Unbound candidate plan')
                                    checks=check()
                                    candidate['checks']=checks
                                    candidate['status']='CORRECT_APPLIED' if all(c['passed'] for c in checks) else 'CANDIDATE_ACCURACY_FAILED'
                                    if candidate['status']=='CORRECT_APPLIED':
                                        candidate['binary_sha256']=hashlib.sha256(artifact.kernel).hexdigest()
                                        def launch_candidate(options=options,values=values):
                                            value=fn[launch_grid](**values,**options)
                                            if value.last_launch['runtime_status']!='APPLIED':
                                                raise RuntimeError('Candidate fell back during timing')
                                            return value.candidate
                                        variants.append(dict(id=f'f{args.factor}_a{axis}',name=artifact.name,
                                                             hash=artifact.hash,prepare=prepare,launch=launch_candidate))
                                except Exception:
                                    candidate.update(status='CANDIDATE_COMPILE_OR_RUNTIME_FAILED',
                                                     error=traceback.format_exc())
                                finally:save(output/'result.json',result)
                    else:
                        for variant in variants[1:]:
                            candidate=next(c for c in result['candidates'] if f'f{args.factor}_a{c["axis"]}'==variant['id'])
                            options=dict(baseline_options,block_batching_cv=True,
                                         block_batching_factor=args.factor,block_batching_axis=candidate['axis'])
                            prepare()
                            launched=fn[launch_grid](**values,**options)
                            if launched.last_launch.get('runtime_status')!='APPLIED':
                                result['status']='CANDIDATE_FALLBACK_SECOND_SEED'
                                save(output/'result.json',result)
                                return result
                            checks=check()
                            result['accuracy'].append(dict(seed=seed,variant=variant['id'],checks=checks))
                            if not all(c['passed'] for c in checks):
                                result['status']='CANDIDATE_ACCURACY_FAILED'
                                save(output/'result.json',result)
                                return result
                    if seed==case['seeds'][-1]:
                        # Reuse the same allocation for both seeds, then time seed 1.
                        for name,tensor in first_cpu.items():
                            device[name].copy_(tensor)
                            initial[name].copy_(tensor)
                if args.correctness_only:
                    result['status']='CORRECT'
                    return result
                result['status']='TIMING'
                save(output/'result.json',result)
                sample_start=len(monitor.samples)-1
                for round_index in range(2):
                    reading=measure(variants,output/f'timing-{round_index}',round_index)
                    result['timing'].append(reading)
                    save(output/'result.json',result)
                monitor.sample()
                if not monitor.eligible_since(sample_start):
                    raise RuntimeError('Device monitor rejected timing')
                result['baseline_us']=statistics.mean(x['variants']['baseline']['us'] for x in result['timing'])
                for variant in variants[1:]:
                    measured=statistics.mean(x['variants'][variant['id']]['us'] for x in result['timing'])
                    candidate=next(c for c in result['candidates'] if f'f{args.factor}_a{c["axis"]}'==variant['id'])
                    candidate['kernel_us']=measured
                times=[c['kernel_us'] for c in result['candidates'] if c.get('kernel_us')]
                if times:result['candidate_us']=min(times)
                result['status']='MEASURED'
                return result
    except Exception as error:
        if result['status']=='BASELINE_COMPILE' and compile_error(error):
            result['status']='BASELINE_COMPILE_FAILED'
            result['error']=traceback.format_exc()
            return result
        if result['status'] not in ('BASELINE_ACCURACY_FAILED','CANDIDATE_ACCURACY_FAILED'):
            result['status']='FAILED_REVIEW_REQUIRED'
        result['error']=traceback.format_exc()
        raise
    finally:
        result['ended_epoch']=time.time()
        save(output/'result.json',result)


def run_batch(args):
    rows=selected(args.cases,args.operator)
    output=args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError('Output exists; use --resume to continue only unfinished rows')
    if args.resume and not (output/'summary.json').exists():
        raise FileNotFoundError('Resume requires an existing batch summary')
    output.mkdir(parents=True,exist_ok=True)
    summary_path=output/'summary.json'
    config=dict(physical_card=args.physical_card,factor=args.factor,axis=args.axis,
                include_control_flow=args.include_control_flow,
                correctness_only=args.correctness_only,
                environment=str(args.environment.resolve()),
                environment_sha256=sha(args.environment) if args.environment.is_file() else None,
                task_lock=str(args.task_lock.resolve()) if args.task_lock else None,
                card_lock=str(args.card_lock.resolve()) if args.card_lock else None,
                reference_timeout=args.reference_timeout,worker_timeout=args.worker_timeout,
                manifest_sha256=sha(GROUP/'cv-manifest.json'))
    if summary_path.exists():
        summary=read_json(summary_path)
        if not args.resume or summary['selection']!=[r['row'] for r in rows]:
            raise ValueError('Resume requires the same frozen selection')
        if summary.get('run_config')!=config:
            raise ValueError('Resume requires the original environment, locks, card and run options')
    else:
        summary=dict(schema_version=1,completed=False,selection=[r['row'] for r in rows],
                     run_config=config,
                     rows={},started_epoch=time.time())
    try:
        for row in rows:
            key=str(row['row'])
            if key in summary['rows']:
                continue
            if row['historical_status']=='SOURCE_NOT_FOUND':
                status='SOURCE_NOT_FOUND'
            elif row['historical_status']=='CONTROL_FLOW_BLOCKED' and not args.include_control_flow:
                status='CONTROL_FLOW_BLOCKED_PRECHECK_ONLY'
            else:
                status=None
            if status:
                summary['rows'][key]=dict(status=status,baseline_us=None,candidate_us=None,npu_executed=False)
            else:
                dest=output/row['id']
                if dest.exists():
                    results=sorted(dest.glob('attempt-*/result.json'))
                    if not results:
                        raise RuntimeError('Existing incomplete attempt must be reviewed: '+str(dest))
                    latest=read_json(results[-1])
                    if latest['status'] not in ('MEASURED','CORRECT','BASELINE_COMPILE_FAILED',
                                                'BASELINE_ACCURACY_FAILED','CANDIDATE_ACCURACY_FAILED'):
                        raise RuntimeError('Existing failed attempt needs review: '+str(results[-1]))
                    record=latest
                    result_path=results[-1]
                else:
                    attempt=dest/'attempt-001'
                    command=[sys.executable,'-I',str(ROOT/'tools/run_fla_cv.py'),'_worker',
                             '--cases',key,'--physical-card',str(args.physical_card),
                             '--environment',str(args.environment.resolve()),'--output',str(attempt),
                             '--factor',str(args.factor),'--axis',args.axis,
                             '--reference-timeout',str(args.reference_timeout)]
                    if args.correctness_only:command.append('--correctness-only')
                    if args.task_lock:command+=['--task-lock',str(args.task_lock.resolve())]
                    if args.card_lock:command+=['--card-lock',str(args.card_lock.resolve())]
                    dest.mkdir(parents=True)
                    with (dest/'worker.log').open('w',encoding='utf-8') as log:
                        process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=args.worker_timeout)
                    if not (attempt/'result.json').exists():
                        raise RuntimeError('Worker produced no result; review '+str(dest/'worker.log'))
                    record=read_json(attempt/'result.json')
                    if process.returncode or record['status']=='FAILED_REVIEW_REQUIRED':
                        raise RuntimeError('Worker failed; review '+str(dest/'worker.log'))
                    result_path=attempt/'result.json'
                summary['rows'][key]=dict(status=record['status'],baseline_us=record['baseline_us'],
                                          candidate_us=record['candidate_us'],npu_executed=True,
                                          result=str(result_path))
            save(summary_path,summary)
            print(json.dumps(dict(row=row['row'],**summary['rows'][key]),ensure_ascii=False),flush=True)
        summary['completed']=True
    finally:
        summary['ended_epoch']=time.time()
        save(summary_path,summary)
        with (output/'performance.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=['row','kernel','operator_id','status','baseline_us','candidate_us','npu_executed','result'])
            writer.writeheader()
            for row in rows:
                value=summary['rows'].get(str(row['row']),{})
                writer.writerow(dict(row=row['row'],kernel=row['kernel'],operator_id=row['operator_id'],
                                     status=value.get('status','PENDING'),baseline_us=value.get('baseline_us'),
                                     candidate_us=value.get('candidate_us'),npu_executed=value.get('npu_executed',False),
                                     result=value.get('result','')))
