"""CPU-only runner checks; no accelerator runtime is initialized."""
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from opbench import fla_test_runner as runner

try:
    import torch
except ImportError:
    torch = None


class BlockEvidence(unittest.TestCase):
    def report(self):
        return dict(runtime_status='APPLIED',owner='block_batching',requested_factor=2,
                    applied_factor=2,selected_axis=0,original_grid=[5,2,1],batch_grid=[3,2,1],plan_id='p')

    def test_actual_branch_runtime_schema_accepted(self):
        self.assertEqual(runner.block_status(self.report(),2,0,(5,2)), 'applied')

    def test_fallback_compile_reject_and_mismatched_grid_not_applied(self):
        report=self.report();report['runtime_status']='BASELINE_SELECTED'
        self.assertEqual(runner.block_status(report,2,0,(5,2)), 'runtime_fallback')
        report=self.report();report['batch_grid']=[2,2,1]
        self.assertEqual(runner.block_status(report,2,0,(5,2)), 'block_runtime_evidence_mismatch')
        self.assertEqual(runner.block_status(None,2,0,(5,2),{'compile_status':'REJECTED'}), 'compile_rejected')
        self.assertEqual(runner.block_status({'status':'APPLIED'},2,0,(5,2)), 'block_not_confirmed_applied')

    def test_candidate_runtime_name_selected(self):
        c=SimpleNamespace(last_launch=self.report(),packed_metadata={'kernel_name':'baseline'},
                          candidate=SimpleNamespace(packed_metadata={'kernel_name':'merged'}))
        self.assertEqual(runner.runtime_kernel_name(c,'target'), 'merged')
        c.last_launch={'runtime_status':'BASELINE_SELECTED'}
        self.assertEqual(runner.runtime_kernel_name(c,'target'), 'baseline')

    def test_environment_failure_still_saves_result(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);info=root/'test.json';info.write_text(json.dumps(dict(id='t',case_id='c',kernel='k')))
            env=root/'env.json';env.write_text('{}');out=root/'out'
            with patch.object(runner,'ROOT',root),patch.object(runner,'select',return_value=[{'path':'test.json'}]), \
                    patch('opbench.environment.configure',side_effect=RuntimeError('wrong runtime')),redirect_stdout(io.StringIO()):
                code=runner.main(['--case','t','--physical-card','1','--environment',str(env),'--output',str(out)])
            self.assertEqual(code,1)
            result=json.loads((out/'result.json').read_text())
            self.assertEqual(result['status'],'runtime_failed')
            self.assertIn('wrong runtime',result['error'])
            self.assertIsNone(result['timing'])


@unittest.skipIf(torch is None, 'CPU execution checks require PyTorch')
class CpuExecution(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);(self.root/'generated').mkdir();(self.root/'source').mkdir()
        (self.root/'generated/fixture.py').write_text('def make_fixtures():\n import torch\n return {"input.pt":torch.arange(4,dtype=torch.float32)}\n')
        self.test=self.root/'source/cpu_test.py'
        self.spec=dict(provider='generated/fixture.py',pythonpath=['source'],kernel='target_kernel',
                       module='cpu_test',function='test_case',invocation={'args':[],'kwargs':{}},autotune_config={},
                       files=[dict(path='source/cpu_test.py',role='test')])
        self.args=SimpleNamespace(correctness_only=True,block_factor=None,axis=None,output=self.root/'out',samples=5,warmup=1)
        self.result={'timing':None}
        class Jit:
            def __init__(self, fn):self.fn=fn;self.arg_names=['x','y']
            def run(self,*values,grid,warmup,**kwargs):
                if not warmup:self.fn(*values)
                base=SimpleNamespace(packed_metadata={'kernel_name':'target_kernel'})
                if kwargs.get('block_batching_factor'):
                    factor=kwargs['block_batching_factor'];axis=kwargs['block_batching_axis']
                    original=list(grid)+[1]*(3-len(grid));batch=list(original);batch[axis]=(batch[axis]+factor-1)//factor
                    base.candidate=SimpleNamespace(packed_metadata={'kernel_name':'merged'})
                    base.last_launch=dict(runtime_status='APPLIED',owner='block_batching',requested_factor=factor,
                        applied_factor=factor,selected_axis=axis,original_grid=original,batch_grid=batch,plan_id='cpu-fixture')
                    base.block_batching={'compile_status':'COMPILED_GUARDED'}
                return base
        class Tune:
            def __init__(self,fn):self.fn=fn;self.configs=[None,None]
            def run(self,*a,**kw):
                assert len(self.configs)==1, 'fixed autotune selection was not applied'
                config=self.configs[0]
                return self.fn.run(*a,**dict(kw,**config.kwargs,**config.options))
            def _bench(self,*a,**kw):pass
        self.jit=Jit
        triton=ModuleType('triton');triton.runtime=SimpleNamespace(jit=SimpleNamespace(JITFunction=Jit))
        triton.Config=lambda tile,**options:SimpleNamespace(kwargs=tile,options=options)
        tune=ModuleType('triton.runtime.autotuner');tune.Autotuner=Tune
        backend=ModuleType('fake_backend');backend.Jit=Jit
        npu=ModuleType('torch_npu');npu.get_npu_format=lambda t:2
        self.modules={'triton':triton,'triton.runtime.autotuner':tune,'torch_npu':npu,'fake_backend':backend}

    def write(self, body):
        self.test.write_text('import torch\nfrom fake_backend import Jit\nfrom opbench.fla_fixtures import load_fixture\n'
            'def target_kernel(x,y): y.copy_(x+1)\nkernel=Jit(target_kernel)\ndef test_case():\n'
            ' x=load_fixture("input.pt");y=torch.zeros_like(x)\n'+body)

    def execute(self):
        with patch.dict(sys.modules,self.modules),patch.object(torch,'npu',SimpleNamespace(synchronize=lambda:None)):
            runner.execute(self.root,self.spec,self.args,self.result)

    def test_numeric_acceptance_passes_and_modules_are_cleaned(self):
        self.write(' kernel.run(x,y,grid=(2,),warmup=False)\n torch.testing.assert_close(y,x+1)\n')
        before=list(sys.path);self.execute()
        self.assertEqual(self.result['status'],'correct')
        self.assertEqual(self.result['correctness']['status'],'passed')
        self.assertEqual(sys.path,before);self.assertNotIn('cpu_test',sys.modules)

    def test_no_numeric_assertion_cannot_pass(self):
        self.write(' kernel.run(x,y,grid=(2,),warmup=False)\n assert y.shape==x.shape\n')
        self.execute();self.assertEqual(self.result['status'],'accuracy_unvalidated')

    def test_recorded_autotune_tile_and_options_are_preserved(self):
        self.spec['autotune_config']={'kwargs':{'BT':16},'num_warps':32,'num_stages':2}
        self.write(' kernel.run(x,y,grid=(2,),warmup=False)\n torch.testing.assert_close(y,x+1)\n')
        text=self.test.read_text().replace('kernel=Jit(target_kernel)',
            'from triton.runtime.autotuner import Autotuner\nkernel=Autotuner(Jit(target_kernel))')
        self.test.write_text(text);self.execute()
        self.assertEqual(self.result['status'],'correct')
        self.assertEqual(self.result['launch_options']['BT'],16)
        self.assertEqual(self.result['launch_options']['num_warps'],32)
        self.assertEqual(self.result['launch_options']['num_stages'],2)

    def test_matching_original_autotune_config_object_is_retained(self):
        self.spec['autotune_config']={'kwargs':{'BT':16},'num_warps':32,'num_stages':2}
        self.write(' kernel.run(x,y,grid=(2,),warmup=False)\n assert kernel.configs[0] is original_config\n torch.testing.assert_close(y,x+1)\n')
        setup=('from triton.runtime.autotuner import Autotuner\nfrom types import SimpleNamespace\n'
               'kernel=Autotuner(Jit(target_kernel))\n'
               'original_config=SimpleNamespace(kwargs={"BT":16},num_warps=32,num_stages=2,'
               'options={"num_warps":32,"num_stages":2},pre_hook=lambda x: None)\n'
               'kernel.configs=[original_config]')
        self.test.write_text(self.test.read_text().replace('kernel=Jit(target_kernel)',setup))
        self.execute();self.assertEqual(self.result['status'],'correct')

    def test_wrong_numerical_result_retains_failure(self):
        self.write(' kernel.run(x,y,grid=(2,),warmup=False)\n torch.testing.assert_close(y,x+2)\n')
        with self.assertRaises(AssertionError):self.execute()
        self.assertEqual(self.result['correctness']['status'],'failed')

    def test_distinct_target_configurations_not_reported_as_one(self):
        self.write(' for grid in ((2,),(4,)):\n  kernel.run(x,y,grid=grid,warmup=False)\n  torch.testing.assert_close(y,x+1)\n')
        self.execute();self.assertEqual(self.result['status'],'multiple_target_configurations')
        self.assertIsNone(self.result['timing'])

    def test_block_applied_uses_real_runtime_fields(self):
        self.args.block_factor=2;self.args.axis=0
        self.write(' kernel.run(x,y,grid=(5,2),warmup=False)\n torch.testing.assert_close(y,x+1)\n')
        self.execute();self.assertEqual(self.result['status'],'correct')
        self.assertEqual(self.result['last_launch']['batch_grid'],[3,2,1])

    def test_device_samples_are_target_only_mean_and_restore_outside_profile(self):
        active={'value':False};counter={'i':0}
        class Snapshot:
            checks=0
            def restore_pre(self):assert not active['value']
            def verify_post(self):assert not active['value'];self.checks+=1
            def restore_post(self):assert not active['value']
        class Profile:
            def __init__(self,**kw):self.folder=Path(kw['on_trace_ready'])
            def __enter__(self):active['value']=True;return self
            def step(self):pass
            def __exit__(self,*exc):
                active['value']=False;counter['i']+=1;self.folder.mkdir(parents=True)
                with (self.folder/'kernel_details.csv').open('w') as f:
                    f.write('Name,Duration(us)\ntarget,'+str(counter['i'])+'\nother_kernel,100000\n')
        npu=ModuleType('torch_npu');npu.profiler=SimpleNamespace(
            profile=Profile,ProfilerActivity=SimpleNamespace(CPU=1,NPU=2),tensorboard_trace_handler=lambda x:x)
        capture=dict(snapshot=Snapshot(),jit=SimpleNamespace(run=lambda *a,**k:None),args=(),kwargs={},grid=(1,),name='target')
        with patch.dict(sys.modules,{'torch_npu':npu}),patch.object(torch,'npu',SimpleNamespace(synchronize=lambda:None)):
            result=runner.measure(capture,self.root/'out',samples=5,warmup=2)
        self.assertEqual(result['samples_us'],[1,2,3,4,5]);self.assertEqual(result['mean_us'],3)
        self.assertEqual(result['state_restore_checks'],7)


if __name__=='__main__':unittest.main()
