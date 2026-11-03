
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
ACTIVE_ITERS = 20
WAIT_ITERS = 1
SCHEDULE_WARMUP = 1
SCHEDULE_SKIP_FIRST = 1
TOTAL_ITERS = SCHEDULE_SKIP_FIRST + WAIT_ITERS + SCHEDULE_WARMUP + ACTIVE_ITERS
experimental_config = torch_npu.profiler._ExperimentalConfig(aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False)
from yb_5deca3efb6cf0b import prepare_wy_repr_bwd_kernel

def convert_tensor_with_device_type(indata: dict, device_type: str):
    target_device = torch.device(device_type)
    outdata = {}
    for key, value in indata.items():
        if isinstance(value, torch.Tensor):
            if value.device.type != target_device.type:
                outdata[key] = value.to(target_device)
            else:
                outdata[key] = value
        else:
            outdata[key] = value
    return outdata

@pytest.mark.perftest
def test_prepare_wy_repr_bwd_kernel_performance():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/prepare_wy_repr_bwd_kernel/'
    try:
        data = load_fixture(os.path.join(load_dir, 'test_prepare_wy_repr_bwd_kernel_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f'load file failed: {str(e)}')
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    result_dir = os.path.dirname(os.path.abspath(__file__))
    trace_dir = os.path.join(result_dir, 'prof_dir_prepare_wy_repr_bwd_kernel')

    def run_kernel():
        prepare_wy_repr_bwd_kernel[data['grid']](k=input_data['k'], v=input_data['v'], p=input_data['p'], beta=input_data['beta'], g0=input_data['g0'], g=input_data['g'], A=input_data['A'], dw=input_data['dw'], du=input_data['du'], dk=input_data['dk'], dv=input_data['dv'], dp=input_data['dp'], dbeta=input_data['dbeta'], dg0=input_data['dg0'], dg=input_data['dg'], cu_seqlens=input_data['cu_seqlens'], chunk_indices=input_data['chunk_indices'], T=input_data['T'], H=input_data['H'], K=input_data['K'], V=input_data['V'], BT=input_data['BT'], BK=input_data['BK'] // 2, BV=input_data['BV'])
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU], with_stack=False, record_shapes=False, profile_memory=False, schedule=torch_npu.profiler.schedule(wait=WAIT_ITERS, warmup=SCHEDULE_WARMUP, active=ACTIVE_ITERS, repeat=1, skip_first=SCHEDULE_SKIP_FIRST), experimental_config=experimental_config, on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_dir)) as prof:
        for _ in range(TOTAL_ITERS):
            run_kernel()
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    print(f'[prepare_wy_repr_bwd_kernel] Profiling completed, trace saved to {trace_dir}')
