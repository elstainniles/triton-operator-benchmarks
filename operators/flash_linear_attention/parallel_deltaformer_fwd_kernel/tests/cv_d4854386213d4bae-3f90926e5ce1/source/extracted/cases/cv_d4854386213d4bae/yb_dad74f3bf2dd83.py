
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_a87c34e2ef2580 import parallel_deltaformer_fwd_kernel
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
ACTIVE_ITERS = 20
WAIT_ITERS = 1
SCHEDULE_WARMUP = 1
SCHEDULE_SKIP_FIRST = 1
TOTAL_ITERS = SCHEDULE_SKIP_FIRST + WAIT_ITERS + SCHEDULE_WARMUP + ACTIVE_ITERS
experimental_config = torch_npu.profiler._ExperimentalConfig(aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False)

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
def test_parallel_deltaformer_fwd_kernel_performance():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_deltaformer_fwd_kernel'
    try:
        data = load_fixture(os.path.join(load_dir, 'test_parallel_deltaformer_fwd_kernel_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f'load file failed: {str(e)}')
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    result_dir = os.path.dirname(os.path.abspath(__file__))
    trace_dir = os.path.join(result_dir, 'prof_dir_parallel_deltaformer_fwd_kernel')

    def run_kernel():
        parallel_deltaformer_fwd_kernel[data['grid']](input_data['q'], input_data['k'], input_data['v'], input_data['u'], input_data['w'], input_data['lse'], input_data['beta'], input_data['H'], input_data['T'], input_data['C'], input_data['D'], input_data['qk_scale'])
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU], with_stack=False, record_shapes=False, profile_memory=False, schedule=torch_npu.profiler.schedule(wait=WAIT_ITERS, warmup=SCHEDULE_WARMUP, active=ACTIVE_ITERS, repeat=1, skip_first=SCHEDULE_SKIP_FIRST), experimental_config=experimental_config, on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_dir)) as prof:
        for _ in range(TOTAL_ITERS):
            run_kernel()
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    print(f'[parallel_deltaformer_fwd_kernel] Profiling completed, trace saved to {trace_dir}')
