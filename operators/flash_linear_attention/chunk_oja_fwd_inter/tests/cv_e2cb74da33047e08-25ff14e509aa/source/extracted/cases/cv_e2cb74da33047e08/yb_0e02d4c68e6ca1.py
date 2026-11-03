
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
import triton
from yb_8b1f9f64c8c620 import chunk_oja_fwd_inter
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

def make_grid(data):
    input_data = data['input_data']
    T = input_data['T']
    B = input_data['q'].shape[0]
    HQ = input_data['q'].shape[2]
    V = input_data['gv'].shape[-1]
    BT = min(64, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT)

    def grid(meta):
        return (triton.cdiv(V, meta['BV']), NT, B * HQ)
    return grid

@pytest.mark.perftest
def test_chunk_oja_fwd_inter_kernel_performance():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_oja_fwd_inter/'
    try:
        data = load_fixture(os.path.join(load_dir, 'test_chunk_oja_fwd_inter_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f'load file failed: {str(e)}')
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    result_dir = os.path.dirname(os.path.abspath(__file__))
    trace_dir = os.path.join(result_dir, 'prof_dir_chunk_oja_fwd_inter_kernel')
    grid = make_grid(data)

    def run_kernel():
        chunk_oja_fwd_inter[grid](**input_data)
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU], with_stack=False, record_shapes=False, profile_memory=False, schedule=torch_npu.profiler.schedule(wait=WAIT_ITERS, warmup=SCHEDULE_WARMUP, active=ACTIVE_ITERS, repeat=1, skip_first=SCHEDULE_SKIP_FIRST), experimental_config=experimental_config, on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_dir)) as prof:
        for _ in range(TOTAL_ITERS):
            run_kernel()
            torch.npu.synchronize()
            prof.step()
        prof.stop()
    print(f'[chunk_oja_fwd_inter_kernel] Profiling completed, trace saved to {trace_dir}')
