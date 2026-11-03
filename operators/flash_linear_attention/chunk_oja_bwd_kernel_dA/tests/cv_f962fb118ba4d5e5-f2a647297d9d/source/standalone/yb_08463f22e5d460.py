
from opbench.fla_fixtures import load_fixture
import os
import time
import pytest
import torch
import torch_npu
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
DEVICE = f'npu:{torch.npu.current_device()}'
B, T, H, D = (2, 512, 4, 64)
kernel_name = 'chunk_oja_bwd_kernel_dA'

def get_precision_thresholds(dtype):
    return (0.008, 0.008)

@pytest.mark.functiontest
def test_chunk_oja_bwd_kernel_dA_simple_function():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_oja_bwd_kernel_dA'
    from yb_8b1f9f64c8c620 import chunk_oja_bwd_dA
    v = load_fixture(os.path.join(load_dir, 'v.pt')).to(DEVICE)
    gv = load_fixture(os.path.join(load_dir, 'gv.pt')).to(DEVICE)
    do = load_fixture(os.path.join(load_dir, 'do.pt')).to(DEVICE)
    scale = load_fixture(os.path.join(load_dir, 'scale.pt')).item()
    dA = chunk_oja_bwd_dA(v=v, gv=gv, do=do, scale=scale)
    if isinstance(dA, tuple):
        for i, out in enumerate(dA):
            expected_out = load_fixture(os.path.join(load_dir, f'output_{i}.pt')).to(DEVICE)
            atol, rtol = get_precision_thresholds(out.dtype)
            torch.testing.assert_close(out, expected_out, atol=atol, rtol=rtol)
            print(f'Output {i} precision check passed')
    else:
        expected_output = load_fixture(os.path.join(load_dir, 'output.pt')).to(DEVICE)
        atol, rtol = get_precision_thresholds(dA.dtype)
        torch.testing.assert_close(dA, expected_output, atol=atol, rtol=rtol)
        print('Output precision check passed')
    print(f'[{kernel_name}] Precision check completed successfully')

@pytest.mark.perftest
def test_chunk_oja_bwd_kernel_dA_simple_performance():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_oja_bwd_kernel_dA'
    from yb_8b1f9f64c8c620 import chunk_oja_bwd_dA
    v = load_fixture(os.path.join(load_dir, 'v.pt')).to(DEVICE)
    gv = load_fixture(os.path.join(load_dir, 'gv.pt')).to(DEVICE)
    do = load_fixture(os.path.join(load_dir, 'do.pt')).to(DEVICE)
    scale = load_fixture(os.path.join(load_dir, 'scale.pt')).item()
    print('Warming up...')
    for _ in range(3):
        dA = chunk_oja_bwd_dA(v=v, gv=gv, do=do, scale=scale)
        torch.npu.synchronize()
    print('Benchmarking...')
    times = []
    for _ in range(10):
        start = time.perf_counter()
        dA = chunk_oja_bwd_dA(v=v, gv=gv, do=do, scale=scale)
        torch.npu.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000000.0)
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    result = f'[chunk_oja_bwd_kernel_dA] B={B}, T={T}, H={H}, D={D}, avg_time={avg_time:.2f}us, min_time={min_time:.2f}us, max_time={max_time:.2f}us'
    print(result)
    result_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmarks.result')
    with open(result_file, 'a') as f:
        f.write(result + '\n')
    print(f'\nResult saved to {result_file}')
