
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

@pytest.mark.perftest
def test_chunk_oja_bwd_kernel_dqk_simple_performance():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_oja_bwd_kernel_dqk'
    from yb_8b1f9f64c8c620 import chunk_oja_bwd_dqk
    q = load_fixture(os.path.join(load_dir, 'q.pt')).to(DEVICE)
    k_new = load_fixture(os.path.join(load_dir, 'k_new.pt')).to(DEVICE)
    h = load_fixture(os.path.join(load_dir, 'h.pt')).to(DEVICE)
    gv_cumsum = load_fixture(os.path.join(load_dir, 'gv_cumsum.pt')).to(DEVICE)
    dA = load_fixture(os.path.join(load_dir, 'dA.pt')).to(DEVICE)
    do = load_fixture(os.path.join(load_dir, 'do.pt')).to(DEVICE)
    scale = load_fixture(os.path.join(load_dir, 'scale.pt')).item()
    print('Warming up...')
    for _ in range(3):
        A_qk, dq, dk = chunk_oja_bwd_dqk(q=q, k=k_new, h=h, gv=gv_cumsum, dA=dA, do=do, scale=scale)
        torch.npu.synchronize()
    print('Benchmarking...')
    times = []
    for _ in range(10):
        start = time.perf_counter()
        A_qk, dq, dk = chunk_oja_bwd_dqk(q=q, k=k_new, h=h, gv=gv_cumsum, dA=dA, do=do, scale=scale)
        torch.npu.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000000.0)
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    result = f'[chunk_oja_bwd_kernel_dqk] B={B}, T={T}, H={H}, D={D}, avg_time={avg_time:.2f}us, min_time={min_time:.2f}us, max_time={max_time:.2f}us'
    print(result)
    result_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmarks.result')
    with open(result_file, 'a') as f:
        f.write(result + '\n')
    print(f'\nResult saved to {result_file}')
