
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
import triton
NPU_ID = os.environ.get('BENCH_NPU_ID', '0')
torch.npu.set_device(int(NPU_ID))
DEVICE = f'npu:{torch.npu.current_device()}'

@pytest.mark.functiontest
def test_parallel_deltaformer_bwd_qk_function():
    from yb_a87c34e2ef2580 import parallel_deltaformer_bwd_kernel_qk
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_deltaformer_bwd_kernel_qk'
    kernel_name = 'parallel_deltaformer_bwd_kernel_qk'
    q = load_fixture(os.path.join(load_dir, 'input_q.pt')).to(DEVICE)
    k = load_fixture(os.path.join(load_dir, 'input_k.pt')).to(DEVICE)
    u = load_fixture(os.path.join(load_dir, 'input_u.pt')).to(DEVICE)
    lse = load_fixture(os.path.join(load_dir, 'input_lse.pt')).to(DEVICE)
    grad_v = load_fixture(os.path.join(load_dir, 'input_grad_v.pt')).to(DEVICE)
    beta = load_fixture(os.path.join(load_dir, 'input_beta.pt')).to(DEVICE)
    row_dot_sum = load_fixture(os.path.join(load_dir, 'input_row_dot_sum.pt')).to(DEVICE)
    params = load_fixture(os.path.join(load_dir, 'params.pt'))
    H, T, D = (params['H'], params['T'], params['D'])
    fa_scale, qk_scale = (params['fa_scale'], params['qk_scale'])
    grad_q = torch.empty_like(q)
    grad_k = torch.empty_like(k)

    def grid_bp(META):
        return (triton.cdiv(T, META['BLOCK_C']), H)
    parallel_deltaformer_bwd_kernel_qk[grid_bp](grad_q, grad_k, q, k, grad_v, u, lse, beta, row_dot_sum, H, T, D, fa_scale, qk_scale, disable_tightly_coupled_buffer_reuse=True)
    torch.npu.synchronize()
    expected_grad_q = load_fixture(os.path.join(load_dir, 'output_grad_q.pt')).to(DEVICE)
    expected_grad_k = load_fixture(os.path.join(load_dir, 'output_grad_k.pt')).to(DEVICE)
    torch.testing.assert_close(grad_q, expected_grad_q, atol=0.001, rtol=0.001)
    torch.testing.assert_close(grad_k, expected_grad_k, atol=0.001, rtol=0.001)
    print(f'[parallel_deltaformer_bwd_qk] Precision check passed!')
