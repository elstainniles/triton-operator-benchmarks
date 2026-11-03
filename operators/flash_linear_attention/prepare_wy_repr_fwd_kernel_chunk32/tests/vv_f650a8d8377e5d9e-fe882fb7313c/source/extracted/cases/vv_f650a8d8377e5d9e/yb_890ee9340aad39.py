
from opbench.fla_fixtures import load_fixture
import os
import pytest
import torch
import torch_npu
from yb_4b48603b09f088 import prepare_wy_repr_fwd_kernel_chunk32
from yb_4b48603b09f088 import prepare_chunk_indices
from yb_22cded5b91bba3 import cal_precision
import triton

def prepare_wy_repr_fwd(A_ab, cu_seqlens, chunk_size, chunk_indices=None):
    B, T, H, BT = (A_ab.shape[0], A_ab.shape[1], A_ab.shape[2], chunk_size)
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BC = min(BT, 32)
    A_ab_inv = torch.empty_like(A_ab)
    prepare_wy_repr_fwd_kernel_chunk32[NT, B * H](A_ab=A_ab, A_ab_inv=A_ab_inv, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T, H=H, BT=BT, BC=BC)
    return A_ab_inv

def prepare_wy_repr_fwd_ref(A_ab, cu_seqlens, chunk_size):
    B, T, H, BT = A_ab.shape
    A_ab_fp64 = A_ab.to(torch.float64)
    A_ab_inv = torch.zeros(B, T, H, BT, dtype=torch.float64)
    if cu_seqlens is None:
        NT = (T + BT - 1) // BT
        for b in range(B):
            for h in range(H):
                for t_idx in range(NT):
                    start = t_idx * BT
                    end = min(start + BT, T)
                    actual_bt = end - start
                    block = A_ab_fp64[b, start:end, h, :actual_bt].clone()
                    mask = torch.arange(actual_bt)[:, None] > torch.arange(actual_bt)[None, :]
                    mask = mask.npu()
                    block = block * mask.to(block.dtype)
                    for i in range(1, actual_bt):
                        row_i = block[i, :].clone()
                        row_i = row_i + torch.sum(row_i[:, None] * block, dim=0) * (torch.arange(actual_bt).npu() < i).to(block.dtype)
                        block[i, :] = row_i
                    block += torch.eye(actual_bt, dtype=torch.float64).npu()
                    A_ab_inv[b, start:end, h, :actual_bt] = block
    else:
        for b in range(B):
            bos = cu_seqlens[b].item()
            eos = cu_seqlens[b + 1].item()
            seq_len = eos - bos
            NT = (seq_len + BT - 1) // BT
            for h in range(H):
                for t_idx in range(NT):
                    start = t_idx * BT
                    end = min(start + BT, seq_len)
                    actual_bt = end - start
                    block = A_ab_fp64[b, bos + start:bos + end, h, :actual_bt].clone()
                    mask = torch.arange(actual_bt)[:, None] > torch.arange(actual_bt)[None, :]
                    mask = mask.npu()
                    block = block * mask.to(block.dtype)
                    for i in range(1, actual_bt):
                        row_i = block[i, :].clone()
                        row_i = row_i + torch.sum(row_i[:, None] * block, dim=0) * (torch.arange(actual_bt).npu() < i).to(block.dtype)
                        block[i, :] = row_i
                    block += torch.eye(actual_bt, dtype=torch.float64).npu()
                    A_ab_inv[b, bos + start:bos + end, h, :actual_bt] = block
    return A_ab_inv

@pytest.mark.functiontest
def test_prepare_wy_repr_fwd_kernel_chunk32(load_dir):
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs_prepare_wy.pt'), map_location='cpu', weights_only=False)
    device = torch.device('npu')
    A_ab = inputs_dict['A_ab'].to(device)
    cu_seqlens = inputs_dict['cu_seqlens']
    chunk_size = inputs_dict['chunk_size']
    A_ab_inv = prepare_wy_repr_fwd(A_ab, cu_seqlens, chunk_size=chunk_size)
    A_ab_inv_ref = prepare_wy_repr_fwd_ref(inputs_dict['A_ab'].npu(), cu_seqlens, chunk_size).to(device)
    outputs_dict = load_fixture(os.path.join(load_dir, 'outputs_prepare_wy.pt'), map_location='cpu', weights_only=False)
    A_ab_inv_gpu = outputs_dict['A_ab_inv'].to(device)
    print('\nComparing A_ab_inv with expected values...')
    relative_max_npu, relative_mean_npu, rmse_npu = cal_precision(A_ab_inv, A_ab_inv_ref)
    relative_max_gpu, relative_mean_gpu, rmse_gpu = cal_precision(A_ab_inv_gpu, A_ab_inv_ref)
    print(f'最大相对误差比 :{relative_max_npu / relative_max_gpu}, 平均相对误差比 :{relative_mean_npu / relative_mean_gpu}, 均方根误差比 :{rmse_npu / rmse_gpu}')
    assert relative_max_npu / relative_max_gpu <= 2, 'A_ab_inv 最大相对误差比 check failed'
    assert relative_mean_npu / relative_mean_gpu <= 1.2, 'A_ab_inv 平均相对误差比 check failed'
    assert rmse_npu / rmse_gpu <= 1.2, 'A_ab_inv 均方根误差比 check failed'
    print('Test prepare_wy_repr_fwd_kernel_chunk32 passed!')
