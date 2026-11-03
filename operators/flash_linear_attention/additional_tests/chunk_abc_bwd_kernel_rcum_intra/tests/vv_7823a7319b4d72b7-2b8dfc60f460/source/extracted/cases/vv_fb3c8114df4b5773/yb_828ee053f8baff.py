# coding=utf-8
# -------------------------------------------------
# Project Name: PythonProject1
# File Name： test_qk_rmsnorm_bias_triton_kernel_npu
# Tool Name: PyCharm
# Author : l00567229
# Date： 2025/12/20 14:19
# -------------------------------------------------
from opbench.fla_fixtures import load_fixture
import pytest
import triton
import torch
import triton.language as tl
import torch_npu
import os
from yb_22cded5b91bba3 import convert_tensor_with_device_type
from yb_22cded5b91bba3 import validate_cmp
from yb_22cded5b91bba3 import cal_precision
from yb_22cded5b91bba3 import compare_data_precision

@triton.jit
def exp(x):
    return tl.exp(x.to(tl.float32))

@triton.jit(do_not_specialize=['T'])
def chunk_abc_bwd_kernel_rcum_intra(s, z, ss, doo, T, S: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, BS: tl.constexpr, NC: tl.constexpr):
    i_s, i_c, i_bh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_t, i_i = (i_c // NC, i_c % NC)
    o_i = tl.arange(0, BC)
    m_o = tl.full([BC, BC], 1.0, dtype=tl.float32)
    p_s = tl.make_block_ptr(s + i_bh * T * S, (T, S), (S, 1), (i_t * BT + i_i * BC, i_s * BS), (BC, BS), (1, 0))
    p_zn = tl.make_block_ptr(z + i_bh * T * S, (T * S,), (1,), ((i_t * BT + i_i * BC + BC - 1) * S + i_s * BS,), (BS,), (0,))
    p_doo = tl.make_block_ptr(doo + i_bh * T * S, (T, S), (S, 1), (i_t * BT + i_i * BC, i_s * BS), (BC, BS), (1, 0))
    b_s = tl.load(p_s, boundary_check=(0, 1))
    b_zn = tl.load(p_zn, boundary_check=(0,))
    b_doo = tl.zeros([BC, BS], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        p_z = tl.make_block_ptr(z + i_bh * T * S, (T, S), (S, 1), (i_t * BT + i_j * BC, i_s * BS), (BC, BS), (1, 0))
        p_ss = tl.make_block_ptr(ss + i_bh * T * S, (T, S), (S, 1), (i_t * BT + i_j * BC, i_s * BS), (BC, BS), (1, 0))
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_ss = tl.load(p_ss, boundary_check=(0, 1))
        b_doo += b_ss * exp(b_zn[None, :] - b_z)
    b_doo = exp(b_s - b_zn[None, :]) * tl.dot(m_o.to(b_s.dtype), b_doo.to(b_s.dtype), allow_tf32=False)
    for j in range(0, BC):
        p_z = tl.make_block_ptr(z + i_bh * T * S, (T * S,), (1,), ((i_t * BT + i_i * BC + j) * S + i_s * BS,), (BS,), (0,))
        p_ss = tl.make_block_ptr(ss + i_bh * T * S, (T * S,), (1,), ((i_t * BT + i_i * BC + j) * S + i_s * BS,), (BS,), (0,))
        b_z = tl.load(p_z, boundary_check=(0,))
        b_ss = tl.load(p_ss, boundary_check=(0,))
        m_i = o_i[:, None] <= j
        b_doo += tl.where(m_i, exp(b_s - b_z[None, :]) * b_ss[None, :], 0.0)
    b_doo += tl.load(p_doo, boundary_check=(0, 1))
    tl.store(p_doo, b_doo.to(p_doo.dtype.element_ty), boundary_check=(0, 1))

@pytest.mark.functiontest
def test_kernel(load_dir):
    try:
        ptfile = os.path.join(load_dir, 'test_chunk_abc_bwd_kernel_rcum_intra.pt')
        data = load_fixture(ptfile, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f'load file {ptfile} failed: {str(e)}')
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    chunk_abc_bwd_kernel_rcum_intra[data['grid']](**input_data)
    try:
        compare_data_precision(data['gpu_output'], input_data, device_type='cpu')
        print('pass')
    except ValueError as e:
        pytest.fail(f'The testcase failed')
