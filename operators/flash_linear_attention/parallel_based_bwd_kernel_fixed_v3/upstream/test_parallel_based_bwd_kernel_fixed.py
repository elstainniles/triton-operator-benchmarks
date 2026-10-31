import torch
import triton
import triton.language as tl

@triton.jit
def _parallel_based_bwd_dq_fixed_v3(
    i_bh,
    i_c,
    i_k,
    i_v,
    q,
    k,
    v,
    do,
    dz,
    dq,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
):
    """修复版本 v3：使用条件判断 + make_block_ptr"""
    p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    p_q = tl.make_block_ptr(q + (i_bh) * T*K, (T, K), (K, 1), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    b_do = tl.load(p_do, boundary_check=(0, 1)).to(b_q.dtype)
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)
    
    p_dz = dz + i_bh * T + i_c * BTL + tl.arange(0, BTL)
    b_dz = tl.load(p_dz, mask=(i_c * BTL + tl.arange(0, BTL)) < T)

    # 第一个循环：使用条件判断避免 tl.dot bug
    if i_c > 0:
        p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (0, i_k * BK), (BTS, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * T*V, (V, T), (1, V), (i_v * BV, 0), (BV, BTS), (0, 1))
        
        for _ in range(0, i_c * BTL, BTS):
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_ds = tl.dot(b_do, b_v, allow_tf32=False)
            if i_v == 0:
                b_ds += b_dz[:, None]
            else:
                b_ds = b_ds
            b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
            b_dq += tl.dot((b_ds * (1 + b_s)).to(b_v.dtype), b_k, allow_tf32=False)
            p_k = tl.advance(p_k, (BTS, 0))
            p_v = tl.advance(p_v, (0, BTS))

    b_dq *= scale
    o_q = tl.arange(0, BTL)
    o_k = tl.arange(0, BTS)
    
    # 第二个循环
    p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_c * BTL, i_k * BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (V, T), (1, V), (i_v * BV, i_c * BTL), (BV, BTS), (0, 1))
    
    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        
        m_s = o_q[:, None] >= o_k[None, :]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        else:
            b_ds = b_ds
        b_ds = tl.where(m_s, b_ds, 0) * scale
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        b_dq += tl.dot((b_ds + b_ds * b_s).to(b_k.dtype), b_k, allow_tf32=False)
        
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))
        o_k += BTS
    
    p_dq = tl.make_block_ptr(dq + (i_bh + B * H * i_v) * T*K, (T, K), (K, 1), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    return


@triton.jit
def _parallel_based_bwd_dkv_fixed_v3(
    i_bh,
    i_c,
    i_k,
    i_v,
    q,
    k,
    v,
    do,
    dz,
    dk,
    dv,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
):
    """修复版本 v3：使用条件判断 + make_block_ptr"""
    p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    b_k, b_v = tl.load(p_k, boundary_check=(0, 1)), tl.load(p_v, boundary_check=(0, 1))
    b_dk, b_dv = tl.zeros([BTL, BK], dtype=tl.float32), tl.zeros([BTL, BV], dtype=tl.float32)

    # 第一个循环：使用条件判断
    if i_c < tl.cdiv(T, BTL) - 1:
        p_q = tl.make_block_ptr(q + i_bh * T*K, (K, T), (1, K), (i_k * BK, (tl.cdiv(T, BTS) * BTS)-BTS), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * T*V, (V, T), (1, V), (i_v * BV, (tl.cdiv(T, BTS) * BTS)-BTS), (BV, BTS), (0, 1))
        
        for i in range((tl.cdiv(T, BTS) * BTS)-BTS, (i_c + 1) * BTL - BTS, -BTS):
            p_dz = dz + i_bh * T + i + tl.arange(0, BTS)
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_do = tl.load(p_do, boundary_check=(0, 1)).to(b_q.dtype)
            b_dz = tl.load(p_dz, mask=(i + tl.arange(0, BTS)) < T)
            b_s = tl.dot(b_k.to(b_q.dtype), b_q, allow_tf32=False) * scale
            b_s2 = 1 + b_s + 0.5 * b_s * b_s
            b_dv += tl.dot(b_s2.to(b_q.dtype), tl.trans(b_do), allow_tf32=False)
            b_ds = tl.dot(b_v, b_do, allow_tf32=False) * scale
            if i_v == 0:
                b_ds += b_dz[None, :] * scale
            else:
                b_ds = b_ds
            b_dk += tl.dot((b_ds + b_ds * b_s).to(b_q.dtype), tl.trans(b_q), allow_tf32=False)
            p_q = tl.advance(p_q, (0, -BTS))
            p_do = tl.advance(p_do, (0, -BTS))

    tl.debug_barrier()
    o_q, o_k = tl.arange(0, BTS), tl.arange(0, BTL)
    
    # 第二个循环
    p_q = tl.make_block_ptr(q + i_bh * T*K, (K, T), (1, K), (i_k * BK, i_c*BTL), (BK, BTS), (0, 1))
    p_do = tl.make_block_ptr(do + i_bh * T*V, (V, T), (1, V), (i_v * BV, i_c*BTL), (BV, BTS), (0, 1))
    
    for i in range(i_c*BTL, (i_c+1)*BTL, BTS):
        p_dz = dz + i_bh * T + i + tl.arange(0, BTS)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(b_q.dtype)
        b_dz = tl.load(p_dz, mask=(i + tl.arange(0, BTS)) < T)
        m_s = o_k[:, None] <= o_q[None, :]
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * scale
        b_s2 = 1 + b_s + 0.5 * b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_s2 = tl.where(m_s, b_s2, 0)

        b_ds = tl.dot(b_v, b_do, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[None, :]
        else:
            b_ds = b_ds
        b_ds = tl.where(m_s, b_ds, 0) * scale
        b_dv += tl.dot(b_s2.to(b_q.dtype), tl.trans(b_do), allow_tf32=False)
        b_dk += tl.dot((b_ds + b_ds * b_s).to(b_q.dtype), tl.trans(b_q), allow_tf32=False)
        p_q = tl.advance(p_q, (0, BTS))
        p_do = tl.advance(p_do, (0, BTS))
        o_q += BTS

    p_dk = tl.make_block_ptr(dk + (i_bh + B * H * i_v) * T*K, (T, K), (K, 1), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (i_bh + B * H * i_k) * T*V, (T, V), (V, 1), (i_c*BTL, i_v*BV), (BTL, BV), (1, 0))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    return


@triton.jit(do_not_specialize=['T'])
def parallel_based_bwd_kernel_fixed_v3(
    q,
    k,
    v,
    do,
    dz,
    dq,
    dk,
    dv,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """修复版本 v3 的 parallel_based_bwd_kernel"""
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % NV
    _parallel_based_bwd_dq_fixed_v3(
        i_bh, i_c, i_k, i_v,
        q, k, v, do, dz, dq,
        scale, T, B, H, BTL, BTS, BK, BV, K, V,
    )
    tl.debug_barrier()
    _parallel_based_bwd_dkv_fixed_v3(
        i_bh, i_c, i_k, i_v,
        q, k, v, do, dz, dk, dv,
        scale, T, B, H, BTL, BTS, BK, BV, K, V,
    )


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


def test_fixed_kernel_v3():
    print("\n" + "="*60)
    print("Testing Fixed Kernel v3")
    print("="*60)
    
    # Load data
    data = torch.load('test_parallel_based_bwd_kernel_v2.pt',
                     map_location='cpu', weights_only=False)
    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')
    
    # Run fixed kernel
    parallel_based_bwd_kernel_fixed_v3[data["grid"]](
        input_data['q'],
        input_data['k'],
        input_data['v'],
        input_data['do'],
        input_data['dz'],
        input_data['dq'],
        input_data['dk'],
        input_data['dv'],
        scale=input_data['scale'],
        T=input_data['T'],
        B=input_data['B'],
        H=input_data['H'],
        K=input_data['K'],
        V=input_data['V'],
        BTL=input_data['BTL'],
        BTS=input_data['BTS'],
        BK=input_data['BK'],
        BV=input_data['BV'],
        enable_auto_bind_sub_block=True,
    )
    
    npu_output = {
        'dq': input_data['dq'].cpu().clone(),
        'dk': input_data['dk'].cpu().clone(),
        'dv': input_data['dv'].cpu().clone(),
    }
    
    # Load GPU reference
    golden_data = torch.load('test_parallel_based_bwd_kernel_fp64_v2.pt',
                            map_location='cpu', weights_only=False)
    golden_output = golden_data['gpu_output']
    
    print("\n" + "="*60)
    print("NPU vs GPU fp64 Precision Comparison (Fixed Kernel v3)")
    print("="*60)
    
    for key in ['dq', 'dk', 'dv']:
        npu_tensor = npu_output[key].float()
        golden_tensor = golden_output[key].float()
        
        abs_diff = torch.abs(npu_tensor - golden_tensor)
        abs_golden = torch.abs(golden_tensor)
        rel_diff = abs_diff / (abs_golden + 1e-10)
        
        max_abs_error = abs_diff.max().item()
        max_rel_error = rel_diff.max().item()
        mean_abs_error = abs_diff.mean().item()
        mean_rel_error = rel_diff.mean().item()
        rmse = torch.sqrt((abs_diff ** 2).mean()).item()
        assert torch.allclose(npu_tensor, golden_tensor, 5e-3, 5e-3)
        
        print(f"\n{key}:")
        print(f"  Max absolute error: {max_abs_error:.6e}")
        print(f"  Max relative error: {max_rel_error:.6e}")
        print(f"  Mean absolute error: {mean_abs_error:.6e}")
        print(f"  Mean relative error: {mean_rel_error:.6e}")
        print(f"  RMSE: {rmse:.6e}")
    
    # Compare with original kernel
    print("\n" + "="*60)
    print("Comparison with Original Kernel")
    print("="*60)
    
    # Load GPU fp32 output
    gpu_fp32_data = torch.load('test_parallel_based_bwd_kernel_v2.pt',
                              map_location='cpu', weights_only=False)
    gpu_fp32_output = gpu_fp32_data['gpu_output']
    
    for key in ['dq', 'dk', 'dv']:
        npu_err = torch.abs(npu_output[key].float() - golden_output[key].float()).max().item()
        gpu_fp32_err = torch.abs(gpu_fp32_output[key].float() - golden_output[key].float()).max().item()
        ratio = npu_err / (gpu_fp32_err + 1e-10)
        print(f"{key}: NPU error / GPU fp32 error = {ratio:.2f}x")
        
        if ratio < 10:
            print(f"  ✅ Fixed! Error ratio is acceptable")
        else:
            print(f"  ❌ Still has issues")


if __name__ == '__main__':
    test_fixed_kernel_v3()
