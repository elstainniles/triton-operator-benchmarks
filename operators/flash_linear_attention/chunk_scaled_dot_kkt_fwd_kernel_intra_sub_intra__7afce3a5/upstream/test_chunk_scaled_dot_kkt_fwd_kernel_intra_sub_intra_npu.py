import torch
import torch_npu
import triton
import triton.language as tl
import os

os.environ['TRITON_F32_DEFAULT'] = 'ieee'


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BK", "BT"]
)
@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel_intra_sub_intra(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + o_i) < T
    o_A = (bos + i_t * BT + i_i * BC + o_i) * H*BT + i_h * BT + i_i * BC

    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_b = beta + (bos + i_t * BT + i_i * BC + o_i) * H + i_h

    b_k = tl.load(p_k, boundary_check=(0, 1)) * tl.load(p_b, mask=m_A, other=0)[:, None]
    b_g = tl.load(p_g, boundary_check=(0, 1))

    p_kt = k + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k
    p_gk = g + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_kt = tl.load(p_kt, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A = tl.sum(b_k * b_kt[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i > j, b_A, 0.)

        tl.store(A + o_A + j, b_A, mask=m_A)
        p_kt += H*K
        p_gk += H*K


def validate_cmp(dtype, y_cal, y_ref, overflow_mode=None, device_type=None):
    if device_type is not None:
        target_device = torch.device(device_type)
        y_cal = y_cal.to(target_device)
        y_ref = y_ref.to(target_device)
    else:
        y_cal = y_cal.npu()
        y_ref = y_ref.npu()
    if overflow_mode == "saturate":
        if dtype in ['float32', 'float16']:
            min_value = -torch.finfo(dtype).min
            max_value = torch.finfo(dtype).max
        elif dtype in ['int32', 'int16', 'int8']:
            min_value = torch.iinfo(dtype).min
            max_value = torch.iinfo(dtype).max
        elif dtype == 'bool':
            min_value = 0
            max_value = 1
        else:
            raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))
        y_ref = torch.clamp(y_ref, min=min_value, max=max_value)
    if dtype == 'float16':
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'bfloat16':
        torch.testing.assert_close(y_ref.to(torch.float32), y_cal.to(torch.float32), rtol=5e-03, atol=5e-03,
                                   equal_nan=True)
    elif dtype == 'float32':
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-5, atol=1e-5, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'int8':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter "dtype" is found : {}'.format(dtype))


def compare_data_precision(dict_ref: dict, dict_cal: dict, device_type: str):
    keys_ref, keys_cal = set(dict_ref.keys()), set(dict_cal.keys())
    if not keys_ref.issubset(keys_cal):
        raise ValueError("The keys of dict_ref is not subset of dict_cal")

    for key in dict_ref.keys():
        val_a, val_b = dict_ref[key], dict_cal[key]
        if val_a is None and val_b is None:
            continue
        if type(val_a) != type(val_b):
            raise ValueError("The data type of two dicts are different")

        if isinstance(val_a, torch.Tensor):
            validate_cmp(dtype=str(val_a.dtype).split('.')[-1], y_ref=val_a, y_cal=val_b, device_type=device_type)
        else:
            raise ValueError("Non-tensor type is not currently supported")


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


def test_chunk_scaled_dot_kkt_fwd_kernel_intra_sub_intra():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/chunk_scaled_dot_kkt_fwd_kernel_intra_sub_intra/test_chunk_scaled_dot_kkt_fwd_kernel_intra_sub_intra_cuda.pt'
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)

    device = torch.device('npu')
    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')

    B = input_data['B']
    T = input_data['T']
    H = input_data['H']
    K = input_data['K']
    BT = input_data['BT']
    BC = input_data['BC']
    BK = input_data['BK']
    NT = input_data['NT']
    NC = input_data['NC']

    A = torch.zeros(B, T, H, BT, device=device, dtype=torch.float32)

    grid = (NT, NC, B * H)
    chunk_scaled_dot_kkt_fwd_kernel_intra_sub_intra[grid](
        k=input_data['k'],
        g=input_data['g'],
        beta=input_data['beta'],
        A=A,
        cu_seqlens=None,
        chunk_indices=None,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
    )

    npu_out = {}
    npu_out['A'] = A

    compare_data_precision(data["gpu_output"], npu_out, device_type='cpu')
    print("Test chunk_scaled_dot_kkt_fwd_kernel_intra_sub_intra passed!")


if __name__ == "__main__":
    test_chunk_scaled_dot_kkt_fwd_kernel_intra_sub_intra()
