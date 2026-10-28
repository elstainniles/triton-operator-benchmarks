"""
Fused MoE Kernel - 100 Generalized Test Cases

All input data is generated on CPU with deterministic seed, then copied to device.
Output is moved to CPU and saved as .pt format (torch.save after .cpu(), always loadable).

Usage:
  On GPU: python test_fused_moe_kernel.py
  On NPU: python test_fused_moe_kernel.py
  Specify output dir:
    python test_fused_moe_kernel.py --output_dir test_results/gpu
  Run specific cases:
    python test_fused_moe_kernel.py --cases 0 1 2
  Compare results (after copying both directories to the same machine):
    python compare_results.py --gpu_dir test_results/gpu --npu_dir test_results/npu
"""

import torch
import numpy as np
import hashlib
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import triton.language as tl
from fused_moe_kernel import invoke_fused_moe_kernel


# =============================================================================
# Type Maps
# =============================================================================

_DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16}
_COMPUTE_TYPE_MAP = {"float16": tl.float16, "float32": tl.float32}
_SCALE_DTYPE_MAP = {"float16": torch.float16, "float32": torch.float32}
_CASE_TYPE_NAMES = {
    1: "int8_w8a16",
    2: "block_int8_w8a8",
    3: "fp8_w8a8_tensor",
    4: "fp16_no_quant",
}


# =============================================================================
# Test Case Generation — 25 dimension combos × 4 quant modes = 100 cases
# =============================================================================

def generate_test_cases():
    """Generate 100 test cases with diverse parameters."""
    cases = []

    # 25 dimension combos: (M, N, K, num_experts, top_k)
    dim_combos = [
        # # Very small
        (1,   64,    64,   2,  1),
        (1,   128,   128,  4,  1),
        (2,   64,    128,  4,  2),
        # # Small
        (4,   128,   128,  4,  2),
        (8,   128,   256,  8,  2),
        (8,   256,   128,  4,  4),
        (8,   512,   512,  8,  2),
        # # Medium
        (16,  256,   256,  8,  4),
        (16,  512,   256,  16, 2),
        (32,  256,   256,  8,  2),
        (32,  512,   512,  16, 4),
        (32,  1024,  256,  8,  2),
        # # Medium-large
        (64,  256,   256,  8,  4),
        (64,  512,   512,  16, 2),
        (64,  1024,  512,  32, 4),
        (128, 512,   256,  8,  2),
        # # Large
        (128, 1024,  512,  16, 4),
        (256, 512,   256,  8,  4),
        # Edge / stress
        (1,   1024,  1024, 32, 1),    # large N/K, single token
        (64,  128,   1024, 16, 4),    # large K, small N
        (32,  2048,  256,  8,  2),    # large N
        (16,  1024,  2048, 32, 4),    # large K
        (128, 256,   1024, 16, 2),    # large K, medium M
        (32,  256,   256,  8,  1),    # top_k=1
        (64,  256,   256,  32, 8),    # many experts
    ]

    case_types = [1, 2, 3, 4]

    for i, (M, N, K, E, top_k) in enumerate(dim_combos):
        for j, case_type in enumerate(case_types):
            case_id = i * 4 + j

            # int8_w8a16: tl.dot(a, b.to(compute_type)) requires a and
            # b.to(compute_type) to be the same dtype. Since a is loaded in
            # the original dtype and b is cast to compute_type, they must match.
            # Only float16 compute_type is available, so dtype must be float16.
            if case_type == 1:
                dtype = "float16"
                compute_type = "float16"
            else:
                dtype_choices = ["float16", "bfloat16"]
                compute_choices = ["float16", "float32"]
                dtype = dtype_choices[(i + j) % 2]
                compute_type = compute_choices[(i * 2 + j) % 2]
            scale_dtype = "float32" if compute_type == "float32" else "float16"

            mul_routed_weight = ((i + j) % 3 != 0)
            c_sorted = ((i + j) % 5 == 0)

            GROUP_N = 64 if N % 64 == 0 else (32 if N % 32 == 0 else 16)
            GROUP_K = 64 if K % 64 == 0 else (32 if K % 32 == 0 else 16)

            cases.append({
                "case_id": case_id,
                "M": M, "N": N, "K": K,
                "num_experts": E, "top_k": top_k,
                "dtype": dtype,
                "compute_type": compute_type,
                "scale_dtype": scale_dtype,
                "case_type": case_type,
                "case_type_name": _CASE_TYPE_NAMES[case_type],
                "mul_routed_weight": mul_routed_weight,
                "filter_expert": False,
                "c_sorted": c_sorted,
                "swap_ab": False,
                "GROUP_N": GROUP_N,
                "GROUP_K": GROUP_K,
                "seed": 42 + case_id,
            })

    return cases[:100]


# =============================================================================
# Device Detection
# =============================================================================

def get_device():
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return torch.device('npu:0')
    elif hasattr(torch, 'cuda') and torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


def get_device_name(device):
    if 'npu' in str(device):
        return 'npu'
    elif 'cuda' in str(device):
        return 'gpu'
    return 'cpu'


# =============================================================================
# Data Preparation (all on CPU to guarantee identical inputs across devices)
# =============================================================================

def create_base_data(cfg):
    """Create base MoE input data on CPU with proper topk gating.

    Conventions (matching vLLM kernel expectations):
    - A has shape (M * top_k, K): row i holds the hidden state of token i // top_k.
    - sorted_token_ids: indices into the M*top_k rows of A, sorted by expert.
      The kernel uses sorted_token_ids to look up A rows, and
      (sorted_token_ids // top_k) to index into A_scale (case type 2).
    - expert_ids[b]: the expert assigned to block b.
    - num_tokens_post_padded: number of valid tokens (= M * top_k).

    Padding must align to the kernel's BLOCK_SIZE_M.  Since BLOCK_SIZE_M
    is chosen by @triton.autotune at runtime, we pad to the maximum
    possible BLOCK_SIZE_M across all autotune configs to guarantee
    correctness regardless of which config is selected.
    """
    torch.manual_seed(cfg["seed"])
    M, N, K = cfg["M"], cfg["N"], cfg["K"]
    num_experts, top_k = cfg["num_experts"], cfg["top_k"]
    dtype = _DTYPE_MAP[cfg["dtype"]]

    # Must equal the BLOCK_SIZE_M used by the kernel's autotune config.
    # The kernel reads expert_ids[pid_m] where pid_m indexes blocks of
    # size BLOCK_SIZE_M, so expert_ids must have one entry per block
    # at this granularity.  Current active config: BLOCK_SIZE_M=32.
    block_size_m = 32

    # Base hidden states (M tokens)
    A_base = (torch.randn(M, K, dtype=dtype) * 0.1)

    # Expand A: each token appears top_k times
    # Row i of A_expanded = hidden state of token i // top_k
    A_expanded = A_base.repeat_interleave(top_k, dim=0)  # (M*top_k, K)

    B = (torch.randn(num_experts, N, K, dtype=dtype) * 0.05)

    # Simulate gating: each token routes to top_k experts
    gating_logits = torch.randn(M, num_experts, dtype=torch.float32)
    topk_vals, topk_indices = torch.topk(gating_logits, top_k, dim=1)
    topk_weights = torch.softmax(topk_vals, dim=1).to(dtype)

    num_tokens = M * top_k

    # Group positions by expert, padding each expert group to a
    # multiple of block_size_m so that every block contains tokens
    # from a single expert (required by the kernel).
    expert_groups = [[] for _ in range(num_experts)]
    for pos in range(num_tokens):
        token_id = pos // top_k
        k_idx = pos % top_k
        expert_id = topk_indices[token_id, k_idx].item()
        expert_groups[expert_id].append(pos)

    sorted_token_ids_list = []
    expert_ids_list = []

    for expert_id in range(num_experts):
        group = expert_groups[expert_id]
        # Pad this expert's group to block_size_m alignment
        padded_len = ((len(group) + block_size_m - 1) // block_size_m) * block_size_m
        group_padded = group + [num_tokens] * (padded_len - len(group))
        sorted_token_ids_list.extend(group_padded)
        # One expert_ids entry per block in this group
        num_blocks_for_expert = padded_len // block_size_m
        expert_ids_list.extend([expert_id] * num_blocks_for_expert)

    sorted_token_ids = torch.tensor(sorted_token_ids_list, dtype=torch.int32)
    expert_ids = torch.tensor(expert_ids_list, dtype=torch.int32)

    EM_PAD = len(sorted_token_ids)

    # Pad A to EM_PAD rows
    if EM_PAD > num_tokens:
        A_expanded = torch.cat([
            A_expanded,
            torch.zeros(EM_PAD - num_tokens, K, dtype=dtype),
        ])

    # num_tokens_post_padded = number of valid (non-padding) tokens
    num_tokens_post_padded = torch.tensor([num_tokens], dtype=torch.int32)

    C = torch.zeros(EM_PAD, N, dtype=dtype)

    return {
        'A': A_expanded, 'A_base': A_base, 'B': B, 'C': C,
        'topk_weights': topk_weights,
        'sorted_token_ids': sorted_token_ids,
        'expert_ids': expert_ids,
        'num_tokens_post_padded': num_tokens_post_padded,
        'EM_PAD': EM_PAD,
    }


def prepare_case1_data(data, cfg):
    """INT8 w8a16 quantization — prepare on CPU."""
    scale_dtype = _SCALE_DTYPE_MAP[cfg["scale_dtype"]]
    B = data['B'].clone()
    B_scale = (B.abs().max(dim=2)[0] / 127.0).to(scale_dtype)
    B_q = (B / (B_scale[:, :, None] + 1e-5)).round().clamp(-128, 127).to(torch.int8)
    return (
        {**data, 'B': B_q.contiguous(), 'A_scale': None, 'B_scale': B_scale.contiguous()},
        {'use_int8_w8a16': True, 'per_channel_quant': True},
    )


def prepare_case2_data(data, cfg):
    """Block-wise INT8 w8a8 quantization — prepare on CPU."""
    scale_dtype = _SCALE_DTYPE_MAP[cfg["scale_dtype"]]
    M, N, K = cfg["M"], cfg["N"], cfg["K"]
    num_experts = cfg["num_experts"]
    top_k = cfg["top_k"]
    GROUP_N, GROUP_K = cfg["GROUP_N"], cfg["GROUP_K"]

    num_k_blocks = K // GROUP_K

    # A_scale is per-original-token (shape [M, num_k_blocks]).
    # The kernel indexes A_scale via (offs_token // top_k), where
    # offs_token = sorted_token_ids[...], which stores position indices
    # 0..M*top_k-1. So position // top_k gives the original token id.
    # Use A_base (the original M token hidden states), not A_expanded.
    A_base = data['A_base'].clone().reshape(M, num_k_blocks, GROUP_K)
    A_scale = A_base.abs().max(dim=-1, keepdim=True)[0] / 127.0
    A_scale = A_scale.squeeze(-1).to(scale_dtype)

    # Quantize A row-by-row using per-original-token scales.
    # Since A[i] = A_base[i // top_k], all rows sharing the same
    # original token use the same scale.
    EM_PAD = data['EM_PAD']
    A_all = data['A'][:EM_PAD].clone().reshape(EM_PAD, num_k_blocks, GROUP_K)
    row_to_token = torch.arange(EM_PAD, dtype=torch.int32) // top_k
    row_to_token = row_to_token.clamp(max=M - 1)
    scale_per_row = A_scale[row_to_token]  # (EM_PAD, num_k_blocks)
    A_q = (
        torch.round(A_all / (scale_per_row.unsqueeze(-1) + 1e-5))
        .clamp(-128, 127)
        .to(torch.int8)
        .reshape(EM_PAD, K)
    )

    B_scale = torch.zeros(num_experts, N // GROUP_N, K // GROUP_K, dtype=scale_dtype)
    B_q = torch.zeros(num_experts, N, K, dtype=torch.int8)
    for e in range(num_experts):
        for ii in range(N // GROUP_N):
            for jj in range(K // GROUP_K):
                blk = data['B'][e, ii*GROUP_N:(ii+1)*GROUP_N, jj*GROUP_K:(jj+1)*GROUP_K]
                s = blk.abs().max() / 127.0
                B_scale[e, ii, jj] = s.to(scale_dtype)
                B_q[e, ii*GROUP_N:(ii+1)*GROUP_N, jj*GROUP_K:(jj+1)*GROUP_K] = (
                    (blk / (s + 1e-5)).round().clamp(-128, 127).to(torch.int8)
                )

    return (
        {**data, 'A': A_q.contiguous(), 'B': B_q.contiguous(),
         'A_scale': A_scale.contiguous(), 'B_scale': B_scale.contiguous()},
        {'use_int8_w8a8': True, 'block_shape': [GROUP_N, GROUP_K]},
    )


def prepare_case3_data(data, cfg):
    """FP8 w8a8 tensor-wise quantization — prepare on CPU."""
    if not hasattr(torch, 'float8_e4m3fn'):
        raise NotImplementedError(
            "torch.float8_e4m3fn not available — requires PyTorch >= 2.1"
        )

    scale_dtype = _SCALE_DTYPE_MAP[cfg["scale_dtype"]]
    N, K = cfg["N"], cfg["K"]
    num_experts = cfg["num_experts"]
    fp8_max = 448.0

    A = data['A'].clone()
    A_scale = A.abs().max() / fp8_max
    A_fp8 = (A / (A_scale + 1e-5)).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)

    B_scale = torch.zeros(num_experts, dtype=scale_dtype)
    B_fp8 = torch.zeros(num_experts, N, K, dtype=torch.float8_e4m3fn)
    for e in range(num_experts):
        s = data['B'][e].abs().max() / fp8_max
        B_scale[e] = s.to(scale_dtype)
        B_fp8[e] = (data['B'][e] / (s + 1e-5)).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)

    return (
        {**data, 'A': A_fp8.contiguous(), 'B': B_fp8.contiguous(),
         'A_scale': A_scale.unsqueeze(0).to(scale_dtype).contiguous(),
         'B_scale': B_scale.contiguous()},
        {'use_fp8_w8a8': True, 'block_shape': None},
    )


def prepare_case4_data(data, cfg):
    """FP16/BF16 no quantization."""
    return data, {'use_fp8_w8a8': False, 'use_int8_w8a8': False, 'use_int8_w8a16': False}


_CASE_PREPARE_MAP = {
    1: prepare_case1_data,
    2: prepare_case2_data,
    3: prepare_case3_data,
    4: prepare_case4_data,
}


# =============================================================================
# Input Checksum — verify identical inputs across devices
# =============================================================================

def tensor_checksum(t: torch.Tensor) -> str:
    """Compute MD5 hex digest of a CPU tensor's raw bytes.

    The tensor must be on CPU and contiguous. For dtypes unsupported by
    numpy (bfloat16, float8_e4m3fn, etc.), we view the raw bytes as uint
    before converting — this preserves the exact bit pattern for checksum.
    """
    assert not t.is_cuda and not (hasattr(t, 'is_npu') and t.is_npu), \
        "Checksum must be computed on CPU tensors"
    t = t.detach().contiguous()
    # Map element size to a numpy-compatible uint view that preserves bit pattern
    _uint_view = {1: torch.uint8, 2: torch.uint16, 4: torch.int32, 8: torch.int64}
    view_dtype = _uint_view.get(t.element_size())
    if view_dtype is not None:
        buf = t.view(view_dtype).numpy().tobytes()
    else:
        buf = t.numpy().tobytes()
    return hashlib.md5(buf).hexdigest()


def compute_input_checksums(kernel_data: dict) -> dict:
    """Compute checksums for all input tensors that go into the kernel.

    These checksums are saved in metadata.json. The comparison script
    can then verify that GPU and NPU received bitwise-identical inputs.
    """
    checksums = {}
    for key in ['A', 'B', 'topk_weights', 'sorted_token_ids',
                'expert_ids', 'num_tokens_post_padded',
                'A_scale', 'B_scale']:
        val = kernel_data.get(key)
        if val is not None and isinstance(val, torch.Tensor):
            checksums[key] = tensor_checksum(val.cpu())
    return checksums


# =============================================================================
# Kernel Execution
# =============================================================================

def run_single_case(cfg, device):
    """Run a single test case.

    Returns (output_np, input_checksums):
      - output_np: output as numpy float32 on CPU
      - input_checksums: MD5 checksums of all kernel input tensors (CPU)
    """
    base_data = create_base_data(cfg)

    # 为当前用例指定独立的 profiling 输出目录，使每个用例的 ascend_pt
    # 结果落在单独文件夹中，方便 extract_moe_perf.py 按用例抓取第 7 列平均性能。
    prof_root = os.environ.get("MOE_PROF_ROOT", "./prof_fused_moe_dir")
    os.environ["MOE_PROF_DIR"] = os.path.join(
        prof_root, f"case_{cfg['case_id']:03d}"
    )

    case_type = cfg["case_type"]
    kernel_data, kernel_extra = _CASE_PREPARE_MAP[case_type](base_data, cfg)

    # A_base is only needed for quantization prep, not for the kernel
    kernel_data.pop('A_base', None)

    # Compute checksums on CPU BEFORE copying to device
    input_checksums = compute_input_checksums(kernel_data)

    # Copy all tensors to device
    device_data = {}
    for key, val in kernel_data.items():
        if isinstance(val, torch.Tensor):
            device_data[key] = val.to(device).contiguous()
        else:
            device_data[key] = val

    C = torch.zeros_like(device_data['C']).contiguous()
    invoke_fused_moe_kernel(
        A=device_data['A'],
        B=device_data['B'],
        bias=None,
        C=C,
        A_scale=device_data.get('A_scale'),
        B_scale=device_data.get('B_scale'),
        topk_weights=device_data['topk_weights'],
        sorted_token_ids=device_data['sorted_token_ids'],
        expert_ids=device_data['expert_ids'],
        num_tokens_post_padded=device_data['num_tokens_post_padded'],
        mul_routed_weight=cfg["mul_routed_weight"],
        top_k=cfg["top_k"],
        compute_type=_COMPUTE_TYPE_MAP[cfg["compute_type"]],
        use_int8_w8a16=kernel_extra.get('use_int8_w8a16', False),
        use_fp8_w8a8=kernel_extra.get('use_fp8_w8a8', False),
        use_int8_w8a8=kernel_extra.get('use_int8_w8a8', False),
        per_channel_quant=kernel_extra.get('per_channel_quant', False),
        block_shape=kernel_extra.get('block_shape', None),
        swap_ab=cfg["swap_ab"],
        c_sorted=cfg["c_sorted"],
        filter_expert=cfg["filter_expert"],
    )

    output_cpu = C.detach().cpu().float()
    return output_cpu, input_checksums


# =============================================================================
# Save Results
# =============================================================================

def save_case_result(output_cpu, cfg, case_dir, input_checksums):
    """Save case output and metadata.

    Output is stored as CPU float32 tensor (.pt) via torch.save —
    since we .cpu() before saving, it can be loaded on any machine.
    input_checksums are MD5 digests of the kernel input tensors, saved in
    metadata so the comparison script can verify identical inputs across devices.
    """
    os.makedirs(case_dir, exist_ok=True)

    torch.save(output_cpu, os.path.join(case_dir, "output.pt"))

    metadata = {
        "case_id": cfg["case_id"],
        "M": cfg["M"], "N": cfg["N"], "K": cfg["K"],
        "num_experts": cfg["num_experts"], "top_k": cfg["top_k"],
        "dtype": cfg["dtype"],
        "compute_type": cfg["compute_type"],
        "scale_dtype": cfg["scale_dtype"],
        "case_type": cfg["case_type"],
        "case_type_name": cfg["case_type_name"],
        "mul_routed_weight": cfg["mul_routed_weight"],
        "filter_expert": cfg["filter_expert"],
        "c_sorted": cfg["c_sorted"],
        "swap_ab": cfg["swap_ab"],
        "GROUP_N": cfg["GROUP_N"],
        "GROUP_K": cfg["GROUP_K"],
        "seed": cfg["seed"],
        "output_shape": list(output_cpu.shape),
        "input_checksums": input_checksums,
    }
    with open(os.path.join(case_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)


def save_case_error(error_msg, case_dir):
    """Save error info for a failed case."""
    os.makedirs(case_dir, exist_ok=True)
    with open(os.path.join(case_dir, "error.txt"), 'w') as f:
        f.write(error_msg)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fused MoE Kernel — 100 Generalized Test Cases"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: ./test_results/<device>)",
    )
    parser.add_argument(
        "--cases", type=int, nargs='+', default=None,
        help="Specific case IDs to run (0-99). Default: all 100 cases",
    )
    args = parser.parse_args()

    device = get_device()
    device_name = get_device_name(device)

    if device_name == 'npu':
        try:
            torch.npu.set_device(device)
        except Exception:
            pass

    output_dir = args.output_dir or os.path.join("./test_results", device_name)

    all_cases = generate_test_cases()

    if args.cases is not None:
        case_ids = set(args.cases)
        all_cases = [c for c in all_cases if c["case_id"] in case_ids]

    print("=" * 60)
    print("FUSED MOE KERNEL — 100 GENERALIZED TEST CASES")
    print("=" * 60)
    print(f"Device: {device} ({device_name.upper()})")
    print(f"Output: {os.path.abspath(output_dir)}")
    print(f"Cases to run: {len(all_cases)}")

    passed = 0
    failed = 0
    skipped = 0

    for idx, cfg in enumerate(all_cases):
        case_id = cfg["case_id"]
        case_name = f"case_{case_id:03d}"
        case_dir = os.path.join(output_dir, case_name)
        case_desc = (
            f"M={cfg['M']}, N={cfg['N']}, K={cfg['K']}, "
            f"E={cfg['num_experts']}, top_k={cfg['top_k']}, "
            f"dtype={cfg['dtype']}, ct={cfg['compute_type']}, "
            f"type={cfg['case_type_name']}, "
            f"mul_rw={cfg['mul_routed_weight']}, "
            f"c_sorted={cfg['c_sorted']}"
        )

        print(f"\n[{idx+1}/{len(all_cases)}] {case_name}: {case_desc}")

        # Check case 2 divisibility
        if cfg["case_type"] == 2:
            if cfg["N"] % cfg["GROUP_N"] != 0 or cfg["K"] % cfg["GROUP_K"] != 0:
                print(f"  SKIP: N%GROUP_N!=0 or K%GROUP_K!=0")
                skipped += 1
                continue

        try:
            output_cpu, input_checksums = run_single_case(cfg, device)

            has_nan = bool(torch.any(output_cpu.isnan()))
            has_inf = bool(torch.any(output_cpu.isinf()))
            if has_nan or has_inf:
                print(f"  WARNING: NaN={has_nan}, Inf={has_inf}")

            save_case_result(output_cpu, cfg, case_dir, input_checksums)
            print(
                f"  PASS: shape={list(output_cpu.shape)}, "
                f"range=[{output_cpu.min():.6f}, {output_cpu.max():.6f}]"
            )
            passed += 1

        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            save_case_error(str(e), case_dir)
            failed += 1

    # Save run summary
    summary = {
        "device": device_name,
        "output_dir": os.path.abspath(output_dir),
        "total": len(all_cases),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
    }
    summary_path = os.path.join(output_dir, "run_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Total:   {len(all_cases)}")
    print(f"  Output:  {os.path.abspath(output_dir)}")
    print(f"  Summary: {summary_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())