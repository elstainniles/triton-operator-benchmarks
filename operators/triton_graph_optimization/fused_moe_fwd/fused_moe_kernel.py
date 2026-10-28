"""
Fused MoE Kernel for Ascend NPU Triton Environment

Extracted from fused_moe_triton_kernels.py
This is a minimal standalone implementation of the fused_moe_kernel function.
Uses @triton.autotune decorator for automatic configuration tuning.
"""

import os

import torch
import torch_npu
import triton
import triton.language as tl

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
    _support_tensor_descriptor = True
except ImportError:
    _support_tensor_descriptor = False


@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    """
    Write zeros to the output tensor when expert is filtered.
    This is used when filter_expert is enabled and the expert is not
    in the current expert parallel rank.
    """
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# =============================================================================
# Autotune Configuration Space
# =============================================================================

def get_configs():
    """
    Define the configuration space for autotune.
    Returns a list of triton.Config objects with different BLOCK_SIZE_M,
    BLOCK_SIZE_N, BLOCK_SIZE_K, and GROUP_SIZE_M combinations.

    Design rationale for Ascend NPU Triton:
    - num_stages and num_warps are NOT set (NPU has no CUDA pipeline/warp concept)
    - BLOCK_SIZE_M is kept small: MoE dispatches few tokens per expert
    - GROUP_SIZE_M=1 when M is tiny (avoids idle blocks);
      GROUP_SIZE_M=4/8 when M is larger (improves L2 cache reuse)
    - BLOCK_SIZE_K >= 32 for tl.dot efficiency
    - Total configs capped at ~10 to limit autotune warmup overhead
    """
    configs = [
        # ================================================================
        # [OPT-STEP-5] 放开 autotune config 空间
        # ----------------------------------------------------------------
        # 依据：autotune.md (优化点 12)
        #   - 单 config autotune 形同虚设，需要给编译器多几个 tiling 选择
        #   - 不同 (N, K) 组合最优 BLOCK_SIZE_N/K 不同，autotune 替我们选
        #
        # 注意：所有 config **强制 BLOCK_SIZE_M=32**，因为 test_fused_moe_kernel.py
        # 里 `block_size_m = 32` 是硬编码 (expert_ids/sorted_token_ids 按 32 颗粒度
        # 构建)。改 BLOCK_SIZE_M 会破坏 pid_m → expert_ids 的索引对齐。
        # ================================================================
        # ---- Medium M: typical inference batch (e.g., 32-128 tokens) ----
        triton.Config(
            {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4},
        ),
        # triton.Config(
        #     {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4},
        # ),
        triton.Config(
            {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
        ),
        # Larger GROUP_SIZE_M for better L2 reuse when M is sufficient
        triton.Config(
            {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
        ),
        # ---- Large N/K: wide model dimensions (e.g., 14336 MLP hidden) ----
        triton.Config(
            {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
        ),
    ]
    return configs


# =============================================================================
# Autotuned Fused MoE Kernel
# =============================================================================

@triton.autotune(
    configs=get_configs(),
    key=['N', 'K', 'EM'],
)
@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    a_desc,
    b_ptr,
    b_desc,
    bias_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    # ================================================================
    # [OPT-STEP-4] 14 个 stride 参数声明为 tl.constexpr
    # ----------------------------------------------------------------
    # 依据：constexpr_parameters.md
    #   - stride 在 kernel 生命周期内不变 (由 tensor.stride() 决定)
    #   - 声明 constexpr 启用编译时常量折叠 + aggressive 常量传播
    #   - 让指针偏移 `off * stride_xxx` 在编译期化简 (尤其 stride=0/1 时)
    # 代价：每个新 (N, K, dtype) 组合触发一次重编译。autotune warmup
    #       时间上升，但稳态推理纯赚。
    # ================================================================
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bias_e: tl.constexpr,
    stride_bias_n: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_ask: tl.constexpr,
    stride_bse: tl.constexpr,
    stride_bsk: tl.constexpr,
    stride_bsn: tl.constexpr,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    even_Ks: tl.constexpr,
    c_sorted: tl.constexpr,
    filter_expert: tl.constexpr,
    swap_ab: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.

    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    # ================================================================
    # [OPT-STEP-2/3] i64 → i32，消除指针算术的标量降级
    # ----------------------------------------------------------------
    # 依据：avoid_scalar_lowering.md §1
    #   - i64 算术 (+,-,*,//) 全部降级到标量循环
    #   - i32 算术不降级
    # 原代码把 offs_token / off_experts 全部 to(tl.int64)，使得后续
    # `offs_token // top_k * stride_am`、`off_experts * stride_be` 等
    # 指针偏移计算全部走标量路径，污染 K-loop，阻断 Cube lowering。
    #
    # i32 安全范围验证 (本 kernel 适用条件)：
    #   - offs_token_id ≤ EM ≤ cdiv(M*top_k, BLOCK_M) ≤ 几千
    #   - offs_token (token 索引) ≤ M*top_k
    #   - off_experts (专家索引) ≤ num_experts ≤ 几十
    #   - 指针偏移 ≤ offs_token * stride_am ≤ M*top_k * K
    #     对推理 batch (M≤4096, top_k≤8, K≤8192) → ≤ 2^28，i32 安全
    # 如未来扩展到训练大 batch，需要重新评估 (i32 上限 2^31)。
    # ================================================================
    # ---- 原始实现 (已废弃) ----
    # offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    # offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    # offs_token = offs_token.to(tl.int64)
    # off_experts = off_experts_i32.to(tl.int64)
    # ---- 优化实现：全 i32 ----
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # i32
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)  # 默认按 ptr dtype load，i32
    # 不再 .to(tl.int64)；offs_token 保持 i32，后续 // top_k 走向量算术
    token_mask = offs_token < num_valid_tokens

    off_experts_i32 = tl.load(expert_ids_ptr + pid_m)  # i32
    off_experts = off_experts_i32  # 保持 i32，不再 .to(tl.int64)

    if filter_expert and off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    # ================================================================
    # [OPT-STEP-2] 重写 `% N` 为 i32 的 `a - (a // b) * b`
    # ----------------------------------------------------------------
    # 依据：avoid_scalar_lowering.md §3 + checklist §4
    #   - `a % b` **无条件**触发标量降级 (无论 i32/i64)
    #   - 标准修复：替换为 `a - (a // b) * b`，全 i32 算术，走向量单元
    # 顺带把 tl.arange 去掉 .to(tl.int64)，配合 OPT-STEP-3 全 i32 化。
    # ================================================================
    # ---- 原始实现 (已废弃) ----
    # offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    # ---- 优化实现：i32 + 重写 % ----
    offs_bn_base = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # i32
    offs_bn = offs_bn_base - (offs_bn_base // N) * N  # i32 等价于 offs_bn_base % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if a_desc is not None:
        assert use_fp8_w8a8 and group_n > 0 and group_k > 0
        start_offs_m = pid_m * BLOCK_SIZE_M
    else:
        a_ptrs = a_ptr + (
            offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
        )

    if b_desc is not None:
        start_offs_n = pid_n * BLOCK_SIZE_N
    else:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        )

    if bias_ptr is not None:
        bias = tl.load(
            bias_ptr + off_experts * stride_bias_e + offs_bn[None, :] * stride_bias_n
        )
    if use_int8_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            if a_desc is not None:
                a_scale_ptrs = a_scale_ptr + offs_token_id * stride_asm
            else:
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            if BLOCK_SIZE_N > group_n:
                offs_bsn = offs_bn // group_n
            else:
                offs_bsn = pid_n * BLOCK_SIZE_N // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    if swap_ab:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        if a_desc is not None:
            a = a_desc.load([start_offs_m, k_start])
        elif even_Ks:
            # ================================================================
            # [OPT-STEP-0] 回滚 EXPERIMENT A，恢复 mask+other 原版 load
            # ----------------------------------------------------------------
            # 依据：cannbot-skills/ops/triton-latency-optimizer/references/
            #       redundant_boundary_operation.md 的 KVR 框架
            #   - tl.load(..., mask=M, other=0.0) 自带 KVR (M, 0.0)，
            #     padding 行的输入对 tl.dot 自然为 0，完全不需要外部 tl.where。
            #   - EXPERIMENT A 引入的 `tl.where(token_mask, a, 0)` 在 KVR
            #     框架里是冗余的（多余一次 vsel 算子）。
            # 之前的 EXPERIMENT A 假设 (mask+other 挡了 Cube lowering) 已被
            # skill 框架重新分析推翻 —— 真正阻断 Cube 的是 P0-b (i64+% 标量
            # 降级) 和 P0-a (grid 远超核数)，与 load 形态无关。
            # ================================================================
            # ---- EXPERIMENT A 实验实现 (已废弃) ----
            # a = tl.load(a_ptrs)
            # a = tl.where(token_mask[:, None], a, tl.zeros([1], dtype=a.dtype))
            # ---- 原版恢复：mask+other load，KVR 自带 (token_mask, 0.0) ----
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
        else:
            # ================================================================
            # [OPT-STEP-0] (cont.) K 不整齐分支同步回滚
            # ----------------------------------------------------------------
            # K 越界 mask 必须保留在 load 内（物理越界保护）；
            # token_mask 也合并在 load 内（KVR 自带屏蔽），无需外部 tl.where。
            # ================================================================
            # ---- EXPERIMENT A 实验实现 (已废弃) ----
            # a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k_start, other=0.0)
            # a = tl.where(token_mask[:, None], a, tl.zeros([1], dtype=a.dtype))
            # ---- 原版恢复：双重 mask (token + K 越界) 都在 load 内 ----
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
                other=0.0,
            )

        if b_desc is not None:
            b = (
                b_desc.load([off_experts_i32, start_offs_n, k_start])
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
                .T
            )
        elif even_Ks:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)

        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                if swap_ab:
                    a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))
                    a_scale, b_scale = b_scale, a_scale
                if BLOCK_SIZE_N > group_n:
                    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
                else:
                    accumulator += tl.dot(a, b) * (a_scale[:, None] * b_scale)
            else:
                if swap_ab:
                    a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))
                accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        if a_desc is None:
            a_ptrs += BLOCK_SIZE_K * stride_ak
        if b_desc is None:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if swap_ab:
        accumulator = tl.trans(accumulator, (1, 0))

    if use_int8_w8a16:
        accumulator *= b_scale
    elif use_fp8_w8a8 or use_int8_w8a8:
        if group_k == 0 or group_n == 0:
            accumulator *= a_scale * b_scale

    if bias_ptr is not None:
        accumulator += bias

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if c_sorted:
        c_ptrs = (
            c_ptr + stride_cm * offs_token_id[:, None] + stride_cn * offs_cn[None, :]
        )
    else:
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

# =============================================================================
# Kernel Invocation Wrapper
# =============================================================================


def _dump_selected_config(prof_dir, N, K, EM):
    """把 autotune 为当前用例选中的最优 config 落盘到 prof_dir/best_config.json。

    autotune 把 key -> 最优 Config 存在 fused_moe_kernel.cache 中。由于每个
    用例执行前已清空缓存（见 MOE_AUTOTUNE_RESET），此时缓存里就是本用例
    刚 benchmark 选出的最优参数。写盘后可与 profiling 抓到的性能对应，确认
    “抓到的确实是最优 config 的性能”。
    """
    try:
        import json as _json

        cache = getattr(fused_moe_kernel, "cache", None)
        if not isinstance(cache, dict) or not cache:
            return

        entries = []
        for key, cfg in cache.items():
            kwargs = getattr(cfg, "kwargs", None)
            if kwargs is None:
                kwargs = getattr(cfg, "all_kwargs", lambda: {})()
            entries.append({
                "key": [str(x) for x in (key if isinstance(key, tuple) else (key,))],
                "config": {k: kwargs[k] for k in kwargs},
                "config_str": str(cfg),
            })

        os.makedirs(prof_dir, exist_ok=True)
        with open(os.path.join(prof_dir, "best_config.json"), "w") as f:
            _json.dump(
                {"N": N, "K": K, "EM": EM, "selected": entries},
                f, indent=2, default=str,
            )
    except Exception as _e:
        # 记录失败不应影响主流程/性能采集
        print(f"[warn] dump best_config failed: {_e}")


def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor | None,
    C: torch.Tensor,
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    compute_type: tl.dtype,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    per_channel_quant: bool = False,
    block_shape: list | None = None,
    c_sorted: bool = False,
    filter_expert: bool = True,
    swap_ab: bool = False,
) -> None:
    """
    Invoke the autotuned fused_moe_kernel.

    The kernel uses @triton.autotune decorator with key=['N', 'K', 'EM']
    for automatic configuration selection.

    Args:
        A: Input tensor (tokens)
        B: Weight tensor (experts)
        bias: Optional bias tensor
        C: Output tensor
        A_scale: Optional activation scale tensor
        B_scale: Optional weight scale tensor
        topk_weights: Top-k weights for routing
        sorted_token_ids: Sorted token indices by expert
        expert_ids: Expert indices for each block
        num_tokens_post_padded: Number of tokens after padding
        mul_routed_weight: Whether to multiply by routed weights
        top_k: Number of top experts
        compute_type: Compute data type (e.g., tl.float16)
        use_fp8_w8a8: Use FP8 w8a8 quantization
        use_int8_w8a8: Use INT8 w8a8 quantization
        use_int8_w8a16: Use INT8 w8a16 quantization
        per_channel_quant: Use per-channel quantization
        block_shape: Optional [block_n, block_k] for block-wise quantization
        c_sorted: Whether output is sorted
        filter_expert: Whether to filter experts
        swap_ab: Whether to swap A and B (for SM90 optimization)
    """
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    N = B.shape[1]
    K = B.shape[2]
    EM = sorted_token_ids.shape[0]

    # Get block sizes for quantization
    group_n = 0 if block_shape is None else block_shape[0]
    group_k = 0 if block_shape is None else block_shape[1]

    # Define grid computation
    def grid(META):
        return (
            triton.cdiv(EM, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    even_Ks = K % 64 == 0  # Heuristic for even K

    # Set TensorDescriptor support
    a_desc = None
    b_desc = None
    # 每个用例可通过环境变量 MOE_PROF_DIR 指定独立的 profiling 输出目录，
    # 便于后续按用例自动抓取性能（见 extract_moe_perf.py）。默认沿用旧路径。
    PROF_BASE_DIR = os.environ.get("MOE_PROF_DIR", "./prof_fused_moe_dir")
    prof_dir = (f"{PROF_BASE_DIR}")
    experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False
        )

    # ================================================================
    # 保证“每个用例抓到的都是该用例最优 config 的性能”
    # ----------------------------------------------------------------
    # autotune 以 key=['N','K','EM'] 缓存最优 config。100 个用例在同一
    # 进程内连续跑，若两个用例 (N,K,EM) 相同（例如同一维度、仅量化模式/
    # dtype 不同），后者会直接复用前者缓存的 config，从而 **不会为当前
    # 用例重新 autotune**，抓到的就不是本用例最优参数。
    #
    # 解决：每次 invoke 前清空 autotuner 缓存，强制本用例重新 benchmark
    # 所有候选 config 并选最优。profiler 的 skip_first=1 恰好把触发
    # autotune 的首个 launch(i=0) 排除在 active 采集窗口之外，因此
    # active 窗口(i=3..22)测到的就是本用例选出的最优 config。
    #
    # 可用环境变量 MOE_AUTOTUNE_RESET=0 关闭该行为（例如想复用缓存加速）。
    # ================================================================
    if os.environ.get("MOE_AUTOTUNE_RESET", "1") == "1":
        _autotune_cache = getattr(fused_moe_kernel, "cache", None)
        if isinstance(_autotune_cache, dict):
            _autotune_cache.clear()

    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        with_stack=False,
        record_shapes=False,
        profile_memory=False,
        schedule=torch_npu.profiler.schedule(
            wait=1, warmup=1, active=20, repeat=1, skip_first=1
        ),
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir)
    ) as prof:
        for i in range(40):
             # Launch the autotuned kernel
            fused_moe_kernel[grid](
                A,
                a_desc,
                B,
                b_desc,
                bias,
                C,
                A_scale,
                B_scale,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                N,
                K,
                EM,
                topk_weights.numel(),
                A.stride(0),
                A.stride(1),
                B.stride(0),
                B.stride(2),
                B.stride(1),
                bias.stride(0) if bias is not None else 0,
                bias.stride(1) if bias is not None else 0,
                C.stride(-2),
                C.stride(-1),
                A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
                A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
                B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
                B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
                B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
                group_n,
                group_k,
                MUL_ROUTED_WEIGHT=mul_routed_weight,
                top_k=top_k,
                compute_type=compute_type,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                per_channel_quant=per_channel_quant,
                even_Ks=even_Ks,
                c_sorted=c_sorted,
                filter_expert=filter_expert,
                swap_ab=swap_ab,
                # myadd(编译优化)
                sync_solver=True,
                multibuffer=True,
                limit_auto_multi_buffer_of_local_buffer="no-limit",
                set_workspace_multibuffer=2,
                auto_blockify_size=2,
                enable_dynamic_cv_pipeline=True, enable_graph_optimize=False,
            )
            torch_npu.npu.synchronize()
            prof.step()
        prof.stop()

    # 记录本用例被 autotune 选中的最优 config（BLOCK_SIZE_N/K、GROUP_SIZE_M 等），
    # 与 profiling 抓到的性能一一对应，方便确认“抓到的是最优参数的性能”。
    _dump_selected_config(prof_dir, N, K, EM)
   


# Export list for easy importing
__all__ = [
    "fused_moe_kernel",
    "invoke_fused_moe_kernel",
    "write_zeros_to_output",
    "get_configs",
    "_support_tensor_descriptor",
]


if __name__ == "__main__":
    """
    Test the fused_moe_kernel on NPU/GPU.
    Note: This is a minimal test for structure verification.
    Actual usage requires proper input tensors prepared for MoE computation.
    """
    # Set device
    if hasattr(torch, 'npu') and torch.npu.is_available():
        device = torch.device('npu:0')
        try:
            torch.npu.set_device(0)
        except:
            pass  # May not be available in all triton-ascend versions
        print(f'Testing fused_moe_kernel structure on NPU...')
    elif hasattr(torch, 'cuda') and torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f'Testing fused_moe_kernel structure on CUDA...')
    else:
        device = torch.device('cpu')
        print(f'Testing fused_moe_kernel structure on CPU...')

    print(f'  Device: {device}')
    print(f'  Triton version: {triton.__version__}')
    print(f'  TensorDescriptor support: {_support_tensor_descriptor}')
    print(f'  ✓ File structure is ready for kernel compilation.')
    print(f'  Note: Actual kernel execution requires proper MoE input tensors.')

    # Show autotune configurations
    print(f'\n  Autotune Configuration Space:')
    configs = get_configs()
    print(f'  Total configs: {len(configs)}')
    print(f'  Autotune keys: ["N", "K", "EM"]')
    print(f'  Using @triton.autotune decorator')
    print(f'\n  Sample configurations:')
    for i, cfg in enumerate(configs[:5]):
        print(f'    [{i}] BLOCK_SIZE_M={cfg.kwargs["BLOCK_SIZE_M"]}, '
              f'BLOCK_SIZE_N={cfg.kwargs["BLOCK_SIZE_N"]}, '
              f'BLOCK_SIZE_K={cfg.kwargs["BLOCK_SIZE_K"]}, '
              f'GROUP_SIZE_M={cfg.kwargs["GROUP_SIZE_M"]}')
    print(f'    ... and {len(configs) - 5} more configs')

    print(f'\n  Kernel parameters:')
    print(f'    - Autotune: Enabled via @triton.autotune decorator')
    print(f'    - Tune keys: ["N", "K", "EM"]')
    print(f'    - num_stages: Not set (uses default)')
    print(f'    - num_warps: Not set (uses default)')
    print(f'    - BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: Tiling block sizes')
    print(f'    - GROUP_SIZE_M: Group size for L2 data reuse')
    print(f'    - compute_type: Data type for computation (float16/float32)')
    print(f'    - use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16: Quantization modes')
    print(f'    - filter_expert: Whether to filter experts by rank')
    print(f'    - swap_ab: Whether to swap A and B (for SM90 optimization)')
    print(f'    - c_sorted: Whether output is sorted by token_id')
    print(f'    - per_channel_quant: Per-channel quantization mode')
    print(f'    - group_n, group_k: Block-wise quantization block sizes')