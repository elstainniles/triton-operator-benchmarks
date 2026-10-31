import math
import torch
import torch_npu
from typing import Optional
import triton
import triton.language as tl


BLOCK_K = 32


def ceil_log(x: int, b: int) -> int:
    return math.ceil(math.log(x, b))


def construct_binary_level_mask(level, T):
    if level == 0:
        return torch.diag(torch.ones(T, dtype=torch.bool))

    indices = torch.cartesian_prod(torch.arange(T), torch.arange(T))

    mask = torch.where(
        torch.logical_and(
            torch.logical_and(
                indices[:, 0] % (1 << level) >= (1 << (level - 1)),
                indices[:, 1] + (1 << (level - 1))
                >= indices[:, 0] - (indices[:, 0] % (1 << (level - 1))),
            ),
            indices[:, 1] < indices[:, 0] - (indices[:, 0] % (1 << (level - 1))),
        ).view(T, T),
        1,
        0,
    )

    return mask


def level_lut(BT, device):
    lut = torch.zeros((BT, BT), dtype=torch.int32, device=device)
    for level in range(1, ceil_log(BT, 2) + 1):
        mask = construct_binary_level_mask(level, BT).to(device)
        lut = torch.where(mask.to(torch.bool), level, lut)
    return lut


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
    },
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BLOCK_K}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V"],
)
@triton.jit(do_not_specialize=["T"])
def chunkwise_fwd_kernel(
    q,
    k,
    v,
    g,
    level_scales,
    llut,
    o,
    h0,
    ht,
    offsets,
    new_offsets,
    cu_seqlens,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    L_IN: tl.constexpr,
    L_OUT: tl.constexpr,
    MIN_LEVEL: tl.constexpr,
    MAX_LEVEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    p_llut = tl.make_block_ptr(llut, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0))
    b_llut = tl.load(p_llut, boundary_check=(0, 1))
    i_k = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_i = tl.arange(0, BT)

    num_intra_levels = (tl.log2(float(BT))).to(tl.int32) + 1
    i_idx = o_i[:, None]
    j_idx = o_i[None, :]

    KV_0_CREATED = MIN_LEVEL <= 1 and MAX_LEVEL >= 0
    KV_1_CREATED = MIN_LEVEL <= 2 and MAX_LEVEL >= 0
    KV_2_CREATED = MIN_LEVEL <= 3 and MAX_LEVEL >= 1
    KV_3_CREATED = MIN_LEVEL <= 4 and MAX_LEVEL >= 2
    KV_4_CREATED = MIN_LEVEL <= 5 and MAX_LEVEL >= 3
    KV_5_CREATED = MIN_LEVEL <= 6 and MAX_LEVEL >= 4
    KV_6_CREATED = MIN_LEVEL <= 7 and MAX_LEVEL >= 5
    KV_7_CREATED = MIN_LEVEL <= 8 and MAX_LEVEL >= 6
    KV_8_CREATED = MIN_LEVEL <= 9 and MAX_LEVEL >= 7
    KV_9_CREATED = MIN_LEVEL <= 10 and MAX_LEVEL >= 8
    KV_10_CREATED = MIN_LEVEL <= 11 and MAX_LEVEL >= 9
    KV_11_CREATED = MIN_LEVEL <= 12 and MAX_LEVEL >= 10

    kv_0 = tl.zeros([BK, V], dtype=tl.float32)
    kv_1 = tl.zeros([BK, V], dtype=tl.float32)
    kv_2 = tl.zeros([BK, V], dtype=tl.float32)
    kv_3 = tl.zeros([BK, V], dtype=tl.float32)
    kv_4 = tl.zeros([BK, V], dtype=tl.float32)
    kv_5 = tl.zeros([BK, V], dtype=tl.float32)
    kv_6 = tl.zeros([BK, V], dtype=tl.float32)
    kv_7 = tl.zeros([BK, V], dtype=tl.float32)
    kv_8 = tl.zeros([BK, V], dtype=tl.float32)
    kv_9 = tl.zeros([BK, V], dtype=tl.float32)
    kv_10 = tl.zeros([BK, V], dtype=tl.float32)
    kv_11 = tl.zeros([BK, V], dtype=tl.float32)

    offset = 0
    first_chunk_index = 0
    if USE_INITIAL_STATE:
        offset = tl.load(offsets + i_n)

        first_chunk_index = offset // BT

        if KV_0_CREATED and (first_chunk_index & 1 > 0):
            p_kv_0 = tl.make_block_ptr(
                h0 + ((i_n * L_IN + 0) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )
            kv_0 = tl.load(p_kv_0, boundary_check=(0, 1))
        if KV_1_CREATED and (first_chunk_index & 2 > 0):
            p_kv_1 = tl.make_block_ptr(
                h0 + ((i_n * L_IN + 1) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )
            kv_1 = tl.load(p_kv_1, boundary_check=(0, 1))
        if KV_2_CREATED and (first_chunk_index & 4 > 0):
            p_kv_2 = tl.make_block_ptr(
                h0 + ((i_n * L_IN + 2) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )
            kv_2 = tl.load(p_kv_2, boundary_check=(0, 1))
        if KV_3_CREATED and (first_chunk_index & 8 > 0):
            p_kv_3 = tl.make_block_ptr(
                h0 + ((i_n * L_IN + 3) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )
            kv_3 = tl.load(p_kv_3, boundary_check=(0, 1))
        if KV_4_CREATED and (first_chunk_index & 16 > 0):
            p_kv_4 = tl.make_block_ptr(
                h0 + ((i_n * L_IN + 4) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )
            kv_4 = tl.load(p_kv_4, boundary_check=(0, 1))
        if KV_5_CREATED and (first_chunk_index & 32 > 0):
            p_kv_5 = tl.make_block_ptr(
                h0 + ((i_n * L_IN + 5) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )
            kv_5 = tl.load(p_kv_5, boundary_check=(0, 1))
        if KV_6_CREATED and (first_chunk_index & 64 > 0):
            p_kv_6 = tl.make_block_ptr(
                h0 + ((i_n * L_IN + 6) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )
            kv_6 = tl.load(p_kv_6, boundary_check=(0, 1))
        if KV_7_CREATED and (first_chunk_index & 128 > 0):
            p_kv_7 = tl.make_block_ptr(
                h0 + ((i_n * L_IN + 7) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )
            kv_7 = tl.load(p_kv_7, boundary_check=(0, 1))
        if KV_8_CREATED and (first_chunk_index & 256 > 0):
            p_kv_8 = tl.make_block_ptr(
                h0 + ((i_n * L_IN + 8) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )
            kv_8 = tl.load(p_kv_8, boundary_check=(0, 1))
        if KV_9_CREATED and (first_chunk_index & 512 > 0):
            p_kv_9 = tl.make_block_ptr(
                h0 + ((i_n * L_IN + 9) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )
            kv_9 = tl.load(p_kv_9, boundary_check=(0, 1))
        if KV_10_CREATED and (first_chunk_index & 1024 > 0):
            p_kv_10 = tl.make_block_ptr(
                h0 + ((i_n * L_IN + 10) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )
            kv_10 = tl.load(p_kv_10, boundary_check=(0, 1))
        if KV_11_CREATED and (first_chunk_index & 2048 > 0):
            p_kv_11 = tl.make_block_ptr(
                h0 + ((i_n * L_IN + 11) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )
            kv_11 = tl.load(p_kv_11, boundary_check=(0, 1))

    NT = tl.cdiv(T, BT)
    output_offset = -1 * (offset % BT)
    for i_t in range(NT):
        b_h_ptrs = level_scales + ((bos + i_t * BT + i_idx) * H + i_h) * L + b_llut
        b_h = tl.load(b_h_ptrs, mask=i_idx >= j_idx)

        p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_q = tl.make_block_ptr(
            q + bos * K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0),
        )
        p_k = tl.make_block_ptr(
            k + bos * K, (K, T), (1, K), (i_k * BK, i_t * BT), (BK, BT), (0, 1),
        )
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, 0),
            (BT, V),
            (1, 0),
        )
        p_o = tl.make_block_ptr(
            o + ((bos * H + i_h) * (K // BK) + i_k) * V,
            (T, V),
            (H * (K // BK) * V, 1),
            (i_t * BT + output_offset, 0),
            (BT, V),
            (1, 0),
        )

        b_g = tl.load(p_g, boundary_check=(0,))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))

        m_t = i_t * BT + o_i < T
        b_s = (tl.dot(b_q, b_k) * tl.where((i_idx >= j_idx) & m_t[:, None] & m_t[None, :], tl.exp(b_g[:, None] - b_g[None, :]), 0)).to(
            b_q.dtype,
        ) * b_h

        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_o = tl.zeros((BT, V), dtype=tl.float32)
        if MIN_LEVEL == 0:
            b_o += tl.dot(b_s, b_v)

        chunk_index = (
            first_chunk_index + i_t
        )

        if MIN_LEVEL <= 0 and MAX_LEVEL >= 0:
            if chunk_index & 1:
                p_l = tl.make_block_ptr(
                    level_scales + (bos * H + i_h) * L,
                    (T, L),
                    (H * L, 1),
                    (i_t * BT, num_intra_levels),
                    (BT, 1),
                    (1, 0),
                )
                b_l = tl.load(p_l, boundary_check=(0, 1))
                b_o += tl.dot((b_l * b_q), kv_0.to(b_q.dtype)) * tl.exp(b_g)[:, None]
        if MIN_LEVEL <= 1 and MAX_LEVEL >= 1:
            if chunk_index & 2:
                p_l = tl.make_block_ptr(
                    level_scales + (bos * H + i_h) * L,
                    (T, L),
                    (H * L, 1),
                    (i_t * BT, num_intra_levels + 1),
                    (BT, 1),
                    (1, 0),
                )
                b_l = tl.load(p_l, boundary_check=(0, 1))
                b_o += tl.dot((b_l * b_q), kv_1.to(b_q.dtype)) * tl.exp(b_g)[:, None]
        if MIN_LEVEL <= 2 and MAX_LEVEL >= 2:
            if chunk_index & 4:
                p_l = tl.make_block_ptr(
                    level_scales + (bos * H + i_h) * L,
                    (T, L),
                    (H * L, 1),
                    (i_t * BT, num_intra_levels + 2),
                    (BT, 1),
                    (1, 0),
                )
                b_l = tl.load(p_l, boundary_check=(0, 1))
                b_o += tl.dot((b_l * b_q), kv_2.to(b_q.dtype)) * tl.exp(b_g)[:, None]
        if MIN_LEVEL <= 3 and MAX_LEVEL >= 3:
            if chunk_index & 8:
                p_l = tl.make_block_ptr(
                    level_scales + (bos * H + i_h) * L,
                    (T, L),
                    (H * L, 1),
                    (i_t * BT, num_intra_levels + 3),
                    (BT, 1),
                    (1, 0),
                )
                b_l = tl.load(p_l, boundary_check=(0, 1))
                b_o += tl.dot((b_l * b_q), kv_3.to(b_q.dtype)) * tl.exp(b_g)[:, None]
        if MIN_LEVEL <= 4 and MAX_LEVEL >= 4:
            if chunk_index & 16:
                p_l = tl.make_block_ptr(
                    level_scales + (bos * H + i_h) * L,
                    (T, L),
                    (H * L, 1),
                    (i_t * BT, num_intra_levels + 4),
                    (BT, 1),
                    (1, 0),
                )
                b_l = tl.load(p_l, boundary_check=(0, 1))
                b_o += tl.dot((b_l * b_q), kv_4.to(b_q.dtype)) * tl.exp(b_g)[:, None]
        if MIN_LEVEL <= 5 and MAX_LEVEL >= 5:
            if chunk_index & 32:
                p_l = tl.make_block_ptr(
                    level_scales + (bos * H + i_h) * L,
                    (T, L),
                    (H * L, 1),
                    (i_t * BT, num_intra_levels + 5),
                    (BT, 1),
                    (1, 0),
                )
                b_l = tl.load(p_l, boundary_check=(0, 1))
                b_o += tl.dot((b_l * b_q), kv_5.to(b_q.dtype)) * tl.exp(b_g)[:, None]
        if MIN_LEVEL <= 6 and MAX_LEVEL >= 6:
            if chunk_index & 64:
                p_l = tl.make_block_ptr(
                    level_scales + (bos * H + i_h) * L,
                    (T, L),
                    (H * L, 1),
                    (i_t * BT, num_intra_levels + 6),
                    (BT, 1),
                    (1, 0),
                )
                b_l = tl.load(p_l, boundary_check=(0, 1))
                b_o += tl.dot((b_l * b_q), kv_6.to(b_q.dtype)) * tl.exp(b_g)[:, None]
        if MIN_LEVEL <= 7 and MAX_LEVEL >= 7:
            if chunk_index & 128:
                p_l = tl.make_block_ptr(
                    level_scales + (bos * H + i_h) * L,
                    (T, L),
                    (H * L, 1),
                    (i_t * BT, num_intra_levels + 7),
                    (BT, 1),
                    (1, 0),
                )
                b_l = tl.load(p_l, boundary_check=(0, 1))
                b_o += tl.dot((b_l * b_q), kv_7.to(b_q.dtype)) * tl.exp(b_g)[:, None]
        if MIN_LEVEL <= 8 and MAX_LEVEL >= 8:
            if chunk_index & 256:
                p_l = tl.make_block_ptr(
                    level_scales + (bos * H + i_h) * L,
                    (T, L),
                    (H * L, 1),
                    (i_t * BT, num_intra_levels + 8),
                    (BT, 1),
                    (1, 0),
                )
                b_l = tl.load(p_l, boundary_check=(0, 1))
                b_o += tl.dot((b_l * b_q), kv_8.to(b_q.dtype)) * tl.exp(b_g)[:, None]
        if MIN_LEVEL <= 9 and MAX_LEVEL >= 9:
            if chunk_index & 512:
                p_l = tl.make_block_ptr(
                    level_scales + (bos * H + i_h) * L,
                    (T, L),
                    (H * L, 1),
                    (i_t * BT, num_intra_levels + 9),
                    (BT, 1),
                    (1, 0),
                )
                b_l = tl.load(p_l, boundary_check=(0, 1))
                b_o += tl.dot((b_l * b_q), kv_9.to(b_q.dtype)) * tl.exp(b_g)[:, None]
        if MIN_LEVEL <= 10 and MAX_LEVEL >= 10:
            if chunk_index & 1024:
                p_l = tl.make_block_ptr(
                    level_scales + (bos * H + i_h) * L,
                    (T, L),
                    (H * L, 1),
                    (i_t * BT, num_intra_levels + 10),
                    (BT, 1),
                    (1, 0),
                )
                b_l = tl.load(p_l, boundary_check=(0, 1))
                b_o += tl.dot((b_l * b_q), kv_10.to(b_q.dtype)) * tl.exp(b_g)[:, None]
        if MIN_LEVEL <= 11 and MAX_LEVEL >= 11:
            if chunk_index & 2048:
                p_l = tl.make_block_ptr(
                    level_scales + (bos * H + i_h) * L,
                    (T, L),
                    (H * L, 1),
                    (i_t * BT, num_intra_levels + 11),
                    (BT, 1),
                    (1, 0),
                )
                b_l = tl.load(p_l, boundary_check=(0, 1))
                b_o += tl.dot((b_l * b_q), kv_11.to(b_q.dtype)) * tl.exp(b_g)[:, None]

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        if i_t < NT - 1 or T % BT == 0:
            last_idx = min((i_t + 1) * BT, T) - 1
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            if KV_0_CREATED:
                kv_0 *= tl.exp(b_g_last)
            if KV_1_CREATED:
                kv_1 *= tl.exp(b_g_last)
            if KV_2_CREATED:
                kv_2 *= tl.exp(b_g_last)
            if KV_3_CREATED:
                kv_3 *= tl.exp(b_g_last)
            if KV_4_CREATED:
                kv_4 *= tl.exp(b_g_last)
            if KV_5_CREATED:
                kv_5 *= tl.exp(b_g_last)
            if KV_6_CREATED:
                kv_6 *= tl.exp(b_g_last)
            if KV_7_CREATED:
                kv_7 *= tl.exp(b_g_last)
            if KV_8_CREATED:
                kv_8 *= tl.exp(b_g_last)
            if KV_9_CREATED:
                kv_9 *= tl.exp(b_g_last)
            if KV_10_CREATED:
                kv_10 *= tl.exp(b_g_last)
            if KV_11_CREATED:
                kv_11 *= tl.exp(b_g_last)

            b_v = (b_v * tl.exp(b_g_last - b_g)[:, None]).to(b_v.dtype)
            if MIN_LEVEL <= 1:
                kv_0 += tl.dot(b_k, b_v)
            elif MIN_LEVEL == 2:
                kv_1 += tl.dot(b_k, b_v)
            elif MIN_LEVEL == 3:
                kv_2 += tl.dot(b_k, b_v)
            elif MIN_LEVEL == 4:
                kv_3 += tl.dot(b_k, b_v)
            elif MIN_LEVEL == 5:
                kv_4 += tl.dot(b_k, b_v)
            elif MIN_LEVEL == 6:
                kv_5 += tl.dot(b_k, b_v)
            elif MIN_LEVEL == 7:
                kv_6 += tl.dot(b_k, b_v)
            elif MIN_LEVEL == 8:
                kv_7 += tl.dot(b_k, b_v)
            elif MIN_LEVEL == 9:
                kv_8 += tl.dot(b_k, b_v)
            elif MIN_LEVEL == 10:
                kv_9 += tl.dot(b_k, b_v)
            elif MIN_LEVEL == 11:
                kv_10 += tl.dot(b_k, b_v)

            check_value = (~chunk_index & (chunk_index + 1)) - 1

            if MIN_LEVEL <= 1 and MAX_LEVEL >= 0:
                if check_value & 1:
                    kv_1 += kv_0
                    kv_0 = tl.zeros([BK, V], dtype=tl.float32)
            if MIN_LEVEL <= 2 and MAX_LEVEL >= 1:
                if check_value & 2:
                    kv_2 += kv_1
                    kv_1 = tl.zeros([BK, V], dtype=tl.float32)
            if MIN_LEVEL <= 3 and MAX_LEVEL >= 2:
                if check_value & 4:
                    kv_3 += kv_2
                    kv_2 = tl.zeros([BK, V], dtype=tl.float32)
            if MIN_LEVEL <= 4 and MAX_LEVEL >= 3:
                if check_value & 8:
                    kv_4 += kv_3
                    kv_3 = tl.zeros([BK, V], dtype=tl.float32)
            if MIN_LEVEL <= 5 and MAX_LEVEL >= 4:
                if check_value & 16:
                    kv_5 += kv_4
                    kv_4 = tl.zeros([BK, V], dtype=tl.float32)
            if MIN_LEVEL <= 6 and MAX_LEVEL >= 5:
                if check_value & 32:
                    kv_6 += kv_5
                    kv_5 = tl.zeros([BK, V], dtype=tl.float32)
            if MIN_LEVEL <= 7 and MAX_LEVEL >= 6:
                if check_value & 64:
                    kv_7 += kv_6
                    kv_6 = tl.zeros([BK, V], dtype=tl.float32)
            if MIN_LEVEL <= 8 and MAX_LEVEL >= 7:
                if check_value & 128:
                    kv_8 += kv_7
                    kv_7 = tl.zeros([BK, V], dtype=tl.float32)
            if MIN_LEVEL <= 9 and MAX_LEVEL >= 8:
                if check_value & 256:
                    kv_9 += kv_8
                    kv_8 = tl.zeros([BK, V], dtype=tl.float32)
            if MIN_LEVEL <= 10 and MAX_LEVEL >= 9:
                if check_value & 512:
                    kv_10 += kv_9
                    kv_9 = tl.zeros([BK, V], dtype=tl.float32)
            if MIN_LEVEL <= 11 and MAX_LEVEL >= 10:
                if check_value & 1024:
                    kv_11 += kv_10
                    kv_10 = tl.zeros([BK, V], dtype=tl.float32)

    chunk_index = offset // BT + T // BT

    if STORE_FINAL_STATE:
        if (MIN_LEVEL <= 0 and MAX_LEVEL >= 0) and (chunk_index & 1 > 0):
            p_kv = tl.make_block_ptr(
                ht + ((i_n * L_OUT + 0) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )
            tl.store(p_kv, kv_0, boundary_check=(0, 1))
        if (MIN_LEVEL <= 1 and MAX_LEVEL >= 1) and (chunk_index & 2 > 0):
            p_kv = tl.make_block_ptr(
                ht + ((i_n * L_OUT + 1) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )
            tl.store(p_kv, kv_1, boundary_check=(0, 1))
        if (MIN_LEVEL <= 2 and MAX_LEVEL >= 2) and (chunk_index & 4 > 0):
            p_kv = tl.make_block_ptr(
                ht + ((i_n * L_OUT + 2) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )
            tl.store(p_kv, kv_2, boundary_check=(0, 1))
        if (MIN_LEVEL <= 3 and MAX_LEVEL >= 3) and (chunk_index & 8 > 0):
            p_kv = tl.make_block_ptr(
                ht + ((i_n * L_OUT + 3) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )
            tl.store(p_kv, kv_3, boundary_check=(0, 1))
        if (MIN_LEVEL <= 4 and MAX_LEVEL >= 4) and (chunk_index & 16 > 0):
            p_kv = tl.make_block_ptr(
                ht + ((i_n * L_OUT + 4) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )
            tl.store(p_kv, kv_4, boundary_check=(0, 1))
        if (MIN_LEVEL <= 5 and MAX_LEVEL >= 5) and (chunk_index & 32 > 0):
            p_kv = tl.make_block_ptr(
                ht + ((i_n * L_OUT + 5) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )
            tl.store(p_kv, kv_5, boundary_check=(0, 1))
        if (MIN_LEVEL <= 6 and MAX_LEVEL >= 6) and (chunk_index & 64 > 0):
            p_kv = tl.make_block_ptr(
                ht + ((i_n * L_OUT + 6) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )
            tl.store(p_kv, kv_6, boundary_check=(0, 1))
        if (MIN_LEVEL <= 7 and MAX_LEVEL >= 7) and (chunk_index & 128 > 0):
            p_kv = tl.make_block_ptr(
                ht + ((i_n * L_OUT + 7) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )
            tl.store(p_kv, kv_7, boundary_check=(0, 1))
        if (MIN_LEVEL <= 8 and MAX_LEVEL >= 8) and (chunk_index & 256 > 0):
            p_kv = tl.make_block_ptr(
                ht + ((i_n * L_OUT + 8) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )
            tl.store(p_kv, kv_8, boundary_check=(0, 1))
        if (MIN_LEVEL <= 9 and MAX_LEVEL >= 9) and (chunk_index & 512 > 0):
            p_kv = tl.make_block_ptr(
                ht + ((i_n * L_OUT + 9) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )
            tl.store(p_kv, kv_9, boundary_check=(0, 1))
        if (MIN_LEVEL <= 10 and MAX_LEVEL >= 10) and (chunk_index & 1024 > 0):
            p_kv = tl.make_block_ptr(
                ht + ((i_n * L_OUT + 10) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )
            tl.store(p_kv, kv_10, boundary_check=(0, 1))
        if (MIN_LEVEL <= 11 and MAX_LEVEL >= 11) and (chunk_index & 2048 > 0):
            p_kv = tl.make_block_ptr(
                ht + ((i_n * L_OUT + 11) * H + i_h) * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, 0),
                (BK, V),
                (1, 0),
            )
            tl.store(p_kv, kv_11, boundary_check=(0, 1))

        tl.store(new_offsets + i_n, (offset // BT) * BT + T)

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


def compare_data_precision(dict_ref, dict_cal, device_type):
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


def convert_tensor_with_device_type(indata, device_type):
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

def cal_precision(input: torch.Tensor, golden: torch.Tensor):
    relative_max = torch.max(torch.abs(input - golden) / (torch.abs(golden) + float(1e-7)))
    relative_mean = torch.mean(torch.abs(input - golden) / (torch.abs(golden) + float(1e-7)))
    rmse = torch.sqrt(torch.mean((input - golden) ** 2))
    mask = (torch.abs(golden) < float(2e-14)) & (torch.abs(input - golden) > float(2e-30))
    error = torch.abs(input - golden)
    total_error = torch.sum(error[mask])
    return {
        'max_relative_error': relative_max.item(),
        'mean_relative_error': relative_mean.item(),
        'rmse': rmse.item(),
        'small_val_error': total_error.item(),
    }


def fp64_precision_compare(npu_out, gpu_out, fp64_data):
    print("\n" + "=" * 60)
    print("Third-party fp64 precision comparison")
    print("=" * 60)
    for key in fp64_data.keys():
        if not isinstance(fp64_data[key], torch.Tensor):
            continue
        if fp64_data[key].is_floating_point():
            fp64_val = fp64_data[key].to(torch.float64).cpu()
            npu_val = npu_out[key].to(torch.float64).cpu()
            gpu_val = gpu_out[key].to(torch.float64).cpu()

            npu_metrics = cal_precision(npu_val, fp64_val)
            gpu_metrics = cal_precision(gpu_val, fp64_val)

            print(f"\n--- Key: {key} ---")
            print(f"{'Metric':<25} {'NPU vs FP64':<20} {'GPU vs FP64':<20} {'NPU/GPU Ratio':<15}")
            print("-" * 80)
            for metric_name in npu_metrics:
                npu_v = npu_metrics[metric_name]
                gpu_v = gpu_metrics[metric_name]
                if gpu_v != 0:
                    ratio = npu_v / gpu_v
                    ratio_str = f"{ratio:.6f}"
                else:
                    ratio_str = "N/A (gpu=0)"
                print(f"{metric_name:<25} {npu_v:<20.10f} {gpu_v:<20.10f} {ratio_str:<15}")
                if metric_name == "max_relative_error":
                     assert ratio < 2.0, f"Ratio {ratio} exceeds threshold 2.0 for {metric_name}"
                elif metric_name != "small_val_error":
                     assert ratio < 1.2, f"Ratio  exceeds threshold 1.2 for "



def test_chunkwise_fwd_kernel():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunkwise_fwd_kernel/test_chunkwise_fwd_kernel.pt'
    data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)

    device = torch.device('npu')
    input_data = convert_tensor_with_device_type(data["input_data"], device_type='npu')

    import math
    B = input_data['B']
    T = input_data['T']
    H = input_data['H']
    G = input_data['G']
    K = input_data['K']
    V = input_data['V']
    L = input_data['L']
    BT = input_data['BT']
    MAX_LEVEL = input_data['MAX_LEVEL']

    q = input_data['q']
    k = input_data['k']
    v = input_data['v']
    g = input_data['g']
    level_scales = input_data['level_scales']
    llut = level_lut(BT, device)

    o = torch.zeros((B, T, H, (K // BLOCK_K), V), dtype=q.dtype, device=device)
    h0 = input_data['h0']
    ht = input_data['ht']
    offsets = input_data['offsets']
    new_offsets = torch.zeros((B,), dtype=torch.int32, device=device)

    def grid(meta):
        return (triton.cdiv(K, meta["BK"]), B * H)

    chunkwise_fwd_kernel[grid](
        q=q, k=k, v=v, g=g, level_scales=level_scales, llut=llut,
        o=o, h0=h0, ht=ht, offsets=offsets, new_offsets=new_offsets,
        cu_seqlens=input_data['cu_seqlens'],
        T=T, H=H, K=K, V=V, L=L, BT=BT,
        L_IN=None, L_OUT=None, MIN_LEVEL=0, MAX_LEVEL=MAX_LEVEL,
    )

    npu_out = {
        'o': o,
        'new_offsets': new_offsets,
    }

    try:
        compare_data_precision(data["gpu_output"], npu_out, device_type='cpu')
        print("Test chunkwise_fwd_kernel passed!")
    except (AssertionError, ValueError, RuntimeError) as e:
        print(f"Precision comparison failed: {e}")
        print("Falling back to fp64 third-party precision comparison...")
        fp64_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunkwise_fwd_kernel/test_chunkwise_fwd_kernel_fp64.pt'
        fp64_data = torch.load(fp64_path, map_location=torch.device('cpu'), weights_only=False)
        fp64_precision_compare(npu_out, data["gpu_output"], fp64_data["gpu_output"])


if __name__ == "__main__":
    test_chunkwise_fwd_kernel()
