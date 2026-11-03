"""Existing independent CPU equations; file-generation main omitted."""
import torch.nn.functional as F
import torch

def make_inputs():
    torch.manual_seed(42)
    '\n    测试 chunk_cumprod_householder_fwd_kernel\n    \n    使用官方测试配置: (B=3, T=1024, H=2, HQ=8, D=64, use_forget_gate=True, torch.bfloat16)\n    注意: chunk_cumprod_householder_fwd_kernel 只使用 H (不使用 HQ)，HQ 仅在 GQA 中使用\n    \n    关键: w1 和 w2 不能是随机张量！必须满足 Householder 变换的数学约束:\n      - w1 = F.normalize(w, dim=-1, p=2)  (单位归一化的 Householder 向量)\n      - w2 = A_inv * beta @ w  (通过 FLA 流水线计算)\n      - beta 在 [1.5, 2.0] 范围内\n    \n    输入生成流程 (等价于官方 parallel_path_attn 的 backward 中的调用):\n      1. w = F.normalize(torch.randn(...), dim=-1, p=2)  (float32, 单位归一化)\n      2. beta = torch.empty(...).uniform_(1.5, 2.0)      (float32)\n      3. k = torch.randn(...)                             (bfloat16)\n      4. A_raw = beta * (w @ w^T), 严格下三角\n      5. A_inv = (I + A_raw)^{-1}\n      6. w2 = (A_inv * beta) @ w\n      7. k_new_intra = k - (A_inv * beta @ tril(w @ k^T, -1))^T @ w\n      8. w1 = w.to(bfloat16), w2 = w2.to(bfloat16)\n      9. 调用 chunk_cumprod_householder_fwd_fn(k=k_new_intra, w1=w1, w2=w2, S=512, BT=64)\n    '
    B = 3
    T = 1024
    H = 2
    K = 64
    S = 512
    BS = 64
    BT = BS
    device = torch.device('cpu')
    dtype = torch.bfloat16
    torch.manual_seed(42)
    w = F.normalize(torch.randn((B, T, H, K), dtype=torch.float32, device=device), dim=-1, p=2)
    beta = torch.zeros((B, T, H), dtype=torch.float32, device=device).uniform_(1.5, 2.0)
    k = torch.randn((B, T, H, K), dtype=torch.float32, device=device)
    w2_fp32, k_new_intra_fp32 = compute_w2_and_k_new_ref(w, beta, k, BS)
    w1 = w.to(dtype)
    w2 = w2_fp32.to(dtype)
    k_input = k_new_intra_fp32.to(dtype)
    input_data = {'k': k_input.cpu(), 'w1': w1.cpu(), 'w2': w2.cpu(), 'S': S, 'BT': BT, 'w_fp32': w.cpu(), 'beta_fp32': beta.cpu(), 'k_original_fp32': k.cpu(), 'w2_fp32': w2_fp32.cpu(), 'k_new_intra_fp32': k_new_intra_fp32.cpu()}
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            input_data = {'k': k_input.cpu(), 'w1': w1.cpu(), 'w2': w2.cpu(), 'S': S, 'BT': BT, 'w_fp32': w.cpu(), 'beta_fp32': beta.cpu(), 'k_original_fp32': k.cpu(), 'w2_fp32': w2_fp32.cpu(), 'k_new_intra_fp32': k_new_intra_fp32.cpu()}
        except NameError:
            pass
    return locals()

def compute_w2_and_k_new_ref(w: torch.Tensor, beta: torch.Tensor, k: torch.Tensor, BS: int):
    """
    使用 PyTorch 参考实现计算 w2 和 k_new_intra（intra-chunk transformed key）。
    
    这等价于 FLA 官方流水线中的:
      A = chunk_scaled_dot_kkt_fwd(k=w, beta=beta, chunk_size=BS)
      A_inv = solve_tril(A=A)
      w2, k_new = intra_chunk_preprocess_fwd_fn(... 中的 w2 和 k_new 部分)
    
    数学原理:
      A_raw[t,s] = beta[t] * (w[t] @ w[s])  for t > s  (严格下三角)
      A_inv = (I + A_raw)^{-1}               (下三角矩阵求逆)
      w2[t] = sum_{s<=t} A_inv[t,s] * beta[s] * w[s]   (即 A_inv * beta @ w)
      k_new = k - (A_inv * beta @ tril(w @ k^T, -1))^T @ w
    
    Args:
        w: (B, T, H, K) 单位归一化的 Householder 向量, float32
        beta: (B, T, H) Householder beta 参数, float32
        k: (B, T, H, K) 原始 key 张量, float32
        BS: 小 chunk 大小 (与 kernel 的 BT 参数一致)
    
    Returns:
        w2: (B, T, H, K) 变换后的 Householder 向量
        k_new: (B, T, H, K) intra-chunk 变换后的 key
    """
    B, T, H, K_dim = w.shape
    assert T % BS == 0, f'T={T} must be divisible by BS={BS}'
    NT = T // BS
    w2 = torch.zeros_like(w)
    k_new = torch.zeros_like(k)
    for b in range(B):
        for h in range(H):
            for t_idx in range(NT):
                w_chunk = w[b, t_idx * BS:(t_idx + 1) * BS, h, :]
                beta_chunk = beta[b, t_idx * BS:(t_idx + 1) * BS, h]
                k_chunk = k[b, t_idx * BS:(t_idx + 1) * BS, h, :]
                A_raw = beta_chunk[:, None] * (w_chunk @ w_chunk.T)
                A_raw = torch.tril(A_raw, diagonal=-1)
                L = torch.eye(BS, device=w.device, dtype=w.dtype) + A_raw
                A_inv = torch.linalg.solve(L, torch.eye(BS, device=w.device, dtype=w.dtype))
                w2_chunk = A_inv * beta_chunk[None, :] @ w_chunk
                w2[b, t_idx * BS:(t_idx + 1) * BS, h, :] = w2_chunk
                wbk = w_chunk @ k_chunk.T
                wbk = torch.tril(wbk, diagonal=-1)
                T_wbk = A_inv * beta_chunk[None, :] @ wbk
                k_new_chunk = k_chunk - T_wbk.T @ w_chunk
                k_new[b, t_idx * BS:(t_idx + 1) * BS, h, :] = k_new_chunk
    return (w2, k_new)
