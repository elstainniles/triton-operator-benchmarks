"""Existing independent CPU equations; file-generation main omitted."""
import torch

def reference(d):
    B, T, H, K, V, BT = [d[k] for k in ('B', 'T', 'H', 'K', 'V', 'chunk_size')]
    N = (T + BT - 1) // BT
    assert T % BT == 0
    h = d['h']
    if h.ndim == 4 and h.shape[0] != B * (T // BT):
        h = h.unsqueeze(0).expand(B, -1, -1, -1, -1).contiguous().reshape(-1, h.shape[-2], h.shape[-1])
    hh = h.double().reshape(B, N, H, K, V)

    def blocks(name):
        return d[name].reshape(B, N, BT, H, -1).permute(0, 1, 3, 2, 4)
    q = blocks('qg').double()
    v = blocks('v').double()
    vn = blocks('v_new').double()
    a = blocks('A_qk').tril().to(d['v'].dtype).double()
    ab = blocks('A_qb').tril().to(d['v_new'].dtype).double()
    out = q @ hh + a @ v + ab @ vn
    alternate = torch.einsum('bnhik,bnhkv->bnhiv', q, hh) + torch.einsum('bnhij,bnhjv->bnhiv', a, v) + torch.einsum('bnhij,bnhjv->bnhiv', ab, vn)
    torch.testing.assert_close(out, alternate, rtol=1e-12, atol=1e-12)
    result = out.permute(0, 1, 3, 2, 4).contiguous().reshape(B, T, H, V).to(d['v'].dtype)
    assert torch.isfinite(result).all()
    return (result, dict(matrix_einsum_max_abs=float((out - alternate).abs().max()), finite=True, original_h_broadcast_shape=list(h.shape), output_shape=list(result.shape), output_dtype=str(result.dtype)))
