"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def tiled(x, bt):
    b, t, h, d = x.shape
    assert t % bt == 0
    return x.double().reshape(b, t // bt, bt, h, d).transpose(2, 3)

def untiled(x):
    b, n, h, t, d = x.shape
    return x.transpose(2, 3).reshape(b, n * t, h, d).contiguous()

def reference(data):
    """FP64 matrix equations for the supplied tensors, not an output from Triton."""
    assert data['NG'] == 1 and data['K'] == 64 and (data['V'] == 64) and (data['h'] is not None)
    bt = data['BT']
    scale = data['scale']
    q, k, v, g, do, o = (tiled(data[key], bt) for key in ('q', 'k', 'v', 'g', 'do', 'o'))
    h = data['h'].double()
    dg_in = tiled(data['dg'], bt)
    b, n, heads, t, keys = q.shape
    values = v.shape[-1]
    do_exp = do * g.exp()
    da = torch.tril(do_exp @ (v * (-g).exp()).transpose(-1, -2) * scale)
    A = torch.tril(q * scale @ k.transpose(-1, -2))
    assert data['dht'] is None and data['h0'] is None
    state = torch.zeros(b, heads, keys, values, dtype=torch.float64)
    dh = torch.empty_like(h)
    for it in range(n - 1, -1, -1):
        dh[:, it] = state
        state = state + q[:, it].transpose(-1, -2) @ do_exp[:, it]
    gate = (g[..., -1:, :] - g).exp()
    dv_inter = k @ dh * gate
    dq = do_exp @ h.transpose(-1, -2) * scale + da @ k
    dk = v * gate @ dh.transpose(-1, -2) + da.transpose(-1, -2) @ q
    dgv = dg_in + ((h * dh).sum(-2) * g[..., -1, :].exp() + (dv_inter * v).sum(-2))[..., None, :]
    dv_intra = A.transpose(-1, -2) @ do_exp * (-g).exp()
    dv = dv_inter + dv_intra
    dgg = o * do - v * dv
    dg = dgv + dgg.flip(-2).cumsum(-2).flip(-2)
    gold = {key: untiled(value) for key, value in dict(dq=dq, dk=dk, dv=dv, dg=dg, dA=da, A=A).items()}
    q0 = q[0, 0, 0].clone().requires_grad_()
    k0 = k[0, 0, 0].clone().requires_grad_()
    v0 = v[0, 0, 0].clone().requires_grad_()
    g0 = g[0, 0, 0].clone().requires_grad_()
    causal = torch.ones(bt, bt, dtype=torch.float64).tril()
    kernel = q0 @ k0.T * scale * causal
    output = torch.einsum('ij,jv,ijv->iv', kernel, v0, (g0[:, None, :] - g0[None, :, :]).exp())
    loss = (output * do[0, 0, 0]).sum()
    aq, ak, av, ag = torch.autograd.grad(loss, (q0, k0, v0, g0))
    expected = [da[0, 0, 0] @ k0.detach(), da[0, 0, 0].T @ q0.detach(), dv_intra[0, 0, 0], output.detach() * do[0, 0, 0] - v0.detach() * dv_intra[0, 0, 0]]
    for got, want in zip((aq, ak, av, ag), expected):
        torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-10)
    dh_direct = torch.zeros_like(dh)
    for it in range(n):
        for future in range(it + 1, n):
            dh_direct[:, it] += torch.einsum('bhik,bhiv->bhkv', q[:, future], do_exp[:, future])
    torch.testing.assert_close(dh, dh_direct, rtol=1e-11, atol=1e-11)
    for i, j in ((0, 0), (7, 3), (63, 16)):
        direct = scale * math.fsum((float(do[0, 0, 0, i, l]) * float(v[0, 0, 0, j, l]) * math.exp(float(g[0, 0, 0, i, l] - g[0, 0, 0, j, l])) for l in range(values)))
        assert abs(float(da[0, 0, 0, i, j]) - direct) < 1e-09
    for i, l in ((0, 0), (19, 7), (63, 63)):
        direct = math.fsum((float(A[0, 0, 0, j, i]) * float(do[0, 0, 0, j, l]) * math.exp(float(g[0, 0, 0, j, l] - g[0, 0, 0, i, l])) for j in range(i, bt)))
        assert abs(float(dv_intra[0, 0, 0, i, l]) - direct) < 1e-09
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in gold.values()))
    proof = {'method': 'FP64 causal matrix contractions, future-chunk state sum, supplied h/o/dg prestate, and reverse chunk cumulative sum', 'autograd_all_q_k_v_g_coordinates_head0_chunk0': True, 'dh_future_chunks_crosscheck': True, 'scalar_dA_checks': 3, 'scalar_dv_checks': 3, 'actual_dh_kernel_has_no_g_last_state_scaling': True, 'FP64_gold_final_rounding': False}
    return (gold, proof)
