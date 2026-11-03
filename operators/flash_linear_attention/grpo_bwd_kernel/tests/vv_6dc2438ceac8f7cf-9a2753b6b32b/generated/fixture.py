"""Fixed-seed CPU inputs and independent references; no external tensor files."""
import torch

SEED = 20261031

def build(kind):
    gen = torch.Generator().manual_seed(SEED)

    def rand(*shape, dtype=torch.float32):
        return torch.randn(shape, generator=gen, dtype=dtype)

    def uniform(*shape, dtype=torch.float32):
        return torch.rand(shape, generator=gen, dtype=dtype)
    if kind == 'log2':
        x = rand(4, 2048, 8, 1024).abs() + 0.1
        n = x.numel()
        inp = {'input_data': dict(x_ptr=x, output_ptr=torch.full_like(x, float('nan')), n_elements=n, BLOCK_SIZE=4096), 'grid': ((n + 4095) // 4096,)}
        ref = {'output_ptr': x.double().log2()}
        return (inp, ref)
    if kind == 'global_cumsum':
        x = rand(4, 1024, 8, 512)
        return ({'s': x}, {'output': x.double().cumsum(1)})
    if kind == 'grpo':
        B, L, V = (2, 1024, 4096)
        logits = rand(B, L + 1, V, dtype=torch.bfloat16)
        reflog = rand(B, L)
        ids = torch.randint(0, V, (B, L + 100), generator=gen, dtype=torch.int64)
        adv = rand(B)
        mask = torch.ones(B, L, dtype=torch.int32)
        lse = logits[:, :L].float().logsumexp(-1)
        dloss = rand(B, L)
        d = dict(dloss_ptr=dloss, dlogits_ptr=torch.full_like(logits, float('nan')), logits_ptr=logits, ref_logp_ptr=reflog, input_ids_ptr=ids, advantages_ptr=adv, completion_mask_ptr=mask, lse_ptr=lse, beta=0.04, B=B, N=V, L=L, start_idx=100, BLOCK_SIZE=4096)
        z = logits[:, :L].double()
        sel = ids[:, 100:]
        chosen = z.gather(-1, sel.unsqueeze(-1)).squeeze(-1)
        logp = chosen - lse.double()
        dlogp = (0.04 * (1 - (reflog.double() - logp).exp()) - adv.double()[:, None]) * dloss.double()
        y = -(z - lse.double().unsqueeze(-1)).exp() * dlogp.unsqueeze(-1)
        y.scatter_add_(-1, sel.unsqueeze(-1), dlogp.unsqueeze(-1))
        return (dict(input_data=d, grid=(B * L,)), {'dlogits_written_rows': y, 'valid_region': '[:, :L, :]; last row of each sequence is not written by this kernel'})
    if kind == 'kl':
        N, H, V = (4096, 2048, 32000)
        x = rand(N, H)
        weight = rand(V, H)
        logits = x @ weight.T
        x = rand(N, H)
        weight = rand(V, H)
        target = x @ weight.T
        d = dict(logits=logits, target_logits=target, loss=torch.full((N,), float('nan')), s_logits=V, s_loss=1, reduction='batchmean', N=N, V=V, BV=8192)
        grad = torch.empty((N, V), dtype=torch.float64)
        loss = torch.empty(N, dtype=torch.float64)
        for start in range(0, N, 64):
            l = logits[start:start + 64].double().log_softmax(-1)
            t = target[start:start + 64].double().log_softmax(-1)
            grad[start:start + 64] = (l.exp() - t.exp()) / N
            loss[start:start + 64] = (t.exp() * (t - l)).sum(-1) / N
        return (dict(input_data=d, grid=(N,)), {'logits_gradient': grad, 'loss': loss})
    if kind in ('gate_fwd', 'gate_bwd'):
        B, T, H, D = (2, 1024, 16, 64)
        d = dict(o=rand(B, T, H * D), r=rand(B, T, H, D), k=rand(B, T, H, D), r_k=rand(H, D), v=rand(B, T, H, D), g=rand(B, T, H * D), B=B, T=T, H=H, D=D)
        names = ('o', 'r', 'k', 'r_k', 'v', 'g')
        xx = {k: d[k].double().requires_grad_(kind == 'gate_bwd') for k in names}
        correction = (xx['r'] * xx['k'] * xx['r_k']).sum(-1, keepdim=True) * xx['v']
        y = (xx['o'] + correction.reshape(B, T, H * D)) * xx['g']
        if kind == 'gate_fwd':
            return ({'input_data': d}, {'output': y.detach()})
        d['grad_output'] = rand(B, T, H * D)
        grads = torch.autograd.grad(y, tuple((xx[k] for k in names)), d['grad_output'].double())
        return ({'input_data': d}, {'grad_' + k: g.detach() for k, g in zip(names, grads)})
    raise ValueError(kind)

def make_fixtures():
    inputs, reference = build('grpo')
    values = {'grpo_bwd_kernel_in.pt': inputs}
    values['reference_cpu64.pt'] = reference
    return values
