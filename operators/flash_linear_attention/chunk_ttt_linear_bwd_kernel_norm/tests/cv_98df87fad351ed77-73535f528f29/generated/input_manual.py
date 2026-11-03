"""Existing independent CPU equations; file-generation main omitted."""
import torch.nn.functional as F
import torch

def build(kind, src):
    seed = 20261031
    gen = torch.Generator().manual_seed(seed)

    def r(shape, dtype=torch.float32):
        return (torch.randn(shape, generator=gen) * 0.1).to(dtype)
    if 'ttt_norm'.startswith('oja_') and 'ttt_norm' != 'oja_dhu':
        B, T, H, D = (2, 512, 4, 64)
        BT = 64
        shape = (B, T, H, D)
        d = dict(q=r(shape), k_new=r(shape), h=r((B, T // BT, H, D, D)), gv_cumsum=r(shape), dA=r((B, T, H, BT)), do=r(shape), scale=torch.tensor(D ** (-0.5)))
        return (d, seed)
    B, T, H, D, BT = (4, 2048, 8, 64, 16)
    shape = (B, T, H, D)
    fp16 = torch.float16
    d = {k: r(shape, fp16) for k in ['q', 'k', 'v', 'v_new', 'x', 'y', 'dv_new', 'do_tensor']}
    d['k'] = F.normalize(d['k'].float(), p=2, dim=-1).to(fp16)
    d.update(rstd=r((B, T, H, 1)).abs() + 0.5, w=r((H, D), fp16), b=r((H, D), fp16), eta=r((B, T, H, 1), fp16) * 0.005, h0=r((B, H, D, D)), hb0=r((B, H, 1, D)), h=r((B, T // BT, H, D, D), fp16), dht=r((B, H, D, D)), dhbt=r((B, H, 1, D)))
    return (d, seed)
