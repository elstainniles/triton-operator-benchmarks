from opbench.graph_case import setup


def make_case(config, device='npu'):
    torch, rng, case = setup(config, device)
    b, h, s = config['shape']
    c = config['tile']['CHUNK_SIZE']
    if s % c:
        raise ValueError('S must be divisible by chunk size')
    values = [torch.randn(b, h, s, generator=rng) * scale for scale in (1, 1, 0.1, 0.1)]
    g, gr, a, ar = [v.double().reshape(b, h, s // c, c) for v in values]
    suffix = (g * a.exp()).flip(-1).cumsum(-1).flip(-1)
    prefix = (gr * ar.exp()).cumsum(-1)
    exclusive = torch.cat((torch.zeros_like(prefix[..., :1]), prefix[..., :-1]), dim=-1)
    reference = (suffix + exclusive).reshape(b, h, s).float()
    for name, value in zip(('ddA_cs_ptr', 'ddA_cs_rev_ptr', 'dA_cs_ptr', 'dA_cs_rev_ptr'), values):
        case.input(name, value)
    case.output('ddt_out_ptr', reference, rtol=0, atol=1e-4)
    case.grid = (b, h, s // c)
    case.kwargs = dict(case.inputs, stride_batch=h*s, stride_head=s, stride_seq=1,
                       B=b, H=h, S=s, CHUNK_SIZE=c)
    return case
