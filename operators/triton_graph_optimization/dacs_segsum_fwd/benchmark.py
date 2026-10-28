from opbench.graph_case import setup


def make_case(config, device='npu'):
    torch, rng, case = setup(config, device)
    b, h, s = config['shape']
    c = config['tile']['CHUNK_SIZE']
    if s % c:
        raise ValueError('Upstream segsum load is unmasked; S must be divisible by chunk size')
    x = -torch.nn.functional.softplus(torch.randn(b, h, s, generator=rng) - 3)
    chunks = x.double().reshape(b, h, s // c, c)
    forward = chunks.cumsum(-1).clamp(max=0).reshape(b, h, s).float()
    reverse = (chunks.sum(-1, keepdim=True) - chunks.cumsum(-1)).clamp(max=0).reshape(b, h, s).float()
    triangle = torch.tril(torch.ones(c, c, dtype=torch.bool), diagonal=-1)
    segments = torch.where(triangle, chunks[..., :, None], 0).cumsum(-2).clamp(max=0).float()
    da = case.input('da_ptr', x)
    cs = case.output('da_cs_ptr', forward, rtol=0, atol=1e-4)
    rev = case.output('da_cs_rev_ptr', reverse, rtol=0, atol=1e-4)
    seg = case.output('segsum_ptr', segments, rtol=0, atol=1e-4)
    case.grid = (b, h, s // c)
    case.kwargs = dict(case.inputs, SEQLEN=s, CHUNK_SIZE=c)
    for prefix, tensor in [('da', da), ('da_cs', cs), ('da_cs_rev', rev)]:
        for axis, stride in zip(('batch', 'head', 'seq'), tensor.stride()):
            case.kwargs[f'stride_{prefix}_{axis}'] = stride
    for axis, stride in zip(('batch', 'head', 'chunk', 'row', 'col'), seg.stride()):
        case.kwargs[f'stride_segsum_{axis}'] = stride
    return case
