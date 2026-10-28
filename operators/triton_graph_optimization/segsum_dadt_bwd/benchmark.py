from opbench.graph_case import setup


def make_case(config, device='npu'):
    torch, rng, case = setup(config, device)
    b, h, s = config['shape']
    c = config['tile']['CHUNK_SIZE']
    if s % c:
        raise ValueError('S must be divisible by chunk size')
    nchunks = s // c
    grad = torch.randn(b, h, nchunks, c, c, generator=rng)
    da = torch.randn(b, h, nchunks, c, generator=rng) * 0.1
    seg = da[..., :, None] - da[..., None, :]
    product = grad.double().transpose(-1, -2) * seg.double().exp()
    suffix = product.flip(-2).cumsum(-2).flip(-2)
    reference = suffix.tril(-1).sum(-1).reshape(b, h, s).float()
    g = case.input('dSSdA_ptr', grad)
    a = case.input('SSdA_cs_ptr', seg)
    out = case.output('ddt_out_ptr', reference, rtol=0, atol=1e-4, zero=True)
    case.accumulates = True
    case.grid = (b, h, nchunks)
    case.kwargs = dict(case.inputs, B=b, H=h, nchunks=nchunks, C=c, CHUNK_SIZE=c)
    for prefix, tensor in [('dSSdA', g), ('SSdA', a)]:
        for axis, stride in zip(('batch', 'head', 'chunk', 'row', 'col'), tensor.stride()):
            case.kwargs[f'stride_{prefix}_{axis}'] = stride
    for axis, stride in zip(('batch', 'head', 'seq'), out.stride()):
        case.kwargs[f'stride_ddt_{axis}'] = stride
    return case
