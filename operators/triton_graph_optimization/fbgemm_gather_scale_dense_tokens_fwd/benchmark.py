from opbench.graph_case import setup


def make_case(config, device='npu'):
    torch, rng, case = setup(config, device)
    t, d, r, e = config['shape']
    dtype = getattr(torch, config['dtype'])
    x = torch.randn(t, d, generator=rng).to(dtype)
    tokens = torch.randint(t, (r,), generator=rng, dtype=torch.int32)
    experts = torch.randint(e, (r,), generator=rng, dtype=torch.int32)
    scores = torch.rand(t, e, generator=rng)
    valid = r - config.get('inactive_rows', 0)
    reference = (x[tokens.long()].float() * scores[tokens.long(), experts.long(), None]).to(dtype)
    reference[valid:] = 0
    for name, value in dict(x=x, token_indices=tokens, expert_indices=experts, scores=scores).items():
        case.input(name, value)
    case.output('out', reference, rtol=2e-4 if dtype == torch.float32 else 8e-3, atol=1e-5, zero=valid < r)
    case.input('valid_token_count', torch.tensor([valid], dtype=torch.int32))
    tile = config['tile']
    if d % tile['BLOCK_D_OUTER'] or tile['BLOCK_D_OUTER'] % tile['BLOCK_D_INNER']:
        raise ValueError('Unmasked gather requires D divisible by outer tile and outer by inner')
    case.grid = (r, d // tile['BLOCK_D_OUTER'])
    case.kwargs = dict(case.inputs, stride_t=e, stride_e=1, D=d, **tile)
    return case
