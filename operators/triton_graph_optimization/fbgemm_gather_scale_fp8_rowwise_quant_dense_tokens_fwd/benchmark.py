from opbench.graph_case import setup, fp8_tensor


def make_case(config, device='npu'):
    torch, rng, case = setup(config, device)
    t, d, r, e = config['shape']
    dtype = getattr(torch, config['dtype'])
    x = torch.randn(t, d, generator=rng).to(dtype)
    tokens = torch.randint(t, (r,), generator=rng, dtype=torch.int32)
    experts = torch.randint(e, (r,), generator=rng, dtype=torch.int32)
    scores = torch.rand(t, e, generator=rng)
    values = x[tokens.long()].float() * scores[tokens.long(), experts.long(), None]
    maximum = values.abs().amax(-1).clamp(min=1e-6)
    if config.get('clamp_max', False):
        maximum = maximum.clamp(max=2.0)
    inverse = 448.0 / maximum
    quantized = fp8_tensor(torch, values * inverse[:, None])
    for name, value in dict(input_ptr=x, token_indices_ptr=tokens, expert_indices_ptr=experts,
                            scores_ptr=scores, scale_ub_ptr=torch.tensor([2.0])).items():
        case.input(name, value)
    # Allow one FP8 quantization step at a rounding boundary; scales are checked independently.
    case.output('output_ptr', quantized, rtol=0.125, atol=0.001953125)
    case.output('output_scale_ptr', inverse.reciprocal(), rtol=2e-5, atol=1e-7)
    tile = config['tile']
    if d % tile['BLOCK_D']:
        raise ValueError('D must be divisible by BLOCK_D')
    import triton.language as tl
    case.grid = (r,)
    case.kwargs = dict(case.inputs, valid_token_count=None, stride_t=e, stride_e=1, D=d,
                       TL_FP8_DTYPE=tl.float8e4nv, MAX_FP8=448.0, EPS=1e-6,
                       CLAMP_MAX=config.get('clamp_max', False), **tile)
    return case
