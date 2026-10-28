from opbench.graph_case import setup, fp8_tensor, ceildiv


def make_case(config, device='npu'):
    torch, rng, case = setup(config, device)
    m, n, k = config['shape']
    a = fp8_tensor(torch, torch.randn(m, k, generator=rng))
    b = fp8_tensor(torch, torch.randn(n, k, generator=rng))
    a_scale = 0.1 + torch.rand(m, generator=rng)
    b_scale = 0.1 + torch.rand(n, generator=rng)
    bias = torch.randn(n, generator=rng) * 0.1
    reference = (a.double() @ b.double().T) * a_scale.double()[:, None] * b_scale.double()[None, :]
    if config.get('use_bias', False):
        reference += bias.double()[None, :]
    for name, value in dict(A_ptr=a, B_ptr=b, A_scale=a_scale, B_scale=b_scale, Bias=bias).items():
        case.input(name, value)
    case.output('C_ptr', reference.float(), rtol=2e-3, atol=2e-3)
    tile = config['tile']
    sms = min(20, ceildiv(m, tile['BLOCK_M']) * ceildiv(n, tile['BLOCK_N']))
    import triton.language as tl
    case.grid = (sms,)
    case.kwargs = dict(case.inputs, M=m, N=n, K=k, m_key=m, n_key=n, k_key=k,
                       stride_am=k, stride_ak=1, stride_bn=k, stride_bk=1, stride_cm=n, stride_cn=1,
                       dot_out_dtype=tl.float32, allow_tf32=False, fp8_fast_accum=False,
                       skip_scaling_a=False, GROUP_M=4, SPLIT_K=1,
                       USE_BIAS=config.get('use_bias', False), AB_DTYPE=False, NUM_SMS=sms, **tile)
    return case
