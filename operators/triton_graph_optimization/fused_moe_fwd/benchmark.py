"""Self-contained MoE inputs and reference; no GPU .pt oracle is required."""
from opbench.graph_case import setup, fp8_tensor, ceildiv


def route_tokens(torch, selected, block_m):
    count = selected.numel()
    sorted_ids, block_experts = [], []
    for expert in range(int(selected.max()) + 1):
        ids = (selected.flatten() == expert).nonzero().flatten().tolist()
        padded = ceildiv(len(ids), block_m) * block_m
        sorted_ids.extend(ids + [count] * (padded - len(ids)))
        block_experts.extend([expert] * (padded // block_m))
    return torch.tensor(sorted_ids, dtype=torch.int32), torch.tensor(block_experts, dtype=torch.int32)


def make_case(config, device='npu'):
    torch, rng, case = setup(config, device)
    m, n, k, e, top = config['shape']
    mode = config['mode']
    dtype = getattr(torch, config.get('output_dtype', 'float16'))
    scale_dtype = getattr(torch, config.get('scale_dtype', 'float32'))
    tile = config['tile']
    # The kernel divides routed position by top_k: A contains ORIGINAL token rows.
    a = (torch.randn(m, k, generator=rng) * 0.1).to(dtype)
    b = (torch.randn(e, n, k, generator=rng) * 0.05).to(dtype)
    logits = torch.randn(m, e, generator=rng)
    weights, selected = logits.topk(top, dim=1)
    weights = weights.softmax(-1).to(dtype)
    sorted_ids, experts = route_tokens(torch, selected, tile['BLOCK_SIZE_M'])
    em = sorted_ids.numel()
    a_scale = b_scale = None
    group_n = group_k = 0
    if mode == 'int8_w8a16':
        b_scale = (b.float().abs().amax(-1).clamp(min=1e-6) / 127).to(scale_dtype)
        b = (b.float() / b_scale.float()[..., None]).round().clamp(-127, 127).to(torch.int8)
    elif mode == 'block_int8_w8a8':
        group_n, group_k = 64, 64
        if n % group_n or k % group_k or tile['BLOCK_SIZE_K'] != group_k:
            raise ValueError('Block INT8 benchmark requires aligned N/K and BLOCK_SIZE_K=group_k=64')
        blocks = a.float().reshape(m, k // group_k, group_k)
        a_scale = (blocks.abs().amax(-1).clamp(min=1e-6) / 127).to(scale_dtype)
        a = (blocks / a_scale.float()[..., None]).round().clamp(-127, 127).to(torch.int8).reshape(m, k)
        blocks = b.float().reshape(e, n // group_n, group_n, k // group_k, group_k)
        b_scale = (blocks.abs().amax((2, 4)).clamp(min=1e-6) / 127).to(scale_dtype)
        b = (blocks / b_scale.float()[:, :, None, :, None]).round().clamp(-127, 127).to(torch.int8).reshape(e, n, k)
    elif mode == 'fp8_w8a8_tensor':
        a_scale = (a.float().abs().max().clamp(min=1e-6) / 448).reshape(1).to(scale_dtype)
        b_scale = (b.float().abs().amax((1, 2)).clamp(min=1e-6) / 448).to(scale_dtype)
        a = fp8_tensor(torch, a.float() / a_scale.float())
        b = fp8_tensor(torch, b.float() / b_scale.float()[:, None, None])
    elif mode != 'no_quant':
        raise ValueError('Unknown MoE quantization mode: ' + mode)

    # Independent expert-wise CPU matmuls, followed by routing/scaling in logical order.
    reference = torch.empty(m * top, n, dtype=torch.float64)
    af, bf = a.double(), b.double()
    for expert in range(e):
        positions = (selected.flatten() == expert).nonzero().flatten()
        if not positions.numel():
            continue
        tokens = positions // top
        if mode == 'block_int8_w8a8':
            product = torch.zeros(len(tokens), n, dtype=torch.float64)
            for kb in range(k // group_k):
                start = kb * group_k
                part = af[tokens, start:start+group_k] @ bf[expert, :, start:start+group_k].T
                # Source multiplies scale values before promoting the product to FP32.
                scales = (a_scale[tokens, kb, None] * b_scale[expert, :, kb].repeat_interleave(group_n)[None, :])
                product += part * scales.double()
        else:
            product = af[tokens] @ bf[expert].T
            if mode == 'int8_w8a16':
                product *= b_scale[expert].double()[None, :]
            elif mode == 'fp8_w8a8_tensor':
                product *= (a_scale[0] * b_scale[expert]).double()
        if config.get('mul_routed_weight', True):
            product *= weights.flatten()[positions].double()[:, None]
        reference[positions] = product
    reference = reference.to(getattr(torch, config.get('compute_dtype', 'float16'))).to(dtype)
    if config.get('c_sorted', False):
        sorted_reference = torch.zeros(em, n, dtype=dtype)
        mask = sorted_ids < m * top
        sorted_reference[mask] = reference[sorted_ids[mask].long()]
        reference = sorted_reference
    for name, value in dict(a_ptr=a, b_ptr=b, topk_weights_ptr=weights,
                            sorted_token_ids_ptr=sorted_ids, expert_ids_ptr=experts,
                            num_tokens_post_padded_ptr=torch.tensor([em], dtype=torch.int32)).items():
        case.input(name, value)
    for name, value in [('a_scale_ptr', a_scale), ('b_scale_ptr', b_scale)]:
        if value is not None:
            case.input(name, value)
        else:
            case.inputs[name] = None
    out = case.output('c_ptr', reference, rtol=2e-2 if dtype == torch.bfloat16 else 3e-3,
                      atol=2e-3 if dtype == torch.bfloat16 else 2e-4, zero=True)
    import triton.language as tl
    case.grid = (ceildiv(em, tile['BLOCK_SIZE_M']) * ceildiv(n, tile['BLOCK_SIZE_N']),)
    a_strides = a_scale.stride() if a_scale is not None else ()
    b_strides = b_scale.stride() if b_scale is not None else ()
    case.kwargs = dict(case.inputs, a_desc=None, b_desc=None, bias_ptr=None,
        N=n, K=k, EM=em, num_valid_tokens=m*top, stride_am=k, stride_ak=1,
        stride_be=n*k, stride_bk=1, stride_bn=k, stride_bias_e=0, stride_bias_n=0,
        stride_cm=out.stride(0), stride_cn=out.stride(1),
        stride_asm=a_strides[0] if a_strides else 0, stride_ask=a_strides[1] if len(a_strides)>1 else 0,
        stride_bse=b_strides[0] if b_strides else 0,
        stride_bsk=b_strides[2] if len(b_strides)>2 else 0,
        stride_bsn=b_strides[1] if len(b_strides)>1 else 0,
        group_n=group_n, group_k=group_k, **tile,
        MUL_ROUTED_WEIGHT=config.get('mul_routed_weight', True), top_k=top,
        compute_type=getattr(tl, config.get('compute_dtype', 'float16')),
        use_fp8_w8a8=mode=='fp8_w8a8_tensor', use_int8_w8a8=mode=='block_int8_w8a8',
        use_int8_w8a16=mode=='int8_w8a16', per_channel_quant=mode=='int8_w8a16',
        even_Ks=k % tile['BLOCK_SIZE_K'] == 0, c_sorted=config.get('c_sorted', False),
        filter_expert=False, swap_ab=False)
    return case
