
from opbench.fla_fixtures import load_fixture
from yb_0c29d768e9c45b import *

def test_cpu_fp64_absolute():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_attn_bwd_kernel_dq'
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs.pt'))
    device = torch.device('npu:0')
    q = inputs_dict['q'].to(device)
    k = inputs_dict['k'].to(device)
    v = inputs_dict['v'].to(device)
    lse = inputs_dict['lse'].to(device)
    delta = inputs_dict['delta'].to(device)
    do = inputs_dict['do'].to(device)
    dq = inputs_dict['dq'].to(device)
    dg_cumsum = inputs_dict['dg_cumsum']
    g_cumsum = inputs_dict['g_cumsum']
    scale = inputs_dict['scale']
    B = inputs_dict['B']
    T = inputs_dict['T']
    H = inputs_dict['H']
    HQ = inputs_dict['HQ']
    G = inputs_dict['G']
    K = inputs_dict['K']
    V = inputs_dict['V']
    BT = inputs_dict['BT']
    BS = inputs_dict['BS']
    BK = inputs_dict['BK']
    BV = inputs_dict['BV']
    NV = triton.cdiv(V, BV)
    NT = triton.cdiv(T, BT)
    grid = (NV, NT, B * HQ)
    parallel_attn_bwd_kernel_dq[grid](q=q, k=k, v=v, lse=lse, delta=delta, do=do, dq=dq, dg_cumsum=dg_cumsum, g_cumsum=g_cumsum, scale=scale, cu_seqlens=None, chunk_indices=None, T=T, W=None, B=B, H=H, HQ=HQ, G=G, K=K, V=V, BT=BT, BS=BS, BK=BK, BV=BV, num_warps=4)
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({'dq': dq}, reference)

def compare_cpu64_outputs(outputs, references, masks=None):
    import json
    from pathlib import Path
    records = []
    failures = []
    for name, output in outputs.items():
        dtype = str(output.dtype).removeprefix('torch.')
        tolerance = {'float32': 1e-05, 'float16': 0.001, 'bfloat16': 0.01}[dtype]
        actual = output.detach().cpu().double()
        expected = references[name].cpu()
        assert expected.dtype == torch.float64
        defined = torch.ones_like(expected, dtype=torch.bool) if masks is None else masks[name]
        actual, expected = (actual[defined], expected[defined])
        difference = (actual - expected).abs()
        allowed = tolerance + tolerance * expected.abs()
        record = dict(output=name, dtype=dtype, rtol=tolerance, atol=tolerance, defined_elements=actual.numel(), undefined_elements=output.numel() - actual.numel(), max_abs_error=float(difference.max()), mismatched_elements=int((~torch.isfinite(actual) | ~torch.isfinite(expected) | (difference > allowed)).sum()), reference_dtype='float64', reference_rounded_to_output_dtype=False)
        try:
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), 'non-finite defined output or reference'
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=False)
            record['status'] = 'passed'
        except AssertionError as error:
            record.update(status='precision_failed', error=str(error))
            failures.append(name)
        records.append(record)
    Path('cpu_fp64_accuracy.json').write_text(json.dumps(dict(validation_standard='cpu_fp64_absolute_v1', accuracy_criteria_changed=True, original_validation_standard='cuda_error_ratio', outputs=records), indent=2) + '\n')
    assert not failures, 'CPU FP64 absolute validation failed for outputs: ' + ', '.join(failures)
