
from opbench.fla_fixtures import load_fixture
from yb_96620d1805e536 import *

def test_cpu_fp64_absolute():
    config = get_test_config()
    q, k, v, gk, beta, Aqk, Akkd = load_inputs(config)
    B, T, H, K, HV = (*k.shape, gk.shape[2])
    BT = config['chunk_size']
    BC = 16
    NC = triton.cdiv(BT, BC)
    scale = config['scale']
    safe_gate = config['safe_gate']
    Akk = torch.zeros(B, T, HV, BT, device=k.device, dtype=k.dtype)
    NT = triton.cdiv(T, BT)
    grid = (NT, B * HV)
    chunk_kda_fwd_kernel_inter_solve_fused[grid](q=q, k=k, g=gk, beta=beta, Aqk=Aqk, Akkd=Akkd, Akk=Akk, scale=scale, cu_seqlens=None, chunk_indices=None, T=T, H=H, HV=HV, K=K, BT=BT, BC=BC, NC=NC, USE_SAFE_GATE=safe_gate)
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({'Aqk': Aqk, 'Akk': Akk, 'Akkd': Akkd}, reference)

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
