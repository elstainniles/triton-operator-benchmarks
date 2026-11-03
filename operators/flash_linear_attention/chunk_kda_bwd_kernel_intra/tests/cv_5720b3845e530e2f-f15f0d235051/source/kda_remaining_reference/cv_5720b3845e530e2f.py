
from opbench.fla_fixtures import load_fixture
from yb_37b1be0c136796 import *

def test_cpu_fp64_absolute():
    config = get_test_config()
    q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, chunk_size, safe_gate = load_inputs(config)
    B, T, H, K, HV = (*k.shape, g.shape[2])
    BT = chunk_size
    BC = min(16, BT)
    BK = min(32, triton.next_power_of_2(K))
    NT = triton.cdiv(T, BT)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)
    dq2 = torch.empty_like(dq)
    dk2 = torch.empty_like(dk)
    db2 = beta.new_empty(NK, *beta.shape, dtype=torch.float32)
    dg2 = torch.empty_like(dg, dtype=torch.float32)
    grid = (NK * NC, NT, B * HV)
    chunk_kda_bwd_kernel_intra[grid](q=q, k=k, g=g, beta=beta, dAqk=dAqk, dAkk=dAkk, dq=dq, dq2=dq2, dk=dk, dk2=dk2, dg=dg, dg2=dg2, db=db2, cu_seqlens=None, chunk_indices=None, B=B, T=T, H=H, HV=HV, K=K, BT=BT, BC=BC, BK=BK, NC=NC, SAFE_GATE=safe_gate, USE_GATHER=IS_GATHER_SUPPORTED)
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_original_absolute({'dq2': dq2, 'dk2': dk2, 'db2': db2, 'dg2': dg2}, reference, 1e-05)

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

def compare_original_absolute(outputs, references, tolerance):
    failures = []
    for name, output in outputs.items():
        try:
            torch.testing.assert_close(output.detach().cpu().double(), references[name].cpu(), rtol=tolerance, atol=tolerance, equal_nan=True)
        except AssertionError as error:
            failures.append(name + ': ' + str(error))
    if failures:
        raise AssertionError('\n'.join(failures))
