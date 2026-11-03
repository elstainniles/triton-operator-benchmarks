
from opbench.fla_fixtures import load_fixture
from yb_b37133629ed74d import *

def test_reference():
    data = load_fixture('/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_gsa_bwd_k/test_chunk_gsa_bwd_k.pt', map_location='cpu', weights_only=False)
    input_data = data['input_data']
    device = 'npu'
    v = input_data['v'].to(device)
    g = input_data['g'].to(device)
    o = input_data['o'].to(device)
    do = input_data['do'].to(device)
    B = input_data['B']
    T = input_data['T']
    HQ = input_data['HQ']
    H = input_data['H']
    V = input_data['V']
    BT = input_data['BT']
    BC = input_data['BC']
    BV = input_data['BV']
    NG = input_data['NG']
    scale = input_data['scale']
    q = input_data['q']
    k = input_data['k']
    A = torch.zeros(B, T, HQ, BT, dtype=q.dtype)
    NT = T // BT
    for i_b in range(B):
        for i_t in range(NT):
            t_start = i_t * BT
            t_end = min(t_start + BT, T)
            for i_hq in range(HQ):
                q_chunk = q[i_b, t_start:t_end, i_hq, :].float()
                k_chunk = k[i_b, t_start:t_end, i_hq // NG, :].float()
                A_chunk = torch.matmul(q_chunk, k_chunk.T) * scale
                mask = torch.tril(torch.ones(t_end - t_start, t_end - t_start, dtype=torch.bool))
                A[i_b, t_start:t_end, i_hq, :t_end - t_start] = A_chunk * mask
    A = A.to(device)
    dv_in = torch.zeros(B, T, HQ, V, dtype=v.dtype, device=device)
    dg_in = torch.zeros(B, T, HQ, V, dtype=g.dtype, device=device)
    candidate_dv, candidate_dg = run_kernel(chunk_gsa_bwd_k_kernel_intra_dvg_fixed, v, g, o, A, do, dv_in, dg_in, B, T, HQ, H, V, BT, BC, BV, NG)
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({'dv': candidate_dv, 'dg': candidate_dg}, reference)

def compare_cpu64_outputs(outputs, references, masks=None):
    import json
    from pathlib import Path
    records = []
    failures = []
    for name, output in outputs.items():
        dtype = str(output.dtype).removeprefix('torch.')
        tolerance = 0.01
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
    Path('cpu_fp64_accuracy.json').write_text(json.dumps(dict(validation_standard='original_absolute', accuracy_criteria_changed=False, original_validation_standard='original_absolute_1e-2', outputs=records), indent=2) + '\n')
    assert not failures, 'CPU FP64 absolute validation failed for outputs: ' + ', '.join(failures)
