
from opbench.fla_fixtures import load_fixture
from yb_6cac7fae86c5b7 import *

def test_cpu_fp64_absolute():
    from yb_14d6a471f4f932 import fused_chunk_ttt_linear_fwd
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/fused_chunk_ttt_linear_fwd_kernel'
    inputs = load_fixture(os.path.join(load_dir, 'inputs.pt'), map_location=torch.device('cpu'))
    q = inputs['q']
    k = inputs['k']
    v = inputs['v']
    w = inputs['w']
    b = inputs['b']
    eta = inputs['eta']
    sc = inputs['sc']
    eps_val = inputs['eps_val']
    h0 = inputs['h0']
    hb0 = inputs['hb0']
    out_final = inputs['out_final']
    cu_sl = inputs['cu_sl']
    bt = inputs['bt']
    q_npu = q.to(DEVICE)
    k_npu = k.to(DEVICE)
    v_npu = v.to(DEVICE)
    w_npu = w.to(DEVICE)
    b_npu = b.to(DEVICE)
    eta_npu = eta.to(DEVICE)
    h0_npu = h0.to(DEVICE)
    hb0_npu = hb0.to(DEVICE)
    npu_o, npu_ht, npu_hbt = fused_chunk_ttt_linear_fwd(q_npu, k_npu, v_npu, w_npu, b_npu, eta_npu, sc, eps_val, h0_npu, hb0_npu, out_final, cu_sl, bt)
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({'o': npu_o, 'ht': npu_ht, 'hbt': npu_hbt}, reference)
    for name, value in {'o': npu_o, 'ht': npu_ht, 'hbt': npu_hbt}.items():
        assert torch.allclose(value.cpu().double().float(), reference[name].float(), rtol=0.01, atol=0.01)

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
    Path('cpu_fp64_accuracy.json').write_text(json.dumps(dict(validation_standard='cpu_fp64_absolute_v1', accuracy_criteria_changed=True, original_validation_standard='cuda_error_ratio_and_cpu_absolute_1e-2', outputs=records), indent=2) + '\n')
    assert not failures, 'CPU FP64 absolute validation failed for outputs: ' + ', '.join(failures)
