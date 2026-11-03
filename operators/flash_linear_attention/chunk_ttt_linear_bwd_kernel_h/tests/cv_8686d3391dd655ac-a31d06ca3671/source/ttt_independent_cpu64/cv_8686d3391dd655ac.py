
from opbench.fla_fixtures import load_fixture
from yb_0a21c20b381155 import *

def test_cpu_fp64_absolute():
    from yb_7bff0dd87eff27 import chunk_ttt_linear_bwd_h
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_ttt_linear_bwd_kernel_h'
    inputs = load_fixture(os.path.join(load_dir, 'inputs.pt'), map_location=torch.device('cpu'))
    k = inputs['k']
    v = inputs['v']
    w = inputs['w']
    b = inputs['b']
    eta = inputs['eta']
    eps_val = inputs['eps_val']
    h0 = inputs['h0']
    hb0 = inputs['hb0']
    bt = inputs['chunk_size']
    k_npu = k.to(DEVICE)
    v_npu = v.to(DEVICE)
    w_npu = w.to(DEVICE)
    b_npu = b.to(DEVICE)
    eta_npu = eta.to(DEVICE)
    h0_npu = h0.to(DEVICE)
    hb0_npu = hb0.to(DEVICE)
    npu_h, npu_v_new, npu_x, npu_y, npu_rstd = chunk_ttt_linear_bwd_h(k_npu, v_npu, w_npu, b_npu, eta_npu, eps_val, h0_npu, hb0_npu, None, bt)
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({'h': npu_h, 'v2': npu_v_new, 'x': npu_x, 'y': npu_y, 'r': npu_rstd}, reference)

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
