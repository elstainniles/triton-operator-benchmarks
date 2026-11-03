
from opbench.fla_fixtures import load_fixture
from yb_332986ff2b58a3 import *

def test_cpu_fp64_absolute():
    device = torch.device('npu:0')
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/recompute_w_u_fwd_kda_kernel'
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs.pt'))
    k = inputs_dict['k'].to(device)
    v = inputs_dict['v'].to(device)
    beta = inputs_dict['beta'].to(device)
    A = inputs_dict['A'].to(device)
    gk = inputs_dict['gk'].to(device)
    q = inputs_dict['q']
    w, u, qg, kg = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, gk=gk, q=q, cu_seqlens=None)
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({'w': w, 'u': u, 'kg': kg}, reference)

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
