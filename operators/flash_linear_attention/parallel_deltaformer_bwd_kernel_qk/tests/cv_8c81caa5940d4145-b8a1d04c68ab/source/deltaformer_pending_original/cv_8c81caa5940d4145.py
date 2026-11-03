
from opbench.fla_fixtures import load_fixture
from yb_f3bceb06835ce9 import *

def test_original_absolute():
    from yb_a87c34e2ef2580 import parallel_deltaformer_bwd_kernel_qk
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_deltaformer_bwd_kernel_qk'
    q = load_fixture(os.path.join(load_dir, 'input_q.pt')).to(DEVICE)
    k = load_fixture(os.path.join(load_dir, 'input_k.pt')).to(DEVICE)
    u = load_fixture(os.path.join(load_dir, 'input_u.pt')).to(DEVICE)
    lse = load_fixture(os.path.join(load_dir, 'input_lse.pt')).to(DEVICE)
    grad_v = load_fixture(os.path.join(load_dir, 'input_grad_v.pt')).to(DEVICE)
    beta = load_fixture(os.path.join(load_dir, 'input_beta.pt')).to(DEVICE)
    row_dot_sum = load_fixture(os.path.join(load_dir, 'input_row_dot_sum.pt')).to(DEVICE)
    params = load_fixture(os.path.join(load_dir, 'params.pt'))
    H, T, D = (params['H'], params['T'], params['D'])
    fa_scale, qk_scale = (params['fa_scale'], params['qk_scale'])
    grad_q = torch.empty_like(q)
    grad_k = torch.empty_like(k)

    def grid_bp(META):
        return (triton.cdiv(T, META['BLOCK_C']), H)
    parallel_deltaformer_bwd_kernel_qk[grid_bp](grad_q, grad_k, q, k, grad_v, u, lse, beta, row_dot_sum, H, T, D, fa_scale, qk_scale, disable_tightly_coupled_buffer_reuse=True)
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({'grad_q': grad_q, 'grad_k': grad_k}, reference)

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
    Path('cpu_fp64_accuracy.json').write_text(json.dumps(dict(validation_standard='original_absolute', accuracy_criteria_changed=False, original_validation_standard='original_absolute_1e-3', outputs=records), indent=2) + '\n')
    assert not failures, 'CPU FP64 absolute validation failed for outputs: ' + ', '.join(failures)
