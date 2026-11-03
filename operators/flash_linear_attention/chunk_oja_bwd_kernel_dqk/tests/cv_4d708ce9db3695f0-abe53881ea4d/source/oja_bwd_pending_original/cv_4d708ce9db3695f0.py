
from opbench.fla_fixtures import load_fixture
from yb_acb288fc0bc32f import *

def test_original_absolute():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_oja_bwd_kernel_dqk/'
    from yb_8b1f9f64c8c620 import chunk_oja_bwd_dqk
    q = load_fixture(os.path.join(load_dir, 'q.pt')).to(DEVICE)
    k_new = load_fixture(os.path.join(load_dir, 'k_new.pt')).to(DEVICE)
    h = load_fixture(os.path.join(load_dir, 'h.pt')).to(DEVICE)
    gv_cumsum = load_fixture(os.path.join(load_dir, 'gv_cumsum.pt')).to(DEVICE)
    dA = load_fixture(os.path.join(load_dir, 'dA.pt')).to(DEVICE)
    do = load_fixture(os.path.join(load_dir, 'do.pt')).to(DEVICE)
    scale = load_fixture(os.path.join(load_dir, 'scale.pt')).item()
    chunk_size = load_fixture(os.path.join(load_dir, 'chunk_size.pt')).item()
    output = chunk_oja_bwd_dqk(q, k_new, h, gv_cumsum, dA, do, scale, None, None, chunk_size)
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs(dict(zip(('A', 'dq', 'dk'), output)), reference)

def compare_cpu64_outputs(outputs, references, masks=None):
    import json
    from pathlib import Path
    records = []
    failures = []
    for name, output in outputs.items():
        dtype = str(output.dtype).removeprefix('torch.')
        tolerance = 0.008
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
    Path('cpu_fp64_accuracy.json').write_text(json.dumps(dict(validation_standard='original_absolute', accuracy_criteria_changed=False, original_validation_standard='original_absolute_0.008', outputs=records), indent=2) + '\n')
    assert not failures, 'CPU FP64 absolute validation failed for outputs: ' + ', '.join(failures)
