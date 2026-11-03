
from opbench.fla_fixtures import load_fixture
from yb_529d3e216ebc25 import *

def test_cpu_fp64_absolute():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_dplr_bwd_o_kernel'
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs.pt'), map_location='cpu', weights_only=False)
    device = torch.device('npu')
    k = inputs_dict['k'].to(device)
    b = inputs_dict['b'].to(device)
    v = inputs_dict['v'].to(device)
    v_new = inputs_dict['v_new'].to(device)
    gk = inputs_dict['gk'].to(device)
    do = inputs_dict['do'].to(device)
    h = inputs_dict['h'].to(device)
    dh = inputs_dict['dh'].to(device)
    dv = inputs_dict['dv'].to(device)
    w = inputs_dict['w'].to(device)
    chunk_size = inputs_dict['chunk_size']
    scale = inputs_dict['scale']
    dq, dk, dw, db, dgk_last = chunk_dplr_bwd_o(k=k, b=b, v=v, v_new=v_new, gk=gk, do=do, h=h, dh=dh, dv=dv, w=w, chunk_size=chunk_size, scale=scale)
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({'dq': dq, 'dk': dk, 'dw': dw, 'db': db, 'dgk_last': dgk_last}, reference)

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
