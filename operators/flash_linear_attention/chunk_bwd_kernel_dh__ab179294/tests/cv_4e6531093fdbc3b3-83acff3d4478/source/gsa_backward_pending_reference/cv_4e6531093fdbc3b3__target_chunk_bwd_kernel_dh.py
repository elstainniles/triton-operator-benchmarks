
from opbench.fla_fixtures import load_fixture
from yb_7ea4ab39bc3ba6 import *

def test_reference():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_gsa_bwd_k/test_chunk_gsa_bwd_k.pt'
    data = load_fixture(ptfile_path, map_location='cpu', weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    dq, dk, dv, dg, dh0, dA, A = chunk_gsa_bwd_k(q=input_data['q'], k=input_data['k'], v=input_data['v'], g=input_data['g'], h=input_data['h'], h0=input_data['h0'], o=input_data['o'], do=input_data['do'], dht=input_data['dht'], dg=input_data['dg'], scale=input_data['scale'], cu_seqlens=None, chunk_size=input_data['BT'])
    assert dh0 is None, 'original h0 is None so dh0 must be None'
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({'dq': dq, 'dk': dk, 'dv': dv, 'dg': dg, 'dA': dA, 'A': A}, reference)

def compare_cpu64_outputs(outputs, references, masks=None):
    import json
    from pathlib import Path
    records = []
    failures = []
    for name, output in outputs.items():
        dtype = str(output.dtype).removeprefix('torch.')
        tolerance = 0.005
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
    Path('cpu_fp64_accuracy.json').write_text(json.dumps(dict(validation_standard='original_absolute', accuracy_criteria_changed=False, original_validation_standard='original_absolute_5e-3_against_gpu', outputs=records), indent=2) + '\n')
    assert not failures, 'CPU FP64 absolute validation failed for outputs: ' + ', '.join(failures)
