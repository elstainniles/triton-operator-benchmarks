
from opbench.fla_fixtures import load_fixture
from yb_70796ff8813446 import *

def test_cpu_fp64_absolute():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/fused_chunk_based_bwd_kernel/test_fused_chunk_based_bwd_kernel_cuda.pt'
    data = load_fixture(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    q = input_data['q']
    k = input_data['k']
    v = input_data['v']
    do = input_data['do']
    dz = input_data['dz']
    B = input_data['B']
    H = input_data['H']
    T = input_data['T']
    K = input_data['K']
    V = input_data['V']
    scale = input_data['scale']
    model = Model(B, H, T, K, V, scale)
    with torch.no_grad():
        dq, dk, dv = model(q, k, v, do, dz)
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({'dq': dq, 'dk': dk, 'dv': dv}, reference)

def compare_cpu64_outputs(outputs, references, masks=None):
    import json
    from pathlib import Path
    records = []
    failures = []
    for name, output in outputs.items():
        dtype = str(output.dtype).removeprefix('torch.')
        tolerance = {'float32': 1e-05, 'float16': 0.001, 'bfloat16': 0.01}[dtype]
        actual = output.detach().cpu()
        expected = references[name].cpu().to(output.dtype)
        assert references[name].dtype == torch.float64
        defined = torch.ones_like(expected, dtype=torch.bool) if masks is None else masks[name]
        actual, expected = (actual[defined], expected[defined])
        difference = (actual - expected).abs()
        allowed = tolerance + tolerance * expected.abs()
        record = dict(output=name, dtype=dtype, rtol=tolerance, atol=tolerance, defined_elements=actual.numel(), undefined_elements=output.numel() - actual.numel(), max_abs_error=float(difference.max()), mismatched_elements=int((~torch.isfinite(actual) | ~torch.isfinite(expected) | (difference > allowed)).sum()), reference_dtype='float64', reference_rounded_to_output_dtype=True, comparison_dtype=dtype)
        try:
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), 'non-finite defined output or reference'
            validate_cmp(dtype=dtype, y_cal=actual, y_ref=expected, device_type='cpu')
            record['status'] = 'passed'
        except AssertionError as error:
            record.update(status='precision_failed', error=str(error))
            failures.append(name)
        records.append(record)
    Path('original_absolute_accuracy.json').write_text(json.dumps(dict(validation_standard='original_absolute_cpu_reference_v1', accuracy_criteria_changed=False, original_validation_standard='original_absolute_dtype_tolerance', outputs=records), indent=2) + '\n')
    assert not failures, 'CPU FP64 absolute validation failed for outputs: ' + ', '.join(failures)
