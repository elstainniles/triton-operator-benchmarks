
from opbench.fla_fixtures import load_fixture
import triton
from yb_11ce81bc317970 import *
from yb_282df376bcfe75 import chunk_rwkv6_fwd_A_kernel_intra_sub_inter

def chunk_rwkv6_fwd_intra_sub_inter(q: torch.Tensor, k: torch.Tensor, gi: torch.Tensor, ge: torch.Tensor, scale: float, chunk_size: int=64):
    B, T, H, K = k.shape
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    BC = min(16, BT)
    NC = triton.cdiv(BT, BC)
    A = q.new_zeros(B, T, H, BT, dtype=torch.float)
    grid = (NT, NC * NC, B * H)
    chunk_rwkv6_fwd_A_kernel_intra_sub_inter[grid](q, k, gi, ge, A, None, None, scale, T=T, H=H, K=K, BT=BT, BC=BC, NC=NC)
    return A

def test_cpu_fp64_absolute():
    load_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/test_chunk_rwkv6_fwd_A_kernel_intra_sub_inter'
    data = load_fixture(os.path.join(load_path, 'test_chunk_rwkv6_fwd_A_kernel_intra_sub_inter_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    A = chunk_rwkv6_fwd_intra_sub_inter(input_data['q'], input_data['k'], input_data['gi'], input_data['ge'], scale=input_data['scale'], chunk_size=input_data['chunk_size'])
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({'A': A}, reference)

def compare_cpu64_outputs(outputs, references, masks=None):
    import json
    from pathlib import Path
    records = []
    failures = []
    for name, output in outputs.items():
        dtype = str(output.dtype).removeprefix('torch.')
        tolerance = {'float32': 1e-05, 'float16': 0.001, 'bfloat16': 0.01}[dtype]
        actual = output.detach().cpu().double()
        expected = references[name].cpu().to(output.dtype).double()
        assert expected.dtype == torch.float64
        defined = torch.ones_like(expected, dtype=torch.bool) if masks is None else masks[name]
        actual, expected = (actual[defined], expected[defined])
        difference = (actual - expected).abs()
        allowed = tolerance + tolerance * expected.abs()
        record = dict(output=name, dtype=dtype, rtol=tolerance, atol=tolerance, defined_elements=actual.numel(), undefined_elements=output.numel() - actual.numel(), max_abs_error=float(difference.max()), mismatched_elements=int((~torch.isfinite(actual) | ~torch.isfinite(expected) | (difference > allowed)).sum()), reference_dtype='float64', reference_rounded_to_output_dtype=True)
        try:
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), 'non-finite defined output or reference'
            compare_data_precision({name: references[name].cpu().to(output.dtype)}, {name: output.detach().cpu()}, device_type='cpu')
            record['status'] = 'passed'
        except AssertionError as error:
            record.update(status='precision_failed', error=str(error))
            failures.append(name)
        records.append(record)
    Path('cpu_fp64_accuracy.json').write_text(json.dumps(dict(validation_standard='original_test_acceptance', accuracy_criteria_changed=False, original_validation_standard='validate_cmp_dtype_absolute', outputs=records), indent=2) + '\n')
    assert not failures, 'CPU FP64 absolute validation failed for outputs: ' + ', '.join(failures)
