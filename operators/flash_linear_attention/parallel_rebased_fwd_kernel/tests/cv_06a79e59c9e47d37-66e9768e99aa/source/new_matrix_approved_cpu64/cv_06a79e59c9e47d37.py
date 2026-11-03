
from opbench.fla_fixtures import load_fixture
from yb_73a32647d1f8f5 import *

def test_cpu_fp64_absolute():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_rebased_fwd_kernel/test_parallel_rebased_fwd_kernel_cuda.pt'
    data = load_fixture(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    input_data = data['input_data']
    q = input_data['q']
    k = input_data['k']
    v = input_data['v']
    B = input_data['B']
    H = input_data['H']
    T = input_data['T']
    K = input_data['K']
    V_dim = input_data['V']
    BTL = input_data['BTL']
    BTS = input_data['BTS']
    BK = input_data['BK']
    BV = input_data['BV']
    NK = input_data['NK']
    NV = input_data['NV']
    scale = input_data['scale']
    device = torch.device('npu')
    q_npu = q.to(device)
    k_npu = k.to(device)
    v_npu = v.to(device)
    o = torch.empty(NK, B, H, T, V_dim, dtype=torch.float32, device=device)
    z = torch.empty(NK, B, H, T, dtype=torch.float32, device=device)
    num_stages = 2
    num_warps = 4
    grid = (NK * NV, triton.cdiv(T, BTL), B * H)
    parallel_rebased_fwd_kernel[grid](q_npu, k_npu, v_npu, o, z, scale, T=T, B=B, H=H, K=K, V=V_dim, BTL=BTL, BTS=BTS, BK=BK, BV=BV, num_warps=num_warps, num_stages=num_stages)
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({'o': o, 'z': z}, reference)

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
