
from opbench.fla_fixtures import load_fixture
from yb_802f0724912771 import *

def test_cpu_fp64_absolute():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/intra_chunk_preprocess_bwd_kernel'
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs.pt'))
    device = torch.device('npu')
    q = inputs_dict['q'].to(device)
    k = inputs_dict['k'].to(device)
    w = inputs_dict['w'].to(device)
    w2 = inputs_dict['w2'].to(device)
    beta = inputs_dict['beta'].to(device)
    dq = inputs_dict['dq'].to(device)
    dk = inputs_dict['dk'].to(device)
    dA_local = inputs_dict['dA_local'].to(device)
    dw1 = inputs_dict['dw1'].to(device)
    dw2 = inputs_dict['dw2'].to(device)
    A = inputs_dict['A'].to(device)
    L = inputs_dict['L'].to(device)
    D = inputs_dict['D'].to(device)
    do = inputs_dict['do'].to(device)
    scale = inputs_dict['scale']
    cu_seqlens = inputs_dict['cu_seqlens']
    dq_new_npu, dk_new_npu, dbeta_npu, dw_npu = intra_chunk_preprocess_bwd_fn(q=q, k=k, w=w, w2=w2, beta=beta, dq=dq, dk=dk, dA_local=dA_local, dw1=dw1, dw2=dw2, A=A, L=L, D=D, do=do, scale=scale, cu_seqlens=cu_seqlens)
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({'dq_new': dq_new_npu, 'dk_new': dk_new_npu, 'dbeta': dbeta_npu, 'dw': dw_npu}, reference)

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
