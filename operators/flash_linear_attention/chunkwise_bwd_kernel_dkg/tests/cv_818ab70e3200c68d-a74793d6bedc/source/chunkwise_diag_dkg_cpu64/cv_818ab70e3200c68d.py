
from opbench.fla_fixtures import load_fixture
from yb_a58150ab078e75 import *

def test_cpu_fp64_absolute():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunkwise_bwd_kernel_dkg'
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs.pt'))
    device = torch.device('npu:0')
    dh = inputs_dict['dh'].to(device)
    k = inputs_dict['k'].to(device)
    v = inputs_dict['v'].to(device)
    g = inputs_dict['g'].to(device)
    dg_last = inputs_dict['dg_last'].to(device)
    dk = inputs_dict['dk'].to(device)
    dg = inputs_dict['dg'].to(device)
    cu_seqlens = inputs_dict['cu_seqlens']
    B = inputs_dict['B']
    T = inputs_dict['T']
    K = inputs_dict['K']
    H = inputs_dict['H']
    V = inputs_dict['V']
    L = inputs_dict['L']
    BT = inputs_dict['BT']
    NT = inputs_dict['NT']
    grid_t = (NT, B * H)
    chunkwise_bwd_kernel_dkg[grid_t](dh=dh, k=k, v=v, g=g, dg_last=dg_last, dk=dk, dg=dg, cu_seqlens=cu_seqlens, T=T, H=H, K=K, V=V, L=L, BT=BT, NT=NT)
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({'dk': dk, 'dg': dg, 'dg_last': dg_last}, reference)

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
