
from opbench.fla_fixtures import load_fixture
from yb_68a6d0a73ba5cb import *

def test_cpu_fp64_absolute():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_ttt_linear_bwd_kernel_norm'
    q = load_fixture(os.path.join(load_dir, 'q.pt'), map_location=torch.device('cpu'))
    k = load_fixture(os.path.join(load_dir, 'k.pt'), map_location=torch.device('cpu'))
    v = load_fixture(os.path.join(load_dir, 'v.pt'), map_location=torch.device('cpu'))
    v_new = load_fixture(os.path.join(load_dir, 'v_new.pt'), map_location=torch.device('cpu'))
    x = load_fixture(os.path.join(load_dir, 'x.pt'), map_location=torch.device('cpu'))
    y = load_fixture(os.path.join(load_dir, 'y.pt'), map_location=torch.device('cpu'))
    rstd = load_fixture(os.path.join(load_dir, 'rstd.pt'), map_location=torch.device('cpu'))
    w = load_fixture(os.path.join(load_dir, 'w.pt'), map_location=torch.device('cpu'))
    b = load_fixture(os.path.join(load_dir, 'b.pt'), map_location=torch.device('cpu'))
    eta = load_fixture(os.path.join(load_dir, 'eta.pt'), map_location=torch.device('cpu'))
    h0 = load_fixture(os.path.join(load_dir, 'h0.pt'), map_location=torch.device('cpu'))
    hb0 = load_fixture(os.path.join(load_dir, 'hb0.pt'), map_location=torch.device('cpu'))
    h = load_fixture(os.path.join(load_dir, 'h.pt'), map_location=torch.device('cpu'))
    dht = load_fixture(os.path.join(load_dir, 'dht.pt'), map_location=torch.device('cpu'))
    dhbt = load_fixture(os.path.join(load_dir, 'dhbt.pt'), map_location=torch.device('cpu'))
    dv_new = load_fixture(os.path.join(load_dir, 'dv_new.pt'), map_location=torch.device('cpu'))
    do_tensor = load_fixture(os.path.join(load_dir, 'do_tensor.pt'), map_location=torch.device('cpu'))
    B, T, H, D = q.shape
    scale = D ** (-0.5)
    chunk_size = 16
    from yb_7bff0dd87eff27 import chunk_ttt_linear_bwd_norm
    q_npu = q.to(DEVICE)
    k_npu = k.to(DEVICE)
    v_npu = v.to(DEVICE)
    v_new_npu = v_new.to(DEVICE)
    x_npu = x.to(DEVICE)
    y_npu = y.to(DEVICE)
    rstd_npu = rstd.to(DEVICE)
    w_npu = w.to(DEVICE)
    b_npu = b.to(DEVICE)
    eta_npu = eta.to(DEVICE)
    h0_npu = h0.to(DEVICE)
    hb0_npu = hb0.to(DEVICE)
    h_npu = h.to(DEVICE)
    dht_npu = dht.to(DEVICE)
    dhbt_npu = dhbt.to(DEVICE)
    dv_new_npu = dv_new.to(DEVICE)
    do_npu = do_tensor.to(DEVICE)
    npu_dh, npu_dhb, npu_dh0, npu_dhb0, npu_dv, npu_dk, npu_dw, npu_db = chunk_ttt_linear_bwd_norm(q_npu, k_npu, v_npu, v_new_npu, x_npu, y_npu, rstd_npu, w_npu, b_npu, eta_npu, h0_npu, hb0_npu, h_npu, dht_npu, dhbt_npu, dv_new_npu, do_npu, scale, None, chunk_size)
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({'dh': npu_dh, 'dhb': npu_dhb, 'dh0': npu_dh0, 'dhb0': npu_dhb0, 'dv': npu_dv, 'dk': npu_dk, 'dw': npu_dw, 'db': npu_db}, reference)

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
