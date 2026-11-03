
from opbench.fla_fixtures import load_fixture
from yb_66daa30c773811 import *

def test_original_absolute():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_oja_bwd_kernel_dhu_blockdim64'
    q = load_fixture(os.path.join(load_dir, 'q.pt'), map_location=torch.device('cpu'))
    vg = load_fixture(os.path.join(load_dir, 'vg.pt'), map_location=torch.device('cpu'))
    w = load_fixture(os.path.join(load_dir, 'w.pt'), map_location=torch.device('cpu'))
    do = load_fixture(os.path.join(load_dir, 'do.pt'), map_location=torch.device('cpu'))
    dk = load_fixture(os.path.join(load_dir, 'dk.pt'), map_location=torch.device('cpu'))
    gv = load_fixture(os.path.join(load_dir, 'gv.pt'), map_location=torch.device('cpu'))
    h0 = load_fixture(os.path.join(load_dir, 'h0.pt'), map_location=torch.device('cpu'))
    dht = load_fixture(os.path.join(load_dir, 'dht.pt'), map_location=torch.device('cpu'))
    scale = load_fixture(os.path.join(load_dir, 'scale.pt')).item()
    chunk_size = load_fixture(os.path.join(load_dir, 'chunk_size.pt')).item()
    from yb_5a9aa3bd6a5163 import chunk_oja_bwd_dhu
    q_npu = q.to(DEVICE)
    vg_npu = vg.to(DEVICE)
    w_npu = w.to(DEVICE)
    do_npu = do.to(DEVICE)
    dk_npu = dk.to(DEVICE)
    gv_npu = gv.to(DEVICE)
    h0_npu = h0.to(DEVICE)
    dht_npu = dht.to(DEVICE)
    npu_dh, npu_dh0, npu_dk2 = chunk_oja_bwd_dhu(q_npu, vg_npu, w_npu, do_npu, dk_npu, gv=gv_npu, h0=h0_npu, dht=dht_npu, scale=scale, chunk_size=chunk_size)
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({'dh': npu_dh, 'dh0': npu_dh0, 'dk2': npu_dk2}, reference)

def compare_cpu64_outputs(outputs, references, masks=None):
    import json
    from pathlib import Path
    records = []
    failures = []
    for name, output in outputs.items():
        dtype = str(output.dtype).removeprefix('torch.')
        tolerance = 0.01
        actual = output.detach().cpu().double()
        expected = references[name].cpu().float().double()
        assert expected.dtype == torch.float64
        defined = torch.ones_like(expected, dtype=torch.bool) if masks is None else masks[name]
        actual, expected = (actual[defined], expected[defined])
        difference = (actual - expected).abs()
        allowed = tolerance + tolerance * expected.abs()
        record = dict(output=name, dtype=dtype, rtol=tolerance, atol=tolerance, defined_elements=actual.numel(), undefined_elements=output.numel() - actual.numel(), max_abs_error=float(difference.max()), mismatched_elements=int((~torch.isfinite(actual) | ~torch.isfinite(expected) | (difference > allowed)).sum()), reference_dtype='float64', reference_rounded_to_output_dtype=True)
        try:
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), 'non-finite defined output or reference'
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=False)
            record['status'] = 'passed'
        except AssertionError as error:
            record.update(status='precision_failed', error=str(error))
            failures.append(name)
        records.append(record)
    Path('cpu_fp64_accuracy.json').write_text(json.dumps(dict(validation_standard='original_absolute', accuracy_criteria_changed=False, original_validation_standard='original_absolute_0.01', outputs=records), indent=2) + '\n')
    assert not failures, 'CPU FP64 absolute validation failed for outputs: ' + ', '.join(failures)
