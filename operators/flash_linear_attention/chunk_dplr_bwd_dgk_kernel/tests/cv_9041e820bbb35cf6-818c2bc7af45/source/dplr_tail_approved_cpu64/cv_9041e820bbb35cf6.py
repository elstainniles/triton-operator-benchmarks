
from opbench.fla_fixtures import load_fixture
from yb_069b2bf2928455 import *

def test_cpu_fp64_absolute():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_dplr_bwd_kernel_intra_tensorcore/'
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs.pt'), map_location='cpu', weights_only=False)
    device = torch.device('npu')
    q = inputs_dict['q'].to(device)
    k = inputs_dict['k'].to(device)
    a = inputs_dict['a'].to(device)
    b = inputs_dict['b'].to(device)
    gi = inputs_dict['gi'].to(device)
    ge = inputs_dict['ge'].to(device)
    dAqk = inputs_dict['dAqk'].to(device)
    dAqb = inputs_dict['dAqb'].to(device)
    dAak = inputs_dict['dAak'].to(device)
    dAab = inputs_dict['dAab'].to(device)
    dqg = inputs_dict['dqg'].to(device)
    dkg = inputs_dict['dkg'].to(device)
    dag = inputs_dict['dag'].to(device)
    dbg = inputs_dict['dbg'].to(device)
    dgk_last = inputs_dict['dgk_last'].to(device)
    scale = inputs_dict['scale']
    cu_seqlens = inputs_dict['cu_seqlens']
    chunk_size = inputs_dict['chunk_size']
    safe_gate = inputs_dict['safe_gate']
    chunk_indices = inputs_dict['chunk_indices']
    dq, dk, da, db, dgk_output = chunk_dplr_bwd_dqk_intra(q=q, k=k, a=a, b=b, gi=gi, ge=ge, dAqk=dAqk, dAqb=dAqb, dAak=dAak, dAab=dAab, dqg=dqg, dkg=dkg, dag=dag, dbg=dbg, dgk_last=dgk_last, scale=scale, cu_seqlens=cu_seqlens, chunk_size=chunk_size, safe_gate=safe_gate, chunk_indices=chunk_indices)
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({'dq': dq, 'dk': dk, 'da': da, 'db': db, 'dgk_output': dgk_output}, reference)

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
