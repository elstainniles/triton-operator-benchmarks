"""Preserve the original test while routing its single optional-file query."""
import os
from pathlib import Path
from unittest.mock import patch
import yb_75a9ad0bf765c5 as original
import yb_18dc009e5fff3c as original_kda_module

ORIGINAL_DBIAS = '/data/PytorchFile/Inference_or_Training_solution/fla_vv_ops_2/kda_gate_bwd_kernel/dbias.pt'


def run_test(data_dir):
    local_dbias = Path(data_dir).resolve() / 'dbias.pt'
    if not local_dbias.is_file():
        raise RuntimeError('Missing independent dbias reference; all outputs must be checked')
    exists_before = os.path.exists
    original_kernel_call = original.kda_gate_bwd
    original_validate = original.validate_cmp
    evidence = {'mapped_exists_queries': 0, 'comparisons': 0}

    def mapped_exists(path):
        if os.fspath(path) == ORIGINAL_DBIAS:
            evidence['mapped_exists_queries'] += 1
            return exists_before(local_dbias)
        return exists_before(path)

    def call_checked(*args, **kwargs):
        result = original_kernel_call(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 3 or result[2] is None:
            raise RuntimeError('Bias input must produce a non-None dbias output')
        return result

    def validate_checked(*args, **kwargs):
        result = original_validate(*args, **kwargs)
        evidence['comparisons'] += 1
        return result

    with patch('os.path.exists', side_effect=mapped_exists), \
            patch.object(original, 'kda_gate_bwd', side_effect=call_checked), \
            patch.object(original, 'validate_cmp', side_effect=validate_checked):
        original.test_kda_gate_bwd_kernel()
    if evidence['mapped_exists_queries'] != 1 or evidence['comparisons'] != 3:
        raise RuntimeError('Expected one dbias existence query and three successful output comparisons: ' + repr(evidence))
    print('KDA_CHECKED_OUTPUTS=dg,dA,dbias')
