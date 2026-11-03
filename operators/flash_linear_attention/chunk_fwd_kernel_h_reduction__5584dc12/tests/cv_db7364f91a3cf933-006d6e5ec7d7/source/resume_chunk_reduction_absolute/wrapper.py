import torch
import yb_5584dc12a89f56 as original
original_compare=original.validate_cmp

def validate_checked(output, ref_output, name, atol=1e-3, rtol=1e-3):
    accepted=original_compare(output, ref_output, name, atol=atol, rtol=rtol)
    # The original test ignores validate_cmp's False return. Preserve its
    # comparison and surface the same tolerance as an observable assertion.
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol, equal_nan=False)
    return accepted

def test_reference():
    saved=original.validate_cmp
    original.validate_cmp=validate_checked
    try:
        return original.test_chunk_fwd_kernel_h_reduction()
    finally:
        original.validate_cmp=saved
