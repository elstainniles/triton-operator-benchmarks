
"""One original GRPO launch per save_kl specification. No kernel/tile rewrite."""
from opbench.fla_fixtures import load_fixture
import os
import torch
from yb_3e77d2739161a0 import grpo_fwd_kernel
from yb_294bdc0c5fab75 import validate_cmp
from blue_adapters.grpo.contract import region

def original_forward_with_lse(logits, ref_logp, input_ids, advantages, beta, completion_mask, save_kl, inplace=True):
    input_shape = logits.shape
    B, L_ADD_1, N = input_shape
    L = L_ADD_1 - 1
    M = B * L
    input_ids_start_index = input_ids.size(1) - L
    if not save_kl:
        loss = torch.empty(B, L, device=logits.device, dtype=torch.float32)
    else:
        loss = torch.empty(B * 2, L, device=logits.device, dtype=torch.float32)
    lse = torch.empty(B, L, device=logits.device, dtype=torch.float32)
    if completion_mask is None:
        completion_mask = torch.ones(B, L, device=logits.device, dtype=torch.int32)
    else:
        loss[:B].masked_fill_(completion_mask.logical_not(), 0.0)
    grpo_fwd_kernel[M,](logits_ptr=logits, ref_logp_ptr=ref_logp, input_ids_ptr=input_ids, advantages_ptr=advantages, completion_mask_ptr=completion_mask, loss_ptr=loss, lse_ptr=lse, beta=beta, save_kl=save_kl, B=B, M=M, N=N, L=L, start_idx=input_ids_start_index)
    return (loss, lse)

def test_grpo_spec(load_dir, save_kl):
    inp = load_fixture(os.path.join(load_dir, 'input.pt'), map_location='cpu', weights_only=True)
    ref = load_fixture(os.path.join(load_dir, 'reference.pt'), map_location='cpu', weights_only=True)
    device = 'npu:0'
    logits = inp['logits'].to(device).requires_grad_(True)
    ref_logp = inp['ref_log_probs'].to(device)
    ids = inp['input_ids'].to(device)
    advantages = inp['advantages'].to(device)
    mask = inp['completion_mask'].to(device)
    packed, lse = original_forward_with_lse(logits, ref_logp, ids, advantages, 0.04, mask, save_kl, inplace=False)
    loss = packed[:2]
    with region('original_loss'):
        validate_cmp('bfloat16', loss, ref['loss'].to(device))
    if save_kl:
        with region('added_kl'):
            torch.testing.assert_close(packed[2:], ref['independent_kl'].to(device), rtol=1e-05, atol=1e-05, equal_nan=False)
    with region('packed_storage'):
        torch.testing.assert_close(packed, ref['packed_true' if save_kl else 'packed_false'].to(device), rtol=0.005, atol=0.005, equal_nan=False)
    with region('added_lse'):
        torch.testing.assert_close(lse, ref['lse'].to(device), rtol=1e-05, atol=1e-05, equal_nan=False)
