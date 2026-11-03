
from opbench.fla_fixtures import load_fixture
from yb_58af370c8c7b7d import *
from resume_precision_contract import compare_raw
from yb_e7a3f4d28ef6ad import *

@input_guard
def checked_forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float | None=None, initial_state: torch.Tensor | None=None, output_final_state: bool=False, cu_seqlens: torch.LongTensor | None=None):
    B, T, H, K, V = (*k.shape, v.shape[-1])
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    if output_final_state:
        final_state = q.new_empty(B, H, K, V, dtype=torch.float32)
    else:
        final_state = None
    ha = torch.empty_like(v, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(V, meta['BV']), N * H)
    o = torch.empty_like(v)
    fused_recurrent_fwd_kernel[grid](q=q, k=k, v=v, a=a, b=b, o=o, ha=ha, h0=initial_state, ht=final_state, scale=scale, cu_seqlens=cu_seqlens, H=H, T=T, K=K, V=V, BK=BK)
    compare_raw('vv_bea43e08f1b68a1e', REFERENCE, {'o': o, 'ha': ha}, compare_data_precision)
    ctx.save_for_backward(q, k, v, a, b, ha, initial_state)
    ctx.scale = scale
    ctx.cu_seqlens = cu_seqlens
    return (o, final_state)

def test_diagnostic(load_dir):
    data = load_fixture(os.path.join(load_dir, 'test_fused_recurrent_fwd_kernel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    global REFERENCE
    REFERENCE = load_fixture(os.path.join(load_dir, 'reference_cpu64.pt'), map_location='cpu', weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    from unittest.mock import patch
    with patch.object(FusedRecurrentIPLRDeltaRuleFunction, 'forward', staticmethod(checked_forward)):
        o, final_state = fused_recurrent_iplr_delta_rule(q=input_data['q'], k=input_data['k'], v=input_data['v'], a=input_data['a'], b=input_data['b'], scale=input_data['scale'], initial_state=input_data['initial_state'], output_final_state=input_data['output_final_state'], cu_seqlens=input_data['cu_seqlens'])
