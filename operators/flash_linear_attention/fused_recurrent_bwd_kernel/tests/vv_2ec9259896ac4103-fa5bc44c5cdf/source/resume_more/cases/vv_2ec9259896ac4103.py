
from opbench.fla_fixtures import load_fixture
from resume_precision_contract import compare_raw
from yb_ecb4498f323529 import *

def fused_recurrent_iplr_delta_rule_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor, ha: torch.Tensor, do: torch.Tensor, dht: torch.Tensor, initial_state: torch.Tensor, scale: float, cu_seqlens: torch.Tensor | None=None):
    B, T, H, K, V = (*q.shape, v.shape[-1])
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = (triton.next_power_of_2(K), min(triton.next_power_of_2(V), 64))
    NV = triton.cdiv(V, BV)
    dq = q.new_empty(NV, *q.shape)
    dk = k.new_empty(NV, *k.shape)
    da = a.new_empty(NV, *a.shape)
    db = b.new_empty(NV, *b.shape)
    dv = torch.empty_like(v)
    dha = torch.empty_like(ha)
    grid = (NV, N * H)
    if initial_state is not None and initial_state.requires_grad:
        dh0 = torch.empty_like(initial_state, dtype=torch.float32)
    else:
        dh0 = None
    fused_recurrent_bwd_kernel[grid](q=q, k=k, v=v, a=a, b=b, ha=ha, dht=dht, dh0=dh0, do=do, dq=dq, dk=dk, dv=dv, da=da, db=db, dha=dha, h0=initial_state, scale=scale, cu_seqlens=cu_seqlens, B=B, H=H, T=T, K=K, V=V, BK=BK, BV=BV)
    compare_raw('vv_2ec9259896ac4103', REFERENCE, {'dq': dq, 'dk': dk, 'dv': dv, 'da': da, 'db': db, 'dha': dha}, compare_data_precision)
    dq = dq.sum(0)
    dk = dk.sum(0)
    da = da.sum(0)
    db = db.sum(0)
    return (dq.to(q), dk.to(k), dv.to(v), da.to(a), db.to(b), dh0)

def test_diagnostic(load_dir):
    data = load_fixture(os.path.join(load_dir, 'test_fused_recurrent_bwd_kernel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    global REFERENCE
    REFERENCE = load_fixture(os.path.join(load_dir, 'reference_cpu64.pt'), map_location='cpu', weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    dq, dk, dv, da, db, dh0 = fused_recurrent_iplr_delta_rule_bwd(q=input_data['q'], k=input_data['k'], v=input_data['v'], a=input_data['a'], b=input_data['b'], ha=input_data['ha'], do=input_data['do'], dht=input_data['dht'], initial_state=input_data['initial_state'], scale=input_data['scale'], cu_seqlens=input_data['cu_seqlens'])
