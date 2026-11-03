
from opbench.fla_fixtures import load_fixture
import torch
import torch_npu
from yb_4f08bd3c58af0a import chunk_fwd_mesa_cg_dim64_kernel

def to_npu(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to('npu')
        else:
            out[k] = v
    return out

def compare(name, a, b):
    abs_diff = (a - b).abs()
    rel_diff = abs_diff / (b.abs() + 1e-07)
    print(f'{name}: max_abs={abs_diff.max().item():.6e}, max_rel={rel_diff.max().item():.6e}, mean_abs={abs_diff.mean().item():.6e}')

def test_chunk_fwd_mesa_cg_dim64_kernel_vs_torch():
    data = load_fixture('/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_mesa_cg_fwd/test_chunk_mesa_cg_dim64_kernel_v2.pt', map_location='cpu', weights_only=False)
    input_data = data['input_data']
    npu_input = to_npu(data['input_data'])
    chunk_fwd_mesa_cg_dim64_kernel[data['grid']](q=npu_input['q'], q_final=npu_input['q_final'], k=npu_input['k'], h=npu_input['h'], o=npu_input['o'], v=npu_input['v'], h_kv=npu_input['h_kv'], g=npu_input['g'], beta=npu_input['beta'], lamb=npu_input['lamb'], cu_seqlens=npu_input['cu_seqlens'] if npu_input['cu_seqlens'].numel() > 0 else None, chunk_indices=npu_input['chunk_indices'] if npu_input['chunk_indices'].numel() > 0 else None, T=npu_input['T'], max_CG_iteration=npu_input['max_CG_iteration'], H=npu_input['H'], K=npu_input['K'], BT=npu_input['BT'], BK=npu_input['BK'])
    torch.npu.synchronize()
    npu_q_final = npu_input['q_final'].cpu()
    npu_o = npu_input['o'].cpu()
    from yb_4e3b7af0db0a21 import chunk_mesa_cg_fwd_pt
    q_final_pt, o_pt = chunk_mesa_cg_fwd_pt(input_data['q'], input_data['k'], input_data['v'], input_data['h'], input_data['h_kv'], input_data['g'], input_data['beta'], input_data['lamb'], chunk_size=input_data['BT'], max_CG_iteration=input_data['max_CG_iteration'])
    print('\n=== PT native (matmul) vs NPU Triton ===')
    compare('q_final', q_final_pt, npu_q_final)
    compare('o', o_pt, npu_o)
    print('\n=== allclose check (rtol=5e-3, atol=5e-3) ===')
    pt_npu_q = torch.allclose(q_final_pt, npu_q_final, 0.005, 0.005)
    pt_npu_o = torch.allclose(o_pt, npu_o, 0.005, 0.005)
    print(f'PT(bf16) vs NPU q_final: {pt_npu_q}')
    print(f'PT(bf16) vs NPU o:       {pt_npu_o}')
    assert pt_npu_q and pt_npu_o, f'PT vs NPU failed: q_final={pt_npu_q}, o={pt_npu_o}'
    print('\nAll precision checks passed!')
