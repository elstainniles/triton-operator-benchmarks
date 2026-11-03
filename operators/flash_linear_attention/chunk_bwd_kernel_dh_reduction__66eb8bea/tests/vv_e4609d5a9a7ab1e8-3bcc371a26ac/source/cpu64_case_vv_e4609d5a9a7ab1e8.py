
from opbench.fla_fixtures import load_fixture
from yb_43e166b2d13510 import *

def launch_only(load_dir, reference_path):
    reference = load_fixture(reference_path, map_location='cpu', weights_only=False)
    data = load_fixture(os.path.join(load_dir, 'test_chunk_h_parallel_chunk_bwd_kernel_dh_reduction_v2.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = data['input_data']
    B = input_data['B']
    T = input_data['T']
    H = input_data['H']
    HQ = input_data['HQ']
    K = input_data['K']
    V = input_data['V']
    BT = input_data['BT']
    NG = input_data['NG']
    g_cumsum = input_data['g_cumsum'].to('npu')
    dh = input_data['dh'].to('npu')
    doq0 = input_data['doq0']
    h0 = input_data['h0']
    dh0 = torch.empty(B, HQ, K, V, dtype=torch.float32, device='npu') if h0 is not None else None

    def grid_reduction(meta):
        return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), B * HQ)
    chunk_bwd_kernel_dh_reduction[grid_reduction](g=g_cumsum, gk=None, gv=None, dh=dh, doq0=doq0, dh0=dh0, cu_seqlens=None, chunk_offsets=None, T=T, HQ=HQ, H=H, K=K, V=V, BT=BT, NG=NG, USE_G=g_cumsum is not None, USE_GK=False, USE_GV=False)
    return ({'dh': dh}, reference)

def test_cpu64(load_dir, reference_path):
    from cpu64_absolute import compare_named
    values, reference = launch_only(load_dir, reference_path)
    compare_named(torch, values, reference)
