
from opbench.fla_fixtures import load_fixture
from yb_27029facd726c8 import *

def test_reference(load_dir, reference_path):
    data = load_fixture(os.path.join(load_dir, 'test_chunk_dplr_bwd_kernel_intra_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    dq, dk, da, db, dgk_output = chunk_dplr_bwd_dqk_intra(q=input_data['q'], k=input_data['k'], a=input_data['a'], b=input_data['b'], gi=input_data['gi'], ge=input_data['ge'], dAqk=input_data['dAqk'], dAqb=input_data['dAqb'], dAak=input_data['dAak'], dAab=input_data['dAab'], dqg=input_data['dqg'], dkg=input_data['dkg'], dag=input_data['dag'], dbg=input_data['dbg'], dgk_last=input_data['dgk_last'], scale=input_data['scale'], cu_seqlens=input_data['cu_seqlens'], chunk_size=input_data['chunk_size'], safe_gate=input_data['safe_gate'], chunk_indices=input_data['chunk_indices'])
    reference = load_fixture(reference_path, map_location='cpu', weights_only=False)
    values = {'dq': dq, 'dk': dk, 'da': da, 'db': db, 'dgk_output': dgk_output}
    from baseline_reference_checks import compare_original
    compare_original(values, reference, compare_data_precision)
