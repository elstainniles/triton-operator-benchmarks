
from opbench.fla_fixtures import load_fixture
from yb_46d16242a62a42 import *

def test_reference(load_dir, reference_path):
    inputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_in.pt'), map_location=torch.device('cpu'), weights_only=False)
    s_cpu = inputs['s']
    s = s_cpu.to(DEVICE)
    npu_output = chunk_global_cumsum_vector(s)
    reference = load_fixture(reference_path, map_location='cpu', weights_only=False)
    values = {'output': npu_output}
    from baseline_reference_checks import compare_cpu64
    compare_cpu64(values, reference)
