
from opbench.fla_fixtures import load_fixture
from yb_46d16242a62a42 import *

def test_diagnostic(load_dir):
    inputs = load_fixture(os.path.join(load_dir, f'{kernel_name}_in.pt'), map_location=torch.device('cpu'), weights_only=False)
    s_cpu = inputs['s']
    s = s_cpu.to(DEVICE)
    npu_output = chunk_global_cumsum_vector(s)
    raise RuntimeError('DIAGNOSTIC_LAUNCH_FINISHED: blocked_missing_gpu_comparison_reference; original GPU-error-ratio acceptance unavailable; no accuracy or timing success claimed')
