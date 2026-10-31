"""Fixed original autotune candidate, without autotuning overhead."""
from .launch import launch as launch_kernel

ID = "original_candidate_w4_s2"
CONFIG = {'num_warps': 4, 'num_stages': 2}

def launch(case):
    return launch_kernel(case, CONFIG)
