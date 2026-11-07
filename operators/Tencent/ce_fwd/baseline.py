"""Original attachment baseline; use compiler default launch options."""
from .launch import launch as launch_kernel

ID = "baseline"
CONFIG = {'BLOCK_SIZE': 2048}


def launch(case):
    return launch_kernel(case, CONFIG)
