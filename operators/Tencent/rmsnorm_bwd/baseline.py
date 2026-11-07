"""Original attachment baseline; use compiler default launch options."""
from .launch import launch as launch_kernel

ID = "baseline"
CONFIG = {'GROUPS': 64, 'eps': 10.0}


def launch(case):
    return launch_kernel(case, CONFIG)
