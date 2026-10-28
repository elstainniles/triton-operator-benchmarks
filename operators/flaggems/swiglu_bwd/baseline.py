"""Fixed historical control; GPU dispatch/autotune is not benchmarked."""
from .launch import launch as launch_kernel

ID = "M64H64"
CONFIG = {"BM":64,"BH":64}


def launch(case):
    return launch_kernel(case, CONFIG)
