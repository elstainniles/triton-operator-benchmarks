"""Fixed historical control; GPU dispatch/autotune is not benchmarked."""
from .launch import launch as launch_kernel

ID = "B1024"
CONFIG = {"B":1024}


def launch(case):
    return launch_kernel(case, CONFIG)
