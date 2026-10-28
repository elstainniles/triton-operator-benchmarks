"""Explicit manual tile/grid candidates; selection is not an autotuner."""
from .launch import launch as launch_kernel

CONFIGS = {
    "B256": {
        "B": 256
    },
    "B4096": {
        "B": 4096
    },
    "B16384": {
        "B": 16384
    }
}


def launch(case, variant):
    return launch_kernel(case, CONFIGS[variant])
