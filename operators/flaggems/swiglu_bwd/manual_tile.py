"""Explicit manual tile/grid candidates; selection is not an autotuner."""
from .launch import launch as launch_kernel

CONFIGS = {
    "M1H512": {
        "BM": 1,
        "BH": 512
    },
    "M4H512": {
        "BM": 4,
        "BH": 512
    },
    "M16H256": {
        "BM": 16,
        "BH": 256
    },
    "M4H2048": {
        "BM": 4,
        "BH": 2048
    }
}


def launch(case, variant):
    return launch_kernel(case, CONFIGS[variant])
