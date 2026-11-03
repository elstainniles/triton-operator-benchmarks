"""Seeded CPU inputs and independent reference; no external tensors."""
import torch
from recipe import make_fixtures as _build

def make_fixtures():
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        return _build()
    finally:
        torch.set_num_threads(previous)
