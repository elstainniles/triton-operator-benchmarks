"""Existing independent CPU equations; file-generation main omitted."""
import torch

def reference(case, contract, data):
    outputs = helper.reference('dplr_dv', data)
    alternate = helper.reference('dplr_dv', data, True)
    torch.testing.assert_close(outputs['dv'], alternate['dv'], rtol=1e-10, atol=1e-10)
    bt = int(data['chunk_size'])
    a = data['A_qk'][0, :bt, 0].double().tril()
    kg = data['kg'][0, :bt, 0].double()
    do = data['do'][0, :bt, 0].double()
    dh = data['dh'][0, 0, 0].double()
    v = torch.zeros_like(do, requires_grad=True)
    loss = (a @ v * do).sum() + (kg.T @ v * dh).sum()
    grad = torch.autograd.grad(loss, v)[0]
    target = outputs['dv'][0, :bt, 0]
    torch.testing.assert_close(target, grad, rtol=1e-10, atol=1e-10)
    return outputs

import math_helper as helper
