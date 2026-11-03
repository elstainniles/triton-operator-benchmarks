"""Existing independent CPU equations; file-generation main omitted."""
def reference(case, contract, data):
    cid = case['id']
    outputs, checks = helper.wy_math(data)
    method = 'CPU64 triangular inverse checked against finite Neumann series and residual'
    return outputs

import math_helper as helper
