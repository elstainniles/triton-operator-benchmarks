"""Existing independent CPU equations; file-generation main omitted."""
import torch
IDS = ('cv_b68fd39646541eae', 'cv_60edb64ab2e833aa', 'cv_f17617e5f9b9b0a0')

def chunks(tensor, chunk_size):
    batch, time, heads, width = tensor.shape
    assert time % chunk_size == 0
    return tensor.reshape(batch, time // chunk_size, chunk_size, heads, width).permute(0, 1, 3, 2, 4)

def unchunk(tensor):
    batch, count, heads, length, width = tensor.shape
    return tensor.permute(0, 1, 3, 2, 4).reshape(batch, count * length, heads, width)

def forward_output(inputs):
    block = inputs['chunk_size']
    q, v, g, attention, state = (inputs[name] for name in ('q', 'v', 'g', 'A', 'h'))
    batch, time, heads, width = q.shape
    count = time // block
    gated_q = (q.float() * torch.exp2(g.float())).to(q.dtype)
    q_tiles = chunks(gated_q, block).double()
    h_tiles = state.to(q.dtype).reshape(batch, count, heads, width, v.shape[-1]).double()
    cross = (q_tiles @ h_tiles).float() * inputs['scale']
    a_tiles = torch.tril(chunks(attention, block)).to(v.dtype).double()
    local = (a_tiles @ chunks(v, block).double()).float()
    return {'o': unchunk((cross + local).to(v.dtype))}

def validate_scalar_points(inputs, outputs, case_id):
    points = [(0, 0, 0, 0), (0, 17, 1, 7), (1, inputs['q'].shape[1] - 1, 2, 15)]
    errors = []
    block = inputs['chunk_size']
    for batch, time, head, coordinate in points:
        start = time // block * block
        if case_id == IDS[0]:
            q, k, g, da = (inputs[name].double() for name in ('q', 'k', 'g', 'dA'))
            before = torch.arange(start, time + 1)
            after = torch.arange(time, start + block)
            dq = (da[batch, time, head, before - start] * k[batch, before, head, coordinate] * torch.exp2(g[batch, time, head, coordinate] - g[batch, before, head, coordinate])).sum()
            dk = (da[batch, after, head, time - start] * q[batch, after, head, coordinate] * torch.exp2(g[batch, after, head, coordinate] - g[batch, time, head, coordinate])).sum()
            for key, value in [('dq_fp64', dq), ('dk_fp64', dk)]:
                actual = outputs[key][batch, time, head, coordinate]
                torch.testing.assert_close(actual, value, rtol=1e-12, atol=1e-12)
                errors.append(float(abs(actual - value)))
        elif case_id == IDS[1]:
            time = start + 32
            column = 7
            q, k, g = (inputs[name].double() for name in ('q', 'k', 'g'))
            value = (q[batch, time, head] * k[batch, start + column, head] * torch.exp2(g[batch, time, head] - g[batch, start + column, head])).sum() * inputs['scale']
            actual = outputs['A_fp64'][batch, time, head, column]
            torch.testing.assert_close(actual, value, rtol=1e-12, atol=1e-12)
            errors.append(float(abs(actual - value)))
        else:
            q, v, g, attention, state = (inputs[name] for name in ('q', 'v', 'g', 'A', 'h'))
            qg = (q[batch, time, head].float() * torch.exp2(g[batch, time, head].float())).half().double()
            h = state[batch * (q.shape[1] // block) + time // block, head, :, coordinate].half().double()
            cross = (qg * h).sum().float() * inputs['scale']
            a = attention[batch, time, head, :time - start + 1].half().double()
            local = (a * v[batch, start:time + 1, head, coordinate].double()).sum().float()
            value = (cross + local).half()
            actual = outputs['o'][batch, time, head, coordinate]
            torch.testing.assert_close(actual, value, rtol=0, atol=0)
            errors.append(float(abs(actual - value)))
    return {'checked_values': len(errors), 'max_abs_difference': max(errors), 'passed': True}
