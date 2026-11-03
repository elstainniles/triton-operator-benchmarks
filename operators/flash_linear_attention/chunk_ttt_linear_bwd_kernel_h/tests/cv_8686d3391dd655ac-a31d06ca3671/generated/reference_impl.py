"""Existing independent CPU equations; file-generation main omitted."""
def original_naive(folder, name):
    return (naive_chunk_ttt_linear_bwd_h, {'function': 'naive_chunk_ttt_linear_bwd_h', 'function_AST_unmodified': True})

def reference(cid, d):
    fn, origin = original_naive('chunk_ttt_linear_bwd_kernel_h', 'naive_chunk_ttt_linear_bwd_h')
    eps, bt = (d['eps_val'], d['chunk_size'])
    values = fn(*(d[n] for n in ('k', 'v', 'w', 'b', 'eta')), eps, d['h0'], d['hb0'], bt)
    out = dict(zip(('h', 'v2', 'x', 'y', 'r'), values))
    return (out, {'method': 'Original unmodified naive CPU64 state/intermediate reference; independent q-adjoint for fused bwd_h', **origin})

from original_cpu import naive_chunk_ttt_linear_bwd_h
