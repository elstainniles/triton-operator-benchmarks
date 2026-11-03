"""Existing independent CPU equations; file-generation main omitted."""
def original_naive(folder, name):
    return (naive_fused_chunk_ttt_linear_fwd, {'function': 'naive_fused_chunk_ttt_linear_fwd', 'function_AST_unmodified': True})

def reference(cid, d):
    fn, origin = original_naive('fused_chunk_ttt_linear_fwd_kernel', 'naive_fused_chunk_ttt_linear_fwd')
    values = fn(*(d[n] for n in ('q', 'k', 'v', 'w', 'b', 'eta', 'sc', 'eps_val', 'h0', 'hb0', 'out_final', 'bt')))
    return (dict(zip(('o', 'ht', 'hbt'), values)), {'method': 'Original unmodified naive CPU FP64 forward reference', **origin})

from original_cpu import naive_fused_chunk_ttt_linear_fwd
