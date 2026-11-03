from original_cpu import naive_chunk_rwkv6_bwd_dh

def reference(cid, data):
    kwargs = {name: data[name] for name in ('q', 'gi', 'ge', 'dht', 'scale', 'T', 'HQ', 'H', 'K', 'V', 'BT', 'NG', 'STORE_INITIAL_STATE_GRADIENT', 'USE_FINAL_STATE_GRADIENT', 'IS_VARLEN')}
    kwargs.update(do_tensor=data['do'], cu_seqlens=None, chunk_offsets=None)
    dh, dh0 = naive_chunk_rwkv6_bwd_dh(**kwargs)
    return {'dh': dh.half().float(), 'dh0': dh0}
