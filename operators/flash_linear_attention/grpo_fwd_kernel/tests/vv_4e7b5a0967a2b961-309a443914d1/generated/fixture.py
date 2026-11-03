"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: grpo_fwd
"""
def base_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'grpo_fwd'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    logits = torch.randn((2, 1025, 32000), generator=gen, dtype=torch.bfloat16)
    ids = torch.randint(0, 31999, (2, 1088), generator=gen, dtype=torch.int64)
    ref = torch.randn((2, 1024), generator=gen)
    adv = torch.randn((2,), generator=gen)
    mask = torch.ones((2, 1024), dtype=torch.int32)
    mask[::2, 1024 // 3:1024 // 2] = 0
    loss = torch.empty((2, 1024))
    kl = torch.empty_like(loss)
    for b in range(2):
        for start in range(0, 1024, 64):
            x = logits[b, start:start + 64].double()
            target = ids[b, 64 + start:64 + start + 64]
            logp = x.gather(1, target[:, None]).squeeze(1) - torch.logsumexp(x, dim=-1)
            diff = ref[b, start:start + 64].double() - logp
            rr = torch.exp(diff) - diff - 1
            active = mask[b, start:start + 64].bool()
            kl[b, start:start + 64] = torch.where(active, rr, 0).float()
            loss[b, start:start + 64] = torch.where(active, rr * 0.04 - adv[b].double(), 0).float()
    inp = dict(logits=logits, ref_log_probs=ref, input_ids=ids, advantages=adv, completion_mask=mask)
    for name, obj in [('grpo_fwd_kernel_in.pt', inp), ('grpo_fwd_kernel_out.pt', dict(loss=loss, independent_kl=kl))]:
        path = folder / name
        fixtures[path.name] = obj
    return fixtures

def make_fixtures():
    import torch
    values=base_fixtures()
    v=values['grpo_fwd_kernel_in.pt'];old=values['grpo_fwd_kernel_out.pt']
    lse=torch.zeros((2,1024),dtype=torch.float32)
    for b in range(2):
        for start in range(0,1024,32):
            x=v['logits'][b,start:start+32].double();m=x.max(-1).values
            z=m+(x-m[:,None]).exp().sum(-1).log()
            active=v['completion_mask'][b,start:start+32].bool()
            lse[b,start:start+32]=torch.where(active,z,0).float()
    values['input.pt']=v
    values['reference.pt']=dict(loss=old['loss'],independent_kl=old['independent_kl'],packed_false=old['loss'],packed_true=torch.cat((old['loss'],old['independent_kl']),0),lse=lse)
    return values
