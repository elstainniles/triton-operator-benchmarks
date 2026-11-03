"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def bitonic(scores):
    width = scores.shape[-1]
    assert width > 0 and width & width - 1 == 0
    levels = width.bit_length() - 1
    positions = torch.arange(width)
    ids = positions.expand_as(scores).clone()
    values = scores.clone()
    for stage in range(1, levels + 1):
        flip = (positions >> stage & 1).bool() if stage < levels else torch.ones(width, dtype=torch.bool)
        for distance in [1 << i for i in range(stage - 1, -1, -1)]:
            left = positions & ~distance
            right = left | distance
            swap = (values[..., left] > values[..., right]) != flip
            values = torch.where(swap, values[..., positions ^ distance], values)
            ids = torch.where(swap, ids[..., positions ^ distance], ids)
    return (values, ids)

def scalar_bitonic(scores):
    width = len(scores)
    levels = width.bit_length() - 1
    values = list(scores)
    ids = list(range(width))
    for stage in range(1, levels + 1):
        for step in range(stage - 1, -1, -1):
            distance = 1 << step
            for left in range(width):
                if left & distance:
                    continue
                right = left + distance
                descending = bool(left >> stage & 1) if stage < levels else True
                if (values[left] > values[right]) != descending:
                    values[left], values[right] = (values[right], values[left])
                    ids[left], ids[right] = (ids[right], ids[left])
    return (values, ids)

def importance(d, dtype):
    b, t, h, hq, g, k, bs, bc = (int(d[key]) for key in ('B', 'T', 'H', 'HQ', 'G', 'K', 'BS', 'BC'))
    assert h * g == hq and t % bs == 0 and ((t + bs - 1) // bs <= bc)
    assert all((d[key] is None for key in ('cu_seqlens', 'token_indices', 'chunk_offsets')))
    q = (d['q'].float() * d['scale']).to(d['q'].dtype).to(dtype).reshape(b, t, h, g, k)
    keys = d['k_cmp'].to(dtype).permute(0, 2, 1, 3)
    logits = torch.einsum('bthgk,bhck->bthgc', q, keys)
    indices = torch.arange(bc)[None, None, None, None, :]
    current = (torch.arange(t) // bs)[None, :, None, None, None]
    logits = torch.where(indices < current, logits, -torch.inf)
    lse = d['lse'].to(dtype).reshape(b, t, h, g, 1)
    forced = (indices == 0) | (indices == current - 1) | (indices == current)
    probabilities = torch.where(forced, 1.0, torch.exp(logits - lse))
    scores = probabilities.sum(-2)
    assert torch.isfinite(scores).all()
    return (scores, q, keys, lse)

def reference(d):
    scores, q, keys, lse = importance(d, torch.float64)
    sorted_scores, ids = bitonic(scores)
    current = (torch.arange(d['T']) // d['BS'])[None, :, None, None]
    ids = torch.where(ids <= current, ids, -1)[..., :d['S']].to(torch.int32)
    assert torch.all(sorted_scores[..., :-1] >= sorted_scores[..., 1:])
    float_scores, *_ = importance(d, torch.float32)
    _, float_ids = bitonic(float_scores)
    float_ids = torch.where(float_ids <= current, float_ids, -1)[..., :d['S']].to(torch.int32)
    differing = (ids != float_ids).nonzero()
    rounding_sensitive_rows = []
    for b, t, h in torch.unique(differing[:, :3], dim=0).tolist():
        positions = (ids[b, t, h] != float_ids[b, t, h]).nonzero().flatten().tolist()
        rounding_sensitive_rows.append(dict(batch=b, token=t, head=h, positions=positions, cpu64_indices=ids[b, t, h].tolist(), cpu32_indices=float_ids[b, t, h].tolist(), cpu64_scores=scores[b, t, h].tolist(), cpu32_scores=float_scores[b, t, h].tolist()))
    errors = []
    rows = []
    for b, t, h in [(0, 0, 0), (0, 31, 1), (0, 32, 0), (1, 63, 1), (1, 511, 0), (2, 1023, 1)]:
        ic = t // d['BS']
        scalar = []
        for block in range(d['BC']):
            if block in (0, ic - 1, ic):
                value = float(d['G'])
            elif block < ic:
                value = math.fsum((math.exp(math.fsum((float(q[b, t, h, group, j]) * float(keys[b, h, block, j]) for j in range(d['K']))) - float(lse[b, t, h, group, 0])) for group in range(d['G'])))
            else:
                value = 0.0
            scalar.append(value)
        error = max((abs(value - float(scores[b, t, h, j])) for j, value in enumerate(scalar)))
        assert error < 1e-10
        expected_scores, expected_ids = scalar_bitonic(scalar)
        expected = torch.tensor([j if j <= ic else -1 for j in expected_ids[:d['S']]], dtype=torch.int32)
        assert torch.equal(expected, ids[b, t, h])
        errors.append(error)
        rows.append(dict(batch=b, token=t, head=h, indices=expected.tolist()))
    for fixture in [[1.0] * d['BC'], [16.0] + [0.0] * (d['BC'] - 1), [16.0, 16.0, 16.0] + [0.0] * (d['BC'] - 3)]:
        _, got = bitonic(torch.tensor(fixture, dtype=torch.float64))
        _, expected = scalar_bitonic(fixture)
        assert got.tolist() == expected
    return ({'block_indices': ids}, dict(method='Independent CPU64 compressed-attention importance scores; original FP16 q-scale cast; explicit bitonic compare-and-swap ordering', reference_dtype='int32', comparison_reference_dtype='int32', mathematical_scores_dtype='float64', original_q_scale_cast_preserved=True, forced_blocks='first block 0 and last two blocks IC-1, IC score exactly G', invalid_indices='positions > IC become -1; zero scores participate in original bitonic ordering', tie_rule='compare_and_swap uses (left > right) != flip; equal pairs exchange in descending phases', original_selection_loops=1, cpu64_cpu32_indices_exact_match=not bool(len(differing)), cpu64_cpu32_index_mismatch_count=int(len(differing)), rounding_sensitive_rows=rounding_sensitive_rows, numerical_sensitivity_note='Strict original index comparison is retained for every output; close-score ordering differences are recorded, never masked or accepted as precision pass', scalar_max_abs_score_error=max(errors), scalar_rows=rows, tie_fixtures_passed=3, comparison_standard='Original validate_cmp float16 branch: assert_close int32 indices, rtol=atol=1e-3', actual_index_range=[int(ids.min()), int(ids.max())]))
