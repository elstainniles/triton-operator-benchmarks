"""Existing independent CPU equations; file-generation main omitted."""
def tile(t, bt):
    b, n, h, d = t.shape
    assert n % bt == 0
    return t.double().reshape(b, n // bt, bt, h, d).permute(0, 1, 3, 2, 4)
