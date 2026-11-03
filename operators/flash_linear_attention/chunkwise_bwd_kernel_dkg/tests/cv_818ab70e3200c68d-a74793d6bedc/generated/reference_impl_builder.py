"""Existing independent CPU equations; file-generation main omitted."""
def tile(t, bt):
    b, n, h, d = t.shape
    assert n % bt == 0
    return t.double().reshape(b, n // bt, bt, h, d).permute(0, 1, 3, 2, 4)

def untile(t):
    b, n, h, bt, d = t.shape
    return t.permute(0, 1, 3, 2, 4).reshape(b, n * bt, h, d)

def scalar_tile(t, bt):
    return tile(t.unsqueeze(-1), bt).squeeze(-1)
