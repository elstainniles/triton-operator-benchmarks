"""In-memory, seeded data for the imported FLA tests; never read tensor files."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import PurePath
import copy

_ACTIVE = ContextVar('fla_fixtures', default=None)


def lookup(fixtures, name):
    key = str(name).replace('\\', '/')
    if key in fixtures:
        return fixtures[key]
    matches = [k for k in fixtures if key.endswith('/' + k.lstrip('/'))]
    if matches:
        return fixtures[max(matches, key=len)]
    matches = [k for k in fixtures if PurePath(k).name == PurePath(key).name]
    if len(matches) != 1:
        raise KeyError(f'Generated fixture is missing or ambiguous: {PurePath(key).name}')
    return fixtures[matches[0]]


def load_fixture(name, *args, map_location=None, **kwargs):
    """Replace a legacy deserialization call with a fresh in-memory payload.

    The legacy filename is only a dictionary key. Deepcopy preserves aliases,
    strides and repeated references, as loading the original payload did.
    """
    if args:
        if len(args) > 1 or map_location is not None:
            raise TypeError('Expected at most one map_location argument')
        map_location = args[0]
    fixtures = _ACTIVE.get()
    if fixtures is None:
        raise RuntimeError('Run this test with tools/run_fla_test.py to generate its inputs')
    value = copy.deepcopy(lookup(fixtures, name))
    # Historic tests request CPU here and explicitly move inputs to the NPU.
    # Support explicit destination strings without changing unrelated leaves.
    if map_location is not None and str(map_location) != 'cpu':
        import torch
        if isinstance(map_location, (str, torch.device)):
            def move(item):
                if isinstance(item, torch.Tensor):
                    return item.to(map_location)
                if isinstance(item, dict):
                    return {k: move(v) for k, v in item.items()}
                if isinstance(item, list):
                    return [move(v) for v in item]
                if isinstance(item, tuple):
                    return tuple(move(v) for v in item)
                return item
            value = move(value)
        else:
            raise ValueError('This imported test requires an unsupported map_location callback')
    return value


@contextmanager
def use_fixtures(fixtures):
    if not isinstance(fixtures, dict):
        raise ValueError('A fixture provider must return a dictionary')
    token = _ACTIVE.set(fixtures)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
