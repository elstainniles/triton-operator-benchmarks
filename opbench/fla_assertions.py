"""Observe original numeric acceptance in the same, single-target test call."""
import ast
import sys


class OriginalAssertions:
    def __init__(self, torch, original, paths, state):
        self.torch, self.original, self.state = torch, original, state
        self.asserts = {str(p): {n.lineno: n.test for n in ast.walk(ast.parse(p.read_text()))
                                if isinstance(n, ast.Assert)} for p in paths}
        self.pending, self.numeric_calls, self.comparisons = {}, set(), []

    def location(self):
        frame = sys._getframe(1)
        while frame:
            if frame.f_code.co_filename in self.asserts:
                return frame.f_code.co_filename, frame.f_lineno
            frame = frame.f_back
        return None

    def active(self):
        return (self.state['recording'] and self.state['observed']
                and not self.state.get('own_profile') and not self.state.get('tuning'))

    def has_tensor(self, value):
        if isinstance(value, self.torch.Tensor):
            return True
        if isinstance(value, (list, tuple)):
            return any(self.has_tensor(v) for v in value)
        if isinstance(value, dict):
            return any(self.has_tensor(v) for v in value.values())
        return False

    def close(self, function, *args, **kwargs):
        location = self.location() if self.active() else None
        actual = args[0] if args else kwargs.get('actual')
        expected = args[1] if len(args) > 1 else kwargs.get('expected')
        numeric = location and self.has_tensor(actual) and self.has_tensor(expected)
        try:
            result = function(*args, **kwargs)
        except AssertionError as error:
            if location:
                self.original.failed(error, *location)
            raise
        if numeric:
            self.original.close_passes += 1
            self.comparisons.append(dict(kind='torch.testing.assert_close', file=location[0],
                line=location[1], acceptance={k: v for k, v in kwargs.items()
                    if k in ('rtol', 'atol', 'equal_nan', 'check_dtype', 'check_device',
                             'check_layout', 'check_stride')}))
        return result

    def compare(self, function):
        def call(*args, **kwargs):
            location = self.location() if self.active() else None
            result = function(*args, **kwargs)
            if location and self.has_tensor((args, kwargs)):
                self.numeric_calls.add(location)
            return result
        return call

    def numeric_assert(self, frame, expression, location):
        if location in self.numeric_calls:
            return True
        comparisons = [n for n in ast.walk(expression) if isinstance(n, ast.Compare)
                       and all(not isinstance(op, (ast.Is, ast.IsNot, ast.In, ast.NotIn))
                               for op in n.ops)]
        for comparison in comparisons:
            if any(isinstance(n, ast.Attribute) and n.attr in {
                    'shape', 'size', 'stride', 'numel', 'dim', 'ndim', 'dtype', 'device',
                    'data_ptr', 'storage_offset', 'element_size'} for n in ast.walk(comparison)):
                continue
            for n in ast.walk(comparison):
                if isinstance(n, ast.Name):
                    value = frame.f_locals.get(n.id, frame.f_globals.get(n.id))
                    if isinstance(value, (self.torch.Tensor, float, complex)):
                        return True
        return False

    def trace(self, frame, event, arg):
        filename, line = frame.f_code.co_filename, frame.f_lineno
        if filename not in self.asserts:
            return None
        key = id(frame)
        if event == 'exception':
            self.pending.pop(key, None)
            if issubclass(arg[0], AssertionError) and self.state['recording']:
                self.original.failed(arg[1], filename, line)
            return self.trace
        previous = self.pending.get(key)
        if previous and (event == 'return' or (event == 'line' and line != previous[1])):
            self.pending.pop(key, None)
            expression = self.asserts[previous[0]][previous[1]]
            if self.numeric_assert(frame, expression, previous):
                self.original.passed_source_assertions.append({'file': previous[0], 'line': previous[1]})
                self.comparisons.append(dict(kind='source_assert', file=previous[0], line=previous[1],
                    acceptance={'expression': ast.unparse(expression)}))
        if event == 'line' and self.active() and line in self.asserts[filename]:
            self.numeric_calls.discard((filename, line))
            self.pending[key] = (filename, line)
        if event == 'return':
            self.pending.pop(key, None)
        return self.trace

    def summary(self, identities):
        identities = list(identities)
        checked = identities if len(identities) == 1 and self.comparisons else []
        comparisons = [dict(row, launch_identities=checked) for row in self.comparisons]
        failures = list(self.original.failures)
        status = 'failed' if failures else 'passed' if checked else 'unvalidated'
        return dict(status=status, checked_launch_identities=checked,
                    unchecked_launch_identities=[] if checked else identities,
                    comparisons=comparisons, failures=failures,
                    scope='Original numeric acceptance after the single target invocation in this process; no additional reference claim')
