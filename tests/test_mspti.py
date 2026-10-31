import unittest
from types import SimpleNamespace

from opbench.mspti import measure


class MSPTITest(unittest.TestCase):
    def run_measurement(self, events):
        state = {}
        class Monitor:
            def start(self, callback):
                state['callback'] = callback
            def stop(self):
                state.pop('callback')
        def launch():
            for name, start, end in events:
                state['callback'](SimpleNamespace(name=name, start=start, end=end))
        return measure([('baseline', launch, 'target')], lambda: None, Monitor)

    def test_samples_and_units(self):
        result = self.run_measurement([('target', 100, 2100)])
        variant = result['variants']['baseline']
        self.assertEqual(variant['median_us'], 2.0)
        self.assertEqual(len(variant['samples']), 15)
        self.assertEqual(len(variant['rounds']), 3)

    def test_rejects_missing_extra_wrong_and_invalid_records(self):
        for records in ([], [('other', 0, 100)], [('target', 0, 1), ('target', 1, 2)],
                        [('target', 1, 1)]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                self.run_measurement(records)

    def test_stops_monitor_on_launch_error(self):
        state = []
        class Monitor:
            def start(self, callback):
                state.append('start')
            def stop(self):
                state.append('stop')
        def fail():
            raise RuntimeError('launch failed')
        with self.assertRaises(RuntimeError):
            measure([('baseline', fail, 'target')], lambda: None, Monitor)
        self.assertEqual(state, ['start', 'stop'])
