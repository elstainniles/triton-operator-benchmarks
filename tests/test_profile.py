import copy
from decimal import Decimal
import unittest

from opbench.profile import correlate


def fixture():
    events, rows, labels = [], [], []
    for i, duration in enumerate([1, 100, 100, 2, 2, 2]):
        label = f'opbench::B1024::{i//3}::{i%3}'
        labels.append(label)
        cpu = Decimal('10.0000000001') + i*10
        device = Decimal('900000000000.000001') + i*1000
        events.extend([
            dict(ph='X', name=label, pid=1, tid=2, ts=cpu, dur=5),
            dict(ph='s', cat='async_npu', id=i, pid=1, tid=2, ts=cpu+1),
            dict(ph='f', cat='async_npu', id=i, pid=9, tid=9, ts=device)])
        rows.append({'Start Time(us)':str(device), 'Duration(us)':str(duration), 'Name':'mul_kernel'})
    return events, rows, labels


class ProfileTests(unittest.TestCase):
    def test_exact_decimal_mapping_and_round_statistic(self):
        events, rows, labels = fixture()
        result = correlate(events, rows, labels, 'mul_kernel')['variants']['B1024']
        self.assertEqual(result['median_us'], 51)
        self.assertEqual(len(result['samples']), 6)

    def test_missing_call_rejected(self):
        events, rows, labels = fixture()
        with self.assertRaises(ValueError):
            correlate(events[:-3], rows[:-1], labels, 'mul_kernel')

    def test_duplicate_timestamp_rejected(self):
        events, rows, labels = fixture()
        rows.append(copy.deepcopy(rows[0]))
        with self.assertRaises(ValueError):
            correlate(events, rows, labels, 'mul_kernel')

    def test_unknown_kernel_rejected(self):
        events, rows, labels = fixture()
        rows[0]['Name'] = 'pytorch_add'
        with self.assertRaises(ValueError):
            correlate(events, rows, labels, 'mul_kernel')

    def test_unmatched_flow_rejected(self):
        events, rows, labels = fixture()
        events.pop()
        with self.assertRaises(ValueError):
            correlate(events, rows, labels, 'mul_kernel')

    def test_multi_kernel_direct_call_rejected(self):
        events, rows, labels = fixture()
        start = dict(events[1], id=99)
        end = dict(events[2], id=99, ts=Decimal('999999999999.9'))
        events += [start, end]
        rows.append({'Start Time(us)':str(end['ts']), 'Duration(us)':'1.2', 'Name':'mul_kernel'})
        with self.assertRaises(ValueError):
            correlate(events, rows, labels, 'mul_kernel')


if __name__ == '__main__':
    unittest.main()
