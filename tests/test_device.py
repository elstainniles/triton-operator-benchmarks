import unittest

from opbench.device import assess


class DeviceTests(unittest.TestCase):
    def test_exact_driver_host_pid_accepted(self):
        self.assertTrue(assess('Health Status : OK', 'Process id : 12345', 12345)['eligible'])

    def test_container_pid_numerical_collision_not_ownership(self):
        # The container-local PID could be 42. It is not the verified host TGID.
        self.assertFalse(assess('Health Status : OK', 'Process id : 42', 12345)['eligible'])

    def test_foreign_or_unknown_query_rejected(self):
        for output in ('Process id : 12345\nProcess id : 56789', '', 'query unavailable'):
            self.assertFalse(assess('Health Status : OK', output, 12345)['eligible'])

    def test_idle_requires_known_health_and_identity(self):
        self.assertTrue(assess('Health Status : OK', 'No process in device.', 12345)['eligible'])
        self.assertFalse(assess('Health Status : Warning', 'No process in device.', 12345)['eligible'])
        self.assertFalse(assess('Health Status : OK', 'No process in device.', 0)['eligible'])


if __name__ == '__main__':
    unittest.main()
