"""Read-only device monitoring; never signal processes or reset hardware."""
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading


def identity():
    driver = Path('/usr/local/Ascend/driver/lib64/driver/libascend_hal.so')
    library = ctypes.CDLL(str(driver))
    get_tgid = library.drvDeviceGetBareTgid
    get_tgid.argtypes = []
    get_tgid.restype = ctypes.c_int
    host_pid = get_tgid()
    if host_pid <= 0 or get_tgid() != host_pid:
        raise RuntimeError('Driver host PID identity unavailable or unstable')
    with driver.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return dict(container_pid=os.getpid(), driver_host_pid=host_pid,
                pid_namespace=os.readlink('/proc/self/ns/pid'), driver_sha256=digest)


def assess(health, processes, host_pid):
    errors = []
    if not re.search(r'Health Status\s*:\s*OK\b', health):
        errors.append('Unhealthy or unknown health format')
    pids = set(re.findall(r'Process id\s*:\s*(\d+)', processes, re.I))
    if not pids and 'No process in device.' not in processes:
        errors.append('Unknown process-query format')
    # npu-smi reports HOST PIDs. Container PID numerical coincidence is not ownership.
    if host_pid <= 0:
        errors.append('Host identity unavailable')
    foreign = pids - {str(host_pid)}
    if foreign:
        errors.append('Foreign NPU process: ' + ','.join(sorted(foreign)))
    return dict(eligible=not errors, errors=errors, observed_host_pids=sorted(pids))


class Monitor:
    def __init__(self, card, owner, output):
        self.card, self.owner, self.output = card, owner, output
        self.samples = []
        self.mutex = threading.Lock()
        self.stop = threading.Event()
        self.thread = None

    def sample(self):
        with self.mutex:
            queries = {}
            for kind in ('health', 'proc-mem'):
                try:
                    p = subprocess.run(['npu-smi', 'info', '-t', kind, '-i', str(self.card)],
                                       capture_output=True, text=True, timeout=10)
                    queries[kind] = dict(exit=p.returncode, output=p.stdout, stderr=p.stderr)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    queries[kind] = dict(exit=-1, output='', stderr=repr(exc))
            decision = assess(queries['health']['output'], queries['proc-mem']['output'],
                              self.owner['driver_host_pid'])
            for kind, query in queries.items():
                if query['exit'] != 0:
                    decision['errors'].append('Query failed: ' + kind)
            decision['eligible'] = not decision['errors']
            sample = dict(utc=datetime.now(timezone.utc).isoformat(), physical_card=self.card,
                          queries=queries, **decision)
            self.samples.append(sample)
            with self.output.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(sample) + '\n')
            return sample

    def require_clear(self):
        sample = self.sample()
        if not sample['eligible']:
            raise RuntimeError('; '.join(sample['errors']))

    def eligible_since(self, index):
        with self.mutex:
            return bool(self.samples[index:]) and all(s['eligible'] for s in self.samples[index:])

    def __enter__(self):
        self.require_clear()
        def loop():
            while not self.stop.wait(5):
                self.sample()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=25)
        if self.thread.is_alive():
            raise RuntimeError('Device monitor failed to stop')
        self.sample()
