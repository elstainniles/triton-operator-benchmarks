"""Profile one captured Triton launch and accept only attributed device time."""

import csv
import ctypes
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time


DRIVER = Path("/usr/local/Ascend/driver/lib64/driver/libascend_hal.so")


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _host_pid():
    driver = ctypes.CDLL(str(DRIVER))
    function = driver.drvDeviceGetBareTgid
    function.argtypes = []
    function.restype = ctypes.c_int
    first = function()
    if first <= 0 or function() != first:
        raise RuntimeError("Unstable host PID from NPU driver")
    return first


def _telemetry(host_pid):
    card = os.environ["ASCEND_RT_VISIBLE_DEVICES"]
    data = {"time": time.time()}
    for kind in ("health", "proc-mem"):
        try:
            result = subprocess.run(["npu-smi", "info", "-t", kind, "-i", card],
                                    capture_output=True, text=True, timeout=20)
            data[kind] = {"exit": result.returncode, "output": result.stdout,
                          "error": result.stderr}
        except subprocess.TimeoutExpired:
            data[kind] = {"exit": 124, "output": "", "error": "npu-smi timeout"}
    process_text = data["proc-mem"]["output"]
    pids = set(re.findall(r"Process id\s*:\s*(\d+)", process_text, re.I))
    errors = []
    if data["health"]["exit"] != 0 or not re.search(
            r"Health Status\s*:\s*OK", data["health"]["output"]):
        errors.append("DEVICE_HEALTH_UNKNOWN")
    if data["proc-mem"]["exit"] != 0 or (not pids and "No process in device." not in process_text):
        errors.append("DEVICE_OCCUPANCY_UNKNOWN")
    if pids - {str(host_pid)}:
        errors.append("FOREIGN_NPU_PROCESS")
    data.update(pids=sorted(pids), errors=errors)
    return data


def _compiled_evidence(run_dir, name, source):
    evidence = []
    for metadata in sorted((run_dir / "cache").rglob(name + ".json")):
        info = json.loads(metadata.read_text())
        ir = metadata.with_suffix(".source")
        binary = metadata.with_suffix(".npubin")
        if info.get("name") != name or info.get("target", {}).get("backend") != "npu":
            continue
        if not ir.is_file() or not binary.is_file():
            continue
        origin = re.search(r'^#loc = loc\("([^"]+)":\d+:\d+\)',
                           ir.read_text()[:4096], re.M)
        if origin is None or Path(origin.group(1)).resolve() != Path(source).resolve():
            continue
        evidence.append({"metadata": str(metadata), "metadata_sha256": _sha(metadata),
                         "ir": str(ir), "ir_sha256": _sha(ir),
                         "binary": str(binary), "binary_sha256": _sha(binary),
                         "compiler_hash": info.get("hash")})
    if not evidence:
        raise ValueError("No compiled NPU artifact with exact function and source")
    return evidence


def _trace_samples(profile_dir, name):
    trace_files = list(profile_dir.rglob("trace_view.json"))
    if len(trace_files) != 1:
        raise ValueError(f"Expected one profiler trace, found {len(trace_files)}")
    trace_file = trace_files[0]
    csv_file = trace_file.with_name("kernel_details.csv")
    trace = json.loads(trace_file.read_text(), parse_float=Decimal)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    labels = [entry for entry in events if entry.get("ph") == "X" and
              entry.get("name", "").startswith("ligerbase::")]
    expected = {f"ligerbase::{name}::{index}" for index in range(15)}
    if {entry["name"] for entry in labels} != expected or len(labels) != 15:
        raise ValueError("Incomplete or repeated sample labels")
    with csv_file.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_time = {Decimal(row["Start Time(us)"].strip()): row for row in rows}
    if len(by_time) != len(rows):
        raise ValueError("Ambiguous device timestamps")
    starts = {entry["id"]: entry for entry in events
              if entry.get("cat") == "async_npu" and entry.get("ph") == "s"}
    ends = {entry["id"]: entry for entry in events
            if entry.get("cat") == "async_npu" and entry.get("ph") == "f"}
    calls = {label: [] for label in expected}
    for flow, start in starts.items():
        timestamp = Decimal(start["ts"])
        matched = [entry for entry in labels if entry["pid"] == start["pid"] and
                   Decimal(entry["ts"]) <= timestamp < Decimal(entry["ts"]) + Decimal(entry["dur"])]
        if len(matched) != 1:
            continue
        if flow not in ends or Decimal(ends[flow]["ts"]) not in by_time:
            raise ValueError("Unresolved CPU-to-NPU flow in target sample")
        calls[matched[0]["name"]].append(by_time[Decimal(ends[flow]["ts"])])
    samples = []
    for index in range(15):
        label = f"ligerbase::{name}::{index}"
        observed = calls[label]
        if len(observed) != 1 or observed[0]["Name"] != name:
            raise ValueError(f"Expected exactly one {name} device kernel for {label}: "
                             f"{[row['Name'] for row in observed]}")
        samples.append(float(Decimal(observed[0]["Duration(us)"])))
    return {"us": statistics.median(samples), "samples_us": samples,
            "trace": str(trace_file), "trace_sha256": _sha(trace_file),
            "kernel_csv": str(csv_file), "kernel_csv_sha256": _sha(csv_file)}


def measure(launch, name, source, run_dir, method="profiler"):
    import torch
    if method not in ("profiler", "mspti"):
        raise ValueError("Unknown timing method: " + method)
    if method == "profiler":
        import torch_npu

    run_dir = Path(run_dir)
    profile_dir = run_dir / "profiles" / name
    host_pid = _host_pid()
    telemetry = [_telemetry(host_pid)]
    stop = threading.Event()

    def monitor():
        while not stop.wait(0.25):
            telemetry.append(_telemetry(host_pid))

    thread = threading.Thread(target=monitor, daemon=True)
    for _ in range(3):
        launch()
    torch.npu.synchronize()
    telemetry.append(_telemetry(host_pid))
    if any(item["errors"] for item in telemetry):
        return {"status": "INTERFERED_TIMING_REJECTED", "telemetry": telemetry}
    thread.start()
    try:
        if method == "mspti":
            from .mspti import measure_npu
            mspti_profile = measure_npu([("baseline", launch, name)])
        else:
            config = torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone)
            with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU,
                                torch_npu.profiler.ProfilerActivity.NPU],
                    schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=3, repeat=1),
                    record_shapes=False, profile_memory=False, with_stack=False,
                    experimental_config=config,
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir))) as profiler:
                for step in range(4):
                    if step == 1:
                        time.sleep(0.02)
                    for repeat in range(5):
                        if step == 0:
                            launch()
                        else:
                            with torch.profiler.record_function(
                                    f"ligerbase::{name}::{(step - 1) * 5 + repeat}"):
                                launch()
                    torch.npu.synchronize()
                    profiler.step()
    finally:
        stop.set()
        thread.join(timeout=50)
        telemetry.append(_telemetry(host_pid))
    telemetry_file = run_dir / "profiles" / (name + "-telemetry.json")
    telemetry_file.parent.mkdir(parents=True, exist_ok=True)
    telemetry_file.write_text(json.dumps(telemetry, indent=2) + "\n")
    if thread.is_alive() or any(item["errors"] for item in telemetry):
        return {"status": "INTERFERED_TIMING_REJECTED", "telemetry_file": str(telemetry_file)}
    evidence = _compiled_evidence(run_dir, name, source)
    if method == "mspti":
        variant = mspti_profile["variants"]["baseline"]
        samples = {"us": variant["median_us"],
                   "samples_us": [sample["duration_us"] for sample in variant["samples"]],
                   "rounds": variant["rounds"], "profile": mspti_profile}
    else:
        samples = _trace_samples(profile_dir, name)
    return {"status": "MEASURED_KERNEL", "metric": "TARGET_TRITON_KERNEL_DEVICE_US",
            "method": method,
            "sample_count": 15, "host_pid": host_pid, "telemetry_file": str(telemetry_file),
            "compiled_evidence": evidence, **samples}
