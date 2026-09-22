"""Watch the GPU directly through NVML, so "is it actually working?" is measured, not guessed.

    python scripts/gpu_watch.py --seconds 30 [--interval 1] [--pid 12345]

Prints one line per sample: GPU utilisation, memory used, SM clock, power and temperature, plus whether
the named PID (or any python process) holds device memory. Exits non-zero if the GPU stayed idle, which
makes it usable as a check: run it *outside* the job and it will not report the job's own CPU time.
"""
import argparse, os, sys, time


def find_pid(handle, pid):
    """(bytes, name) held by pid, or None. NVML hides other users' processes without root."""
    for p in _procs(handle):
        if p.pid == pid:
            return p.usedGpuMemory, getattr(p, "name", "?")
    return None


def _procs(handle):
    import pynvml
    try:
        return pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--pid", type=int, default=0, help="also track this PID's device memory (0 = any process)")
    a = ap.parse_args()

    try:
        import pynvml
    except ImportError:
        raise SystemExit("pip install nvidia-ml-py (pixi add --pypi nvidia-ml-py)")
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(a.index)
    name = pynvml.nvmlDeviceGetName(handle)
    name = name.decode() if isinstance(name, bytes) else name
    total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
    print(f"[gpu] {name}, {total / 1e9:.1f} GB, sampling every {a.interval}s for {a.seconds}s", flush=True)

    utils, mems, mine = [], [], []
    t_end = time.time() + a.seconds
    while time.time() < t_end:
        try:
            u = pynvml.nvmlDeviceGetUtilizationRates(handle)
            m = pynvml.nvmlDeviceGetMemoryInfo(handle)
            p = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            c = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        except Exception as e:                                  # lost context, driver reset
            print(f"[gpu] NVML error: {type(e).__name__}: {e}", flush=True)
            break
        proc = find_pid(handle, a.pid) if a.pid else None
        tag = ""
        if proc:
            tag = f" pid{a.pid}={proc[0] / 1e9:.2f}GB"
            mine.append(proc[0])
        utils.append(u.gpu); mems.append(m.used)
        print(f"[gpu] util {u.gpu:3d}%  mem {m.used / 1e9:5.2f}/{total / 1e9:.1f} GB  sm {c:4d} MHz  "
              f"pwr {p:5.1f} W  temp {pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU):2d} C"
              f"{tag}", flush=True)
        time.sleep(a.interval)

    if not utils:
        raise SystemExit("[gpu] no samples")
    print(f"[gpu] util mean {sum(utils) / len(utils):.1f}%  max {max(utils)}%  |  "
          f"peak mem {max(mems) / 1e9:.2f} GB  |  peaks: {sum(1 for x in utils if x > 50)}/{len(utils)} samples over 50%")
    if a.pid and not mine:
        print(f"[gpu] pid {a.pid} is not on the device (or NVML cannot see its memory)")
    if max(utils) < 5:
        print("[gpu] the GPU stayed idle during the window")
        sys.exit(2)


if __name__ == "__main__":
    main()
