#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import sys
import time


def read_mem_available_kb():
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    return None


def terminate_process_group(proc, timeout):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        return


def main():
    parser = argparse.ArgumentParser(description="Run a command and stop it when RAM is low.")
    parser.add_argument("--min-available-gb", type=float, default=6.0)
    parser.add_argument("--check-interval", type=float, default=5.0)
    parser.add_argument("--kill-timeout", type=float, default=30.0)
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Command after --")
    args = parser.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("Command is required. Use: oom_guard.py -- <command>")

    min_available_kb = int(args.min_available_gb * 1024 * 1024)
    proc = subprocess.Popen(cmd, preexec_fn=os.setsid)

    try:
        while True:
            ret = proc.poll()
            if ret is not None:
                sys.exit(ret)
            avail_kb = read_mem_available_kb()
            if avail_kb is not None and avail_kb < min_available_kb:
                print(
                    f"[oom_guard] MemAvailable {avail_kb / 1024 / 1024:.2f} GB < "
                    f"{args.min_available_gb:.2f} GB, stopping process group."
                )
                terminate_process_group(proc, args.kill_timeout)
                sys.exit(1)
            time.sleep(args.check_interval)
    except KeyboardInterrupt:
        terminate_process_group(proc, args.kill_timeout)
        raise


if __name__ == "__main__":
    main()
