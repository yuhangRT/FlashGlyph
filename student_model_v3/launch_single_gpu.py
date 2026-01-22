#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


def _validate_gpu_id(gpu_id):
    parts = [p for p in str(gpu_id).split(",") if p.strip()]
    if len(parts) != 1:
        raise ValueError(f"--gpu must be a single id, got '{gpu_id}'")
    return parts[0].strip()


def _build_train_command(config_path, print_args, prepare_cache):
    repo_root = Path(__file__).resolve().parent.parent
    launcher = repo_root / "student_model_v2" / "launch_from_yaml.py"
    if not launcher.exists():
        raise FileNotFoundError(f"Missing launcher: {launcher}")
    cmd = [sys.executable, str(launcher), "--config", str(config_path)]
    if print_args:
        cmd.append("--print_args")
    if prepare_cache:
        cmd.append("--prepare_cache")
    return cmd


def _build_guard_command(min_available_gb, check_interval, kill_timeout):
    guard = Path(__file__).resolve().parent / "oom_guard.py"
    if not guard.exists():
        raise FileNotFoundError(f"Missing oom_guard: {guard}")
    return [
        sys.executable,
        str(guard),
        "--min-available-gb",
        str(min_available_gb),
        "--check-interval",
        str(check_interval),
        "--kill-timeout",
        str(kill_timeout),
        "--",
    ]


def main():
    parser = argparse.ArgumentParser(description="Single-GPU launcher with optional RAM OOM guard.")
    parser.add_argument("--config", required=True, help="Path to v3 YAML config.")
    parser.add_argument("--gpu", default="0", help="Single GPU id (default: 0).")
    parser.add_argument("--min-available-gb", type=float, default=6.0)
    parser.add_argument("--check-interval", type=float, default=5.0)
    parser.add_argument("--kill-timeout", type=float, default=30.0)
    parser.add_argument("--print_args", action="store_true")
    parser.add_argument("--prepare_cache", action="store_true")
    args = parser.parse_args()

    gpu_id = _validate_gpu_id(args.gpu)
    cmd = _build_train_command(args.config, args.print_args, args.prepare_cache)

    if args.min_available_gb > 0:
        cmd = _build_guard_command(args.min_available_gb, args.check_interval, args.kill_timeout) + cmd

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    ret = subprocess.call(cmd, env=env)
    raise SystemExit(ret)


if __name__ == "__main__":
    main()
