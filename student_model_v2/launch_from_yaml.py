import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))


def load_yaml(path):
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping.")
    return data


def merge_config(cfg):
    merged = {}
    for section in ("model", "data", "train"):
        section_cfg = cfg.get(section, {})
        if section_cfg:
            merged.update(section_cfg)
    for key, value in cfg.items():
        if key in ("model", "data", "train", "system"):
            continue
        merged.setdefault(key, value)
    return merged


def build_args(config):
    args = []
    list_keys = {"dataset_json"}
    bool_keys = {
        "use_mock_dataset",
        "persistent_workers",
        "pin_memory",
        "allow_tf32",
        "cudnn_benchmark",
    }

    for key, value in config.items():
        if value is None or value == "":
            continue
        flag = f"--{key}"
        if key == "streaming":
            args.append("--streaming" if value else "--no_streaming")
            continue
        if key in bool_keys:
            if value:
                args.append(flag)
            continue
        if key in list_keys:
            if isinstance(value, (list, tuple)):
                args.append(flag)
                args.extend([str(v) for v in value])
            else:
                args.extend([flag, str(value)])
            continue
        args.extend([flag, str(value)])

    return args


def expand_paths(paths, repo_root):
    expanded = []
    for entry in paths:
        for part in str(entry).split(","):
            part = part.strip()
            if not part:
                continue
            p = Path(part)
            if not p.is_absolute():
                p = (repo_root / p).resolve()
            if p.suffix in {".list", ".txt"}:
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        path_line = Path(line)
                        if not path_line.is_absolute():
                            path_line = (repo_root / path_line).resolve()
                        expanded.append(str(path_line))
            else:
                expanded.append(str(p))
    return expanded


def prepare_cache(config):
    from student_model_v2.dataset_anytext_v2 import JsonlIndex

    dataset_json = config.get("dataset_json", [])
    if not dataset_json:
        return
    json_paths = expand_paths(dataset_json, repo_root)
    wm_thresh = float(config.get("wm_thresh", 1.0))
    streaming_threshold_mb = int(config.get("streaming_threshold_mb", 200))
    cache_dir = config.get("cache_dir") or None
    for path in json_paths:
        JsonlIndex(
            json_path=path,
            wm_thresh=wm_thresh,
            force_streaming=True,
            threshold_mb=streaming_threshold_mb,
            cache_dir=cache_dir,
        )


def main():
    parser = argparse.ArgumentParser(description="Launch LCM training from YAML config")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--print_args", action="store_true", help="Only print parsed arguments.")
    parser.add_argument("--prepare_cache", action="store_true", help="Only build streaming cache files.")
    args = parser.parse_args()

    cfg = merge_config(load_yaml(args.config))
    if args.prepare_cache:
        prepare_cache(cfg)
        return
    use_optimized = bool(cfg.pop("use_optimized", False))
    cli_args = build_args(cfg)
    if args.print_args:
        print(" ".join(cli_args))
        return

    sys.argv = [sys.argv[0]] + cli_args
    if use_optimized:
        from student_model_v2.train_lcm_anytext_v2_2 import main as train_main
    else:
        from student_model_v2.train_lcm_anytext_v2 import main as train_main

    train_main()


if __name__ == "__main__":
    main()
