import argparse
import importlib
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
        "use_cfg",
        "persistent_workers",
        "pin_memory",
        "allow_tf32",
        "cudnn_benchmark",
        "ffl_ave_spectrum",
        "ffl_log_matrix",
        "ffl_batch_matrix",
        "cast_teacher_unet",
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


def normalize_train_script(entry, repo_root):
    entry = str(entry).strip()
    if not entry:
        return ""
    if entry.endswith(".py") or "/" in entry or "\\" in entry:
        path = Path(entry)
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        try:
            rel = path.relative_to(repo_root)
        except ValueError:
            rel = path
        return ".".join(rel.with_suffix("").parts)
    return entry


def main():
    parser = argparse.ArgumentParser(description="Launch LCM training from YAML config (v3)")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--print_args", action="store_true", help="Only print parsed arguments.")
    args = parser.parse_args()

    cfg = merge_config(load_yaml(args.config))
    train_script = str(cfg.pop("train_script", "")).strip()
    cli_args = build_args(cfg)
    if args.print_args:
        print(" ".join(cli_args))
        return

    sys.argv = [sys.argv[0]] + cli_args
    if train_script:
        module_name = normalize_train_script(train_script, repo_root)
        train_main = importlib.import_module(module_name).main
    else:
        from student_model_v3.train_lcm_anytext_v3 import main as train_main

    train_main()


if __name__ == "__main__":
    main()
