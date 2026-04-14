import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


RMSE_REGEX = re.compile(r"['\"]rmse['\"]:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_config = repo_root / "configs" / "Dynamic" / "Bonn" / "bonn_balloon.yaml"
    default_dataset_root = repo_root / "datasets" / "Bonn"
    default_sequence_dir = default_dataset_root / "rgbd_bonn_balloon"
    default_weights = repo_root / "pretrained" / "droid.pth"
    default_output_root = repo_root / "Outputs" / "Bonn"

    parser = argparse.ArgumentParser(
        description="Run DROID-W on rgbd_bonn_balloon and summarize the resulting trajectory metrics."
    )
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used to launch run.py.")
    parser.add_argument("--config", default=str(default_config), help="Base config file.")
    parser.add_argument("--dataset-root", default=str(default_dataset_root), help="Bonn dataset root folder.")
    parser.add_argument(
        "--sequence-dir",
        default=str(default_sequence_dir),
        help="Sequence directory to validate. Defaults to datasets/Bonn/rgbd_bonn_balloon.",
    )
    parser.add_argument("--weights", default=str(default_weights), help="Path to pretrained/droid.pth.")
    parser.add_argument("--output-root", default=str(default_output_root), help="Output root folder.")
    parser.add_argument("--scene", default="bonn_balloon", help="Scene name used by the config/output folder.")
    parser.add_argument("--device", default=None, help="Optional device override, e.g. cuda:0.")
    parser.add_argument("--fast-mode", dest="fast_mode", action="store_true", help="Force fast_mode=True.")
    parser.add_argument("--no-fast-mode", dest="fast_mode", action="store_false", help="Force fast_mode=False.")
    parser.set_defaults(fast_mode=None)
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Do not launch run.py. Only inspect the expected output files.",
    )
    return parser.parse_args()


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def extract_rmse(metrics_path: Path) -> float | None:
    if not metrics_path.exists():
        return None

    text = metrics_path.read_text()
    match = RMSE_REGEX.search(text)
    if match is None:
        return None
    return float(match.group(1))


def build_override_config(args: argparse.Namespace, repo_root: Path) -> Path:
    override = {
        "inherit_from": str(Path(args.config).resolve()),
        "scene": args.scene,
        "data": {
            "root_folder": str(Path(args.dataset_root).resolve()),
            "input_folder": str(Path(args.sequence_dir).resolve()),
            "output": str(Path(args.output_root).resolve()),
        },
        "tracking": {
            "pretrained": str(Path(args.weights).resolve()),
        },
    }

    if args.device is not None:
        override["device"] = args.device

    if args.fast_mode is not None:
        override["fast_mode"] = args.fast_mode

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="validate_bonn_balloon_",
        delete=False,
        dir=repo_root,
    )
    with tmp:
        yaml.safe_dump(override, tmp, sort_keys=False)

    return Path(tmp.name)


def print_summary(output_dir: Path) -> int:
    traj_dir = output_dir / "traj"
    full_metrics = traj_dir / "metrics_full_traj.txt"
    kf_metrics = traj_dir / "metrics_kf_traj.txt"
    full_traj = traj_dir / "est_poses_full.txt"
    gt_traj = output_dir / "gt_poses.txt"
    video_npz = output_dir / "video.npz"

    print(f"Output directory: {output_dir}")
    print(f"Trajectory directory: {traj_dir}")

    if video_npz.exists():
        print(f"Video snapshot: {video_npz}")
    if gt_traj.exists():
        print(f"Ground-truth poses: {gt_traj}")
    if full_traj.exists():
        print(f"Estimated full trajectory: {full_traj}")

    full_rmse = extract_rmse(full_metrics)
    kf_rmse = extract_rmse(kf_metrics)

    if kf_metrics.exists():
        print(f"Keyframe metrics: {kf_metrics}")
        if kf_rmse is not None:
            print(f"Keyframe ATE RMSE: {kf_rmse:.6f} m")

    if full_metrics.exists():
        print(f"Full-trajectory metrics: {full_metrics}")
        if full_rmse is not None:
            print(f"Full-trajectory ATE RMSE: {full_rmse:.6f} m")
        return 0

    print("Full-trajectory metrics were not found. The run may have failed or may still be in progress.")
    return 1


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    config_path = Path(args.config).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    sequence_dir = Path(args.sequence_dir).resolve()
    weights_path = Path(args.weights).resolve()
    output_dir = Path(args.output_root).resolve() / args.scene

    require_path(config_path, "Config file")
    require_path(dataset_root, "Dataset root")
    require_path(sequence_dir, "Sequence directory")
    require_path(weights_path, "Pretrained checkpoint")

    override_config = build_override_config(args, repo_root)

    try:
        print(f"Using python: {args.python}")
        print(f"Using config: {config_path}")
        print(f"Using sequence: {sequence_dir}")
        print(f"Using weights: {weights_path}")
        print(f"Using output directory: {output_dir}")

        if not args.skip_run:
            cmd = [args.python, "run.py", "--config", str(override_config)]
            print("Launching:")
            print(" ".join(cmd))
            subprocess.run(cmd, cwd=repo_root, check=True)

        return print_summary(output_dir)
    finally:
        try:
            override_config.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
