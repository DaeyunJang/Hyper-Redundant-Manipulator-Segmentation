"""Archive a small, non-destructive reproducibility snapshot for a pilot run.

Copies code/config/provenance only. Images, checkpoints and virtual environments
are NEVER copied. Every invocation creates a fresh timestamped generation,
including SHA256 checks, package versions and a read-only Git-status snapshot.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
COPY_SUFFIXES = {".py", ".md", ".json", ".txt", ".csv", ".html"}
FORBIDDEN_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".mp4",
                      ".pt", ".pth", ".onnx", ".npz", ".npy", ".db3"}
PAPER_SNAPSHOT_NAMES = (
    "paper_pose_summary.csv", "paper_pose_summary.json", "paper_pose_summary.html",
    "paper_pose_validation.json", "paper_pose_source_index.json", "figure_sources.json",
)
HASH_ONLY_BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npz", ".npy"}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _command(root, *args):
    try:
        result = subprocess.run(args, cwd=root, text=True, capture_output=True,
                                timeout=15, check=False)
        return dict(command=list(args), returncode=result.returncode,
                    stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(command=list(args), error=str(error))


def _json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def archive(root, run):
    root, run = Path(root).resolve(), Path(run).resolve()
    if not root.is_dir() or not run.is_dir():
        raise FileNotFoundError("Both project and run directories must already exist")
    if not run.is_relative_to(root):
        raise ValueError("Run directory must be inside the segmentation project")
    created = datetime.now(ZoneInfo("Asia/Seoul"))
    destination = run / "provenance" / created.strftime("snapshot_%Y%m%dT%H%M%S_%f_KST")
    destination.mkdir(parents=True, exist_ok=False)

    # Deliberately shallow allowlists: no recursive dataset/image/weight copies.
    sources = {p for pattern in ("*.py", "*.md", "requirements*.txt", "LICENSE*", ".gitignore")
               for p in root.glob(pattern) if p.is_file()}
    sources.add(root / "datasets" / "split.json")
    background_root = root / "datasets" / "backgrounds"
    sources.update(p for p in background_root.glob("*") if p.is_file() and
                   (p.suffix == ".json" or p.name.startswith("LICENSE")))
    sources.update(p for p in (background_root / "manifests").glob("*.json") if p.is_file())
    model_folders = sorted(p.name for p in run.glob("unet_*")
                           if p.is_dir() and (p / "config.json").is_file())
    for model_folder in model_folders:
        sources.update(run / model_folder / name for name in
                       ("config.json", "split.json", "history.csv", "completed.json")
                       if (run / model_folder / name).is_file())
    # Preserve final protocol/audits, including the explicitly exploratory min1
    # short-fragment result. Large per-frame/case arrays stay in place and are
    # indexed by SHA256 below instead of creating another multi-MB data copy.
    sources.update(p for pattern in ("*audit*.json", "*preflight*.json", "report.html", "*.md")
                   for p in run.glob(pattern) if p.is_file())
    sources.update(run / name for name in PAPER_SNAPSHOT_NAMES if (run / name).is_file())
    result_dirs = ("evaluation", "segmentation_evaluation", "pose_evaluation",
                   "pose_evaluation_short_fragments")
    for folder in result_dirs:
        sources.update(run / folder / name for name in
                       ("summary.csv", "summary.json", "protocol.json", "results.json", "independent_audit.json")
                       if (run / folder / name).is_file())
        sources.update(p for pattern in ("*audit*.json", "*preflight*.json")
                       for p in (run / folder).glob(pattern) if p.is_file())
    for folder in ("live_capture", "live_capture_black"):
        sources.update(run / folder / name for name in ("manifest.json", "capture_notes.json")
                       if (run / folder / name).is_file())

    files = []
    for source in sorted(sources):
        if not source.is_file():
            continue
        relative = source.relative_to(root)
        if source.suffix.lower() in FORBIDDEN_SUFFIXES or ".venv" in relative.parts:
            raise ValueError(f"Refusing binary/data copy: {source}")
        if source.suffix.lower() not in COPY_SUFFIXES and not source.name.startswith("LICENSE") and source.name != ".gitignore":
            raise ValueError(f"Unexpected source type: {source}")
        payload = source.read_bytes()
        target = destination / "files" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(payload)
        digest = hashlib.sha256(payload).hexdigest()
        if sha256(target) != digest:
            raise RuntimeError(f"Archive verification failure: {target}")
        files.append(dict(source=str(source), relative_path=str(relative),
                          archive_path=str(target.relative_to(destination)),
                          bytes=len(payload), sha256=digest,
                          source_unchanged_during_copy=sha256(source) == digest))

    package_names = ("torch", "torchvision", "numpy", "opencv-python",
                     "opencv-python-headless", "scipy", "scikit-image",
                     "segmentation-models-pytorch", "timm", "ultralytics",
                     "Pillow", "matplotlib")
    versions = {}
    for name in package_names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    _json(destination / "environment.json",
          dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
               package_versions=versions, timezone="Asia/Seoul",
               nvidia_smi=_command(root, "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                   "--format=csv,noheader"),
               note="Package metadata only; no GPU model inference or CUDA initialization performed"))
    _json(destination / "git_status.json",
          dict(head=_command(root, "git", "rev-parse", "HEAD"),
               status=_command(root, "git", "status", "--short"),
               diff_stat=_command(root, "git", "diff", "--stat")))

    # Hash backgrounds without copying image payloads; record split/order exactly.
    backgrounds = []
    for split in ("train", "val", "test"):
        for path in sorted((background_root / split).glob("*")):
            if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                backgrounds.append(dict(split=split, path=str(path.relative_to(root)),
                                        bytes=path.stat().st_size, sha256=sha256(path)))
    _json(destination / "background_hashes.json", backgrounds)
    checkpoint_comparisons = []
    for model_folder in model_folders:
        config_path = run / model_folder / "config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text())
        comparisons = []
        for name, training_hash in config.get("source_hashes", {}).items():
            current = root / name
            current_hash = sha256(current) if current.is_file() else None
            comparisons.append(dict(path=name, training_start_sha256=training_hash,
                                    current_sha256=current_hash, matches=current_hash == training_hash))
        checkpoint = run / model_folder / "best.pt"
        checkpoint_comparisons.append(dict(model=model_folder, config=str(config_path),
                                            source_hash_comparisons=comparisons,
                                            checkpoint_path=str(checkpoint),
                                            checkpoint_sha256=sha256(checkpoint) if checkpoint.is_file() else None))
    _json(destination / "training_hash_comparison.json", checkpoint_comparisons)

    # Hash existing metadata/results without copying images, NPZ arrays, bags,
    # checkpoints, or previous provenance generations. Paths remain resolvable.
    result_hashes = []
    binary_hashes = []
    for path in sorted(run.rglob("*")):
        if not path.is_file() or "provenance" in path.relative_to(run).parts:
            continue
        if path.suffix.lower() in HASH_ONLY_BINARY_SUFFIXES:
            binary_hashes.append(dict(path=str(path.relative_to(root)),
                                      bytes=path.stat().st_size, sha256=sha256(path),
                                      copied=False))
            continue
        if path.suffix.lower() not in (".json", ".csv", ".md", ".html"):
            continue
        result_hashes.append(dict(path=str(path.relative_to(root)), bytes=path.stat().st_size,
                                  sha256=sha256(path), copied=path in sources))
    _json(destination / "result_metadata_hashes.json", result_hashes)
    _json(destination / "result_binary_hashes.json", binary_hashes)

    # No attempt to hide differences between launch-time hashes and final code.
    caveats = [
        "RGB/mask labels are OpenCV pseudo-labels, not independently annotated manual ground truth.",
        "45deg_1 is disjoint from the current training manifest only; historical development exposure is unknown.",
        "The runner received an evaluation-boundary-F1 zero-denominator/border fix during this pilot; training logic was not changed by that fix.",
        "The v2 runner received an evaluation prediction-validity guard during training; this rejects NaN/nonbinary outputs without changing training logic.",
        "Training-start config hashes and current hashes are both retained; current code is not asserted byte-identical to launch-time code.",
        "Any additional source hash mismatch needs inspection; it is not automatically attributed to the known metric fix.",
        "This snapshot contains metadata/code only; checkpoints and source image datasets remain at their original recorded paths.",
        "Paper summaries/source indexes and shallow preflight/audit JSONs are copied; paper_pose_frames.csv, large cases.json and run-local PNG/NPZ arrays are referenced and hashed only.",
        "Public Endoscapes backgrounds retain their CC BY-NC-SA-4.0 terms, separate from project code licensing.",
        "Synthetic RGB occlusion results alone do not validate actual physical occlusion or full 3D reconstruction.",
        "The min1 short-fragment pose result is an exploratory frontend revision after inspecting min8 failures; original results and both protocols/audits are preserved when available.",
    ]
    manifest = dict(schema_version=1, generated_at=created.isoformat(),
                    project_root=str(root), run=str(run), archive=str(destination),
                    files=files, copied_bytes=sum(f["bytes"] for f in files),
                    background_count=len(backgrounds), binary_reference_count=len(binary_hashes),
                    caveats=caveats)
    _json(destination / "manifest.json", manifest)
    derived_names = ("manifest.json", "environment.json", "git_status.json",
                     "background_hashes.json", "training_hash_comparison.json", "result_metadata_hashes.json",
                     "result_binary_hashes.json")
    _json(destination / "archive_integrity.json", {
        "generated_files": [{"path": name, "sha256": sha256(destination / name),
                             "bytes": (destination / name).stat().st_size} for name in derived_names],
        "copied_file_sha256_in": "manifest.json",
    })
    return dict(archive=str(destination), copied_files=len(files),
                copied_bytes=manifest["copied_bytes"], background_count=len(backgrounds),
                binary_reference_count=len(binary_hashes),
                source_files_changed_during_copy=[f["source"] for f in files
                                                   if not f["source_unchanged_during_copy"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run", type=Path, default=ROOT / "runs" / "feasibility_20261024")
    args = parser.parse_args()
    print(json.dumps(archive(args.root, args.run), indent=2))


if __name__ == "__main__":
    main()
