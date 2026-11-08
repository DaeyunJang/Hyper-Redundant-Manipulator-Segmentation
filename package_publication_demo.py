# AUTHOR: Daeyun - compact, provenance-preserving HRM publication asset packaging.
"""Package existing local assets; never download, retrain, re-encode or delete.

Building requires the full original local datasets/runs and the sibling HRM
record archive. A fresh clone needs only --verify-only, not another build.
All runnable manifest paths are relative to their manifest's directory.
"""

from __future__ import annotations

import argparse
import collections
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
V2 = Path("runs/occlusion_pose_20261024/unet_v2")
PAPER = Path("runs/perspective_pose_followup_20261024")
PROTOCOL = Path("runs/occlusion_pose_20261024/pose_evaluation_short_fragments/protocol.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n")


def provenance(path):
    """Descriptive source location, never a path required by the public demo."""
    path = Path(path).resolve()
    for root in (ROOT, ROOT.parent / "Hyper-Redundant-Manipulator"):
        if path.is_relative_to(root):
            return {"repository": root.name, "path": path.relative_to(root).as_posix()}
    raise ValueError(f"Source outside the two explicitly scoped repositories: {path}")


def verify(destination):
    entries = (destination / "checksums.sha256").read_text().splitlines()
    for entry in entries:
        expected, relative = entry.split("  ", 1)
        path = (destination / relative).resolve()
        if not path.is_relative_to(destination.resolve()) or not path.is_file():
            raise ValueError(f"Invalid/missing bundled path: {relative}")
        if sha256(path) != expected:
            raise ValueError(f"Checksum mismatch: {relative}")
    samples = read_json(destination / "samples/index.json")["records"]
    rgbd = read_json(destination / "rgbd/index.json")["records"]
    if len(samples) != 50 or collections.Counter(x["dataset"] for x in rgbd) != {
            "live_black": 40, "archive_45deg_1": 64}:
        raise ValueError("Unexpected public sample counts")
    for sample in samples:
        rgb = cv2.imread(str(destination / "samples" / sample["rgb"]))
        mask = cv2.imread(str(destination / "samples" / sample["mask"]), cv2.IMREAD_UNCHANGED)
        if rgb is None or mask is None or rgb.shape[:2] != mask.shape:
            raise ValueError("Invalid example RGB/mask pair")
        if mask.dtype != np.uint8 or not set(np.unique(mask)).issubset({0, 255}):
            raise ValueError("The original pseudo-masks must be uint8 binary PNGs")
    depth_values = []
    for record in rgbd:
        rgb = cv2.imread(str(destination / "rgbd" / record["rgb"]))
        depth = cv2.imread(str(destination / "rgbd" / record["depth"]), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None or rgb.shape != (337, 379, 3) or depth.shape != (337, 379):
            raise ValueError("Invalid cropped RGB-D geometry")
        if depth.dtype != np.uint16 or record["depth_scale_m"] != 0.001:
            raise ValueError("Expected original 16-bit PNG depth, one stored unit = 0.001 m")
        if len({record["color_stamp_ns"], record["depth_stamp_ns"], record["source_stamp_ns"]}) != 1:
            raise ValueError("RGB-D source timestamps differ")
        depth_values.extend([int(depth.min()), int(depth.max())])
    files = [p for p in destination.rglob("*") if p.is_file()]
    return {"checksums_verified": len(entries), "files": len(files),
            "bytes": sum(p.stat().st_size for p in files), "sample_pairs": len(samples),
            "rgbd_pairs": len(rgbd), "png_files": sum(p.suffix == ".png" for p in files),
            "depth_stored_min_max": [min(depth_values), max(depth_values)]}


def build(destination):
    copied = []

    def copy_original(source, relative, kind):
        source, target = Path(source), destination / relative
        expected = sha256(source)
        if target.exists() and sha256(target) != expected:
            raise FileExistsError(f"Refusing to replace different existing asset: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        if sha256(target) != expected:
            raise ValueError(f"Copy verification failed: {target}")
        copied.append({"path": str(relative), "kind": kind, "bytes": target.stat().st_size,
                       "sha256": expected, "original_source": provenance(source)})

    source_split = read_json(ROOT / "datasets/split.json")
    v2_split = read_json(ROOT / V2 / "split.json")
    used = {key: {Path(x["rgb"]).name for x in records} for key, records in v2_split.items()}
    selection = np.linspace(0, len(source_split["records"]) - 1, 50, dtype=int)
    sample_records = []
    for index in selection:
        record = source_split["records"][int(index)]
        rgb, mask = ROOT / "datasets" / record["rgb"], ROOT / "datasets" / record["mask"]
        copy_original(rgb, f"samples/rgb/{rgb.name}", "demo_rgb")
        copy_original(mask, f"samples/masks/{mask.name}", "demo_pseudo_mask")
        stem = rgb.stem.split("_")
        sample_records.append({"rgb": f"rgb/{rgb.name}", "mask": f"masks/{mask.name}",
                               "original_index_zero_based": int(index), "group": record["group"],
                               "split": record["split"],
                               "source_stamp_ns": int(stem[-2]) * 10**9 + int(stem[-1]),
                               "used_in_v2_training": rgb.name in used["train"],
                               "used_in_v2_validation": rgb.name in used["val"],
                               "label_provenance": "opencv_pseudo_label_not_independent_manual_ground_truth"})
    write_json(destination / "samples/index.json", {
        "schema_version": 1, "path_base": ".", "purpose": "Small inference/augmentation demo, NOT an independent test set",
        "selection_rule": "np.linspace(0, 5389 - 1, 50, dtype=int), original datasets/split.json record order",
        "original_source_summary": source_split["summary"], "records": sample_records})

    # Include the complete training membership as metadata, not additional images.
    split_rows = []
    for split, records in v2_split.items():
        for record in records:
            split_rows.append({"split": split, "source_rgb_path": provenance(record["rgb"])["path"],
                               "source_mask_path": provenance(record["mask"])["path"],
                               "session": record["session"], "frame_timestamp_ns": record["frame_timestamp_ns"],
                               "included_in_demo": Path(record["rgb"]).name in {Path(x["rgb"]).name for x in sample_records}})
    split_path = destination / "metadata/v2_training_split_provenance.csv"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(split_rows[0]))
        writer.writeheader()
        writer.writerows(split_rows)

    index = read_json(ROOT / PAPER / "paper_pose_source_index.json")
    original_protocol = read_json(ROOT / PROTOCOL)
    if [(x["dataset"], x["frame_id"]) for x in index] != [
            (x["dataset"], x["frame_id"]) for x in original_protocol["records"]]:
        raise ValueError("Paper source index no longer matches the evaluated protocol")
    rgbd_records = []
    live_pairs = {}
    archive_indexes = {}
    for original in index:
        record = {k: copy.deepcopy(v) for k, v in original.items()
                  if k not in {"rgb", "depth", "evaluation_outputs", "source_manifest"}}
        for name in ("rgb", "depth"):
            source = Path(original[name])
            relative = f"{original['dataset']}/{name}/{source.name}"
            copy_original(source, f"rgbd/{relative}", f"paper_{name}")
            record[name] = relative
        if original["dataset"] == "live_black":
            manifest = Path(original["source_manifest"])
            if manifest not in live_pairs:
                live_pairs[manifest] = {Path(x["color_file"]).stem: x for x in read_json(manifest)["frames"]}
            pair = live_pairs[manifest][original["frame_id"]]
            record["color_stamp_ns"] = int(pair["color_stamp_ns"])
            record["depth_stamp_ns"] = int(pair["depth_stamp_ns"])
            if Path(pair["depth_file"]).name != Path(original["depth"]).name:
                raise ValueError("Unexpected live source depth pairing")
        else:
            for name, stamp_key in (("rgb", "color_stamp_ns"), ("depth", "depth_stamp_ns")):
                source = Path(original[name])
                csv_path = source.parent / "index.csv"
                if csv_path not in archive_indexes:
                    with csv_path.open(newline="") as stream:
                        archive_indexes[csv_path] = {x["filename"]: x for x in csv.DictReader(stream)}
                record[stamp_key] = int(archive_indexes[csv_path][source.name]["source_time_ns"])
        if len({record["color_stamp_ns"], record["depth_stamp_ns"], record["source_stamp_ns"]}) != 1:
            raise ValueError("Original RGB-D source timestamps differ")
        record["timestamp_pairing"] = "Exact RGB/depth source timestamp; NOT file sequence counter"
        record["posture_label"] = "straight" if original["dataset"] == "live_black" else "bent"
        record["posture_label_note"] = "Acquisition-condition grouping, not a per-frame curvature threshold"
        rgbd_records.append(record)
    write_json(destination / "rgbd/index.json", {"schema_version": 1, "path_base": ".",
        "purpose": "Exact 104 source RGB-D pairs from the paper occlusion sensitivity evaluation",
        "depth_encoding": "16UC1 PNG, depth_m = stored_value * depth_scale_m; zero is invalid",
        "crop_intrinsics_note": "crop_k is already crop-local; do not subtract source ROI a second time",
        "no_independent_shape_ground_truth": True, "records": rgbd_records})
    protocol = {k: copy.deepcopy(v) for k, v in original_protocol.items()
                if k not in {"archive_source", "capture", "records", "v1_checkpoint", "v2_checkpoint", "methods"}}
    protocol["original_methods"] = original_protocol["methods"]
    protocol["methods"] = ["opencv", "unet_v2"]
    protocol["v2_checkpoint"] = "../models/unet_v2.pt"
    protocol["record_index"] = "../rgbd/index.json"
    protocol["omitted_comparator"] = "U-Net V1 checkpoint is intentionally not bundled"
    protocol["per_frame_seed_rule"] = "61024 + zlib.crc32(f'{dataset}:{source_stamp_ns}'.encode())"
    protocol["occlusion_grade"] = {"clean": [0, 0.0], "thin": [1, 0.02], "thick10": [1, 0.1],
        "thick20": [1, 0.2], "thick30": [1, 0.3], "multi3_15": [3, 0.15], "multi3_30": [3, 0.3]}
    protocol["occlusion_grade_note"] = "[stripe count, requested total centerline occlusion fraction]; actual coverage is measured per frame"
    write_json(destination / "metadata/pose_protocol.json", protocol)
    copy_original(ROOT / V2 / "best.pt", "models/unet_v2.pt", "checkpoint")
    copy_original(ROOT / V2 / "config.json", "metadata/v2_training_config.original.json", "training_config")
    copy_original(ROOT / PAPER / "paper_pose_summary.csv", "metadata/reference_paper_pose_summary.csv", "reference_statistics")
    copy_original(ROOT / PAPER / "paper_pose_frames.csv", "metadata/reference_paper_pose_frames.csv", "reference_statistics")
    write_json(destination / "metadata/source_mapping.json", {
        "note": "Original locations are provenance only; all runnable paths are relative in the indexes",
        "records": copied})
    counts = collections.Counter(x["kind"] for x in copied)
    sizes = {kind: sum(x["bytes"] for x in copied if x["kind"] == kind) for kind in counts}
    write_json(destination / "metadata/asset_report.json", {
        "copied_original_files": len(copied), "copied_original_bytes": sum(x["bytes"] for x in copied),
        "copy_checksums_verified_against_originals": True, "counts_by_kind": dict(counts), "bytes_by_kind": sizes,
        "source_manifests": [{"original_source": provenance(ROOT / p), "sha256": sha256(ROOT / p)}
                             for p in (Path("datasets/split.json"), V2 / "split.json", PAPER / "paper_pose_source_index.json", PROTOCOL)],
        "sample_split_counts": dict(collections.Counter(x["split"] for x in sample_records)),
        "v2_actual_split_counts": {k: len(v) for k, v in v2_split.items()},
        "paper_posture_counts": dict(collections.Counter(x["posture_label"] for x in rgbd_records)),
        "limitations": ["Demo includes training examples; not a held-out benchmark", "Masks are OpenCV pseudo-labels",
                        "Paper evaluation measures paired changes, not physical ground-truth accuracy",
                        "Known synthetic occluder mask rejects depth, not real-world occluders",
                        "Full training datasets/backgrounds/V1/MobileSAM are not included"]})
    files = sorted(p for p in destination.rglob("*") if p.is_file() and p.name != "checksums.sha256")
    (destination / "checksums.sha256").write_text("".join(
        f"{sha256(path)}  {path.relative_to(destination).as_posix()}\n" for path in files))
    return verify(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/publication_demo")
    parser.add_argument("--verify-only", action="store_true", help="Check included files without reading original datasets")
    args = parser.parse_args()
    destination = args.output.resolve()
    report = verify(destination) if args.verify_only else build(destination)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
