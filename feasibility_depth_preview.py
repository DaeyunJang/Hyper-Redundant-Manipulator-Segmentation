#!/usr/bin/env python3
"""Offline observed-point diagnostic, NOT a 3D shape/angle reconstruction test.

Given paired fresh crops and predicted visible-body masks, retain all mask
components, Lee-thin each visible region and deproject only in-mask skeleton
pixels. The local depth median also excludes mask-zero neighbors. No filling
across gaps, polynomial fit, base registration, angle/FK computation or ROS
publication is performed. A false-positive mask can still admit occluder depth.

The camera capture is unoccluded. This diagnostic neither creates physical
occlusion nor demonstrates hidden-body reconstruction. It assumes the saved
image_rect_raw pixels have a pinhole geometry described by crop-local K; D is
preserved in the report but no lens-calibration accuracy is claimed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize


def _binary_mask(mask):
    mask = np.asarray(mask)
    if (mask.ndim != 2 or min(mask.shape) == 0 or not np.isin(mask, [0, 1, 255]).all()):
        raise ValueError("Expected nonempty-shape 2D binary predicted mask with 0/1/255 values.")
    return mask > 0


def mask_aware_depth_samples(depth, visible_mask, pixels_yx, depth_scale_m=0.001,
                             neighborhood_size=5, minimum_valid=3):
    """Median depth at candidate pixels using ONLY visible-mask neighbors.

    Returns accepted pixel coordinates, meter depths, contributing neighbor
    counts and a boolean vector aligned with the input candidates. The query
    center itself must be mask-positive, even if surrounding pixels are valid.
    There is no fallback to out-of-mask neighbors or to a stale depth sample.
    """
    mask = _binary_mask(visible_mask)
    depth = np.asarray(depth)
    pixels = np.asarray(pixels_yx)
    if depth.ndim != 2 or depth.shape != mask.shape:
        raise ValueError("Depth and visible mask must have identical 2D dimensions.")
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.issubdtype(pixels.dtype, np.integer):
        raise ValueError("pixels_yx must be an Nx2 integer array.")
    if (not np.isfinite(depth_scale_m) or depth_scale_m <= 0
            or neighborhood_size < 1 or neighborhood_size % 2 != 1
            or minimum_valid < 1 or minimum_valid > neighborhood_size ** 2):
        raise ValueError("Invalid depth scale/neighborhood/minimum-valid configuration.")
    height, width = depth.shape
    if ((pixels < 0).any() or (pixels[:, 0] >= height).any()
            or (pixels[:, 1] >= width).any()):
        raise ValueError("Candidate pixel is outside the cropped image.")
    depth_m = depth.astype(np.float64) * depth_scale_m
    usable = mask & np.isfinite(depth_m) & (depth_m > 0)
    radius = neighborhood_size // 2
    accepted = np.zeros(len(pixels), dtype=bool)
    medians, counts = [], []
    for i, (y, x) in enumerate(pixels):
        if not mask[y, x]:
            continue
        y0, y1, x0, x1 = max(0, y-radius), min(height, y+radius+1), max(0, x-radius), min(width, x+radius+1)
        values = depth_m[y0:y1, x0:x1][usable[y0:y1, x0:x1]]
        if len(values) < minimum_valid:
            continue
        accepted[i] = True
        medians.append(float(np.median(values)))
        counts.append(len(values))
    return (pixels[accepted].astype(np.int32), np.asarray(medians, dtype=np.float64),
            np.asarray(counts, dtype=np.int32), accepted)


def deproject_crop_pixels(pixels_yx, depth_m, crop_k):
    """Use already-shifted crop K directly; no second ROI offset is accepted."""
    pixels, depths = np.asarray(pixels_yx), np.asarray(depth_m, dtype=np.float64)
    k = np.asarray(crop_k, dtype=np.float64)
    if k.size != 9:
        raise ValueError("crop_k must contain a 3x3 intrinsic matrix.")
    k = k.reshape(3, 3)
    if (not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1]) or abs(np.linalg.det(k)) < 1e-12):
        raise ValueError("Invalid crop intrinsic matrix.")
    if (pixels.ndim != 2 or pixels.shape[1] != 2 or depths.shape != (len(pixels),)
            or not np.isfinite(pixels).all() or not np.isfinite(depths).all()
            or (depths <= 0).any()):
        raise ValueError("Expected finite Nx2 pixels and N strictly positive meter depths.")
    uv1 = np.column_stack([pixels[:, 1], pixels[:, 0], np.ones(len(pixels))])
    rays = uv1 @ np.linalg.inv(k).T
    rays /= rays[:, 2:3]
    return rays * depths[:, None]


def extract_observed_cloud(depth, visible_mask, crop_k, depth_scale_m=0.001,
                           neighborhood_size=5, minimum_valid=3):
    mask = _binary_mask(visible_mask)
    skeleton = skeletonize(mask, method="lee").astype(bool)
    candidate_yx = np.argwhere(skeleton).astype(np.int32)
    pixels, depth_m, counts, valid = mask_aware_depth_samples(
        depth, mask, candidate_yx, depth_scale_m, neighborhood_size, minimum_valid)
    xyz = deproject_crop_pixels(pixels, depth_m, crop_k)
    accepted_image = np.zeros(mask.shape, dtype=np.uint8)
    if len(pixels):
        accepted_image[pixels[:, 0], pixels[:, 1]] = 1
    return dict(points_camera_xyz_m=xyz, pixels_yx=pixels, depth_m=depth_m,
                neighbor_counts=counts, skeleton=skeleton.astype(np.uint8),
                candidate_pixels_yx=candidate_yx, candidate_valid=valid,
                accepted_pixels_image=accepted_image,
                mask_component_count=int(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0]-1))


def diagnostic_overlay(bgr, depth, mask, cloud, depth_scale_m):
    """Original RGB and depth side-by-side; green=used, red=depth-rejected."""
    left = bgr.copy()
    selected = mask > 0
    left[selected] = (0.70 * left[selected] + 0.30 * np.array([255, 170, 0])).astype(np.uint8)
    depth_m = depth.astype(np.float64) * depth_scale_m
    valid = np.isfinite(depth_m) & (depth_m > 0)
    # Use body range when present, otherwise valid full-image range for display.
    display_values = depth_m[valid & selected]
    if not display_values.size:
        display_values = depth_m[valid]
    if display_values.size:
        low, high = np.percentile(display_values, [2, 98])
        high = max(float(high), float(low) + .001)
    else:
        low, high = 0., 1.
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    scaled[valid] = np.clip((depth_m[valid]-low) / (high-low) * 255, 0, 255).astype(np.uint8)
    right = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    right[~valid] = 0
    for pixel, accepted in zip(cloud["candidate_pixels_yx"], cloud["candidate_valid"]):
        y, x = map(int, pixel)
        color = (0, 255, 0) if accepted else (0, 0, 255)
        cv2.circle(left, (x, y), 1, color, -1)
        cv2.circle(right, (x, y), 1, color, -1)
    image = np.concatenate([left, right], axis=1)
    title = np.zeros((45, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(title, "Observed pixels only | green: used / red: rejected", (5, 17),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    cv2.putText(title, f"{len(cloud['pixels_yx'])} points; camera frame, m; no curve/angle reconstruction", (5, 36),
                cv2.FONT_HERSHEY_SIMPLEX, .42, (220, 220, 220), 1)
    return np.concatenate([title, image]), [float(low), float(high)]


def verify_clean_prediction_provenance(capture, mask_dir, frames, mask_suffix=""):
    """Reject synthetic/stale/mismatched masks before any depth export.

    Only the completed evaluate_live_feasibility.py output contract is accepted:
    predicted_masks/<method>/clean + report.json + inference_status.csv. The
    capture manifest hash, source timestamp/path, and decoded inference RGB
    pixels must all match. Merely renaming a synthetic folder to 'clean' cannot
    bypass the per-row condition and original-RGB equality checks.
    """
    capture, mask_dir = Path(capture).resolve(), Path(mask_dir).resolve()
    if mask_dir.name != "clean" or mask_dir.parent.parent.name != "predicted_masks":
        raise ValueError("Depth preview accepts only predicted_masks/<method>/clean; synthetic masks are forbidden.")
    evaluation = mask_dir.parent.parent.parent
    report_file, csv_file = evaluation / "report.json", evaluation / "inference_status.csv"
    if not report_file.is_file() or not csv_file.is_file():
        raise ValueError("Completed live inference report.json and inference_status.csv are required.")
    report = json.loads(report_file.read_text())
    manifest_hash = hashlib.sha256((capture / "manifest.json").read_bytes()).hexdigest()
    if report.get("source_manifest_sha256") != manifest_hash:
        raise ValueError("Capture manifest hash mismatch: predictions may belong to another capture.")
    with csv_file.open(newline="") as stream:
        statuses = list(csv.DictReader(stream))
    verified = {}
    for frame in frames:
        original_path = (capture / frame["color_file"]).resolve()
        mask_path = mask_dir / f"{original_path.stem}{mask_suffix}.png"
        relative_mask = mask_path.relative_to(evaluation).as_posix()
        matches = [row for row in statuses if row.get("predicted_mask") == relative_mask]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one inference record for {relative_mask}; got {len(matches)}")
        row = matches[0]
        if row.get("condition") != "clean":
            raise ValueError(f"Synthetic/non-clean inference row rejected: {relative_mask}")
        if row.get("method") != mask_dir.parent.name:
            raise ValueError("Prediction method folder and inference record disagree.")
        if int(row["source_stamp_ns"]) != int(frame["color_stamp_ns"]):
            raise ValueError(f"Source timestamp mismatch: {original_path.name}")
        if (capture / row["source_color_file"]).resolve() != original_path:
            raise ValueError(f"Source color path mismatch: {original_path.name}")
        inference_rgb = (evaluation / row["input_rgb"]).resolve()
        if not inference_rgb.is_relative_to(evaluation) or not inference_rgb.is_file():
            raise ValueError(f"Invalid saved inference RGB path: {inference_rgb}")
        original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
        inferred = cv2.imread(str(inference_rgb), cv2.IMREAD_COLOR)
        if original is None or inferred is None or not np.array_equal(original, inferred):
            raise ValueError(f"Inference RGB differs from original capture: {original_path.name}")
        verified[original_path.name] = dict(
            inference_condition="clean", inference_status=row.get("status"),
            inference_source_rgb=str(inference_rgb),
            inference_source_rgb_sha256=hashlib.sha256(inference_rgb.read_bytes()).hexdigest())
    return verified, dict(
        evaluation_root=str(evaluation),
        report_sha256=hashlib.sha256(report_file.read_bytes()).hexdigest(),
        inference_csv_sha256=hashlib.sha256(csv_file.read_bytes()).hexdigest(),
        checks=["completed inference report", "capture manifest hash", "clean condition",
                "matching prediction method", "source timestamp", "source RGB path",
                "inference RGB pixel-for-pixel equality to capture"],
        verified_frames=len(verified))


def export_preview(capture, mask_dir, output, mask_suffix="", limit=None,
                   neighborhood_size=5, minimum_valid=3):
    capture, mask_dir, output = map(lambda p: Path(p).resolve(), (capture, mask_dir, output))
    if output.exists():
        raise FileExistsError(f"Output exists; choose a new directory: {output}")
    manifest_file = capture / "manifest.json"
    capture_manifest = json.loads(manifest_file.read_text())
    frames = capture_manifest["frames"]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        frames = frames[:limit]
    if not frames:
        raise ValueError("Capture has no paired frames.")
    verified, provenance = verify_clean_prediction_provenance(capture, mask_dir, frames, mask_suffix)
    prepared = []
    for frame in frames:
        rgb_path = capture / frame["color_file"]
        mask_path = mask_dir / f"{rgb_path.stem}{mask_suffix}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing predicted mask: {mask_path}")
        prepared.append((frame, rgb_path, mask_path))
    output.mkdir(parents=True)
    (output / "points").mkdir()
    (output / "overlays").mkdir()
    rows = []
    for frame, rgb_path, mask_path in prepared:
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        depth_path = capture / frame["depth_file"]
        depth = (np.load(depth_path, allow_pickle=False) if depth_path.suffix == ".npy"
                 else cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED))
        if bgr is None or mask is None or depth is None or bgr.shape[:2] != mask.shape:
            raise ValueError(f"Invalid RGB/depth/mask pair: {rgb_path.name}")
        info = frame["crop_color_camera_info"]
        if (info["height"], info["width"]) != mask.shape:
            raise ValueError("Crop intrinsic image dimensions do not match the prediction mask")
        scale = float(frame["depth_scale_m"])
        cloud = extract_observed_cloud(depth, mask, info["K"], scale, neighborhood_size, minimum_valid)
        name = rgb_path.stem
        np.savez_compressed(output / "points" / f"{name}.npz", **cloud,
                            crop_K=np.array(info["K"]).reshape(3, 3),
                            source_stamp_ns=np.int64(frame["color_stamp_ns"]))
        overlay, depth_range = diagnostic_overlay(bgr, depth, mask, cloud, scale)
        if not cv2.imwrite(str(output / "overlays" / f"{name}.jpg"), overlay):
            raise OSError(f"Could not save diagnostic overlay {name}")
        points = cloud["points_camera_xyz_m"]
        rows.append(dict(
            filename=rgb_path.name, source_stamp_ns=frame["color_stamp_ns"],
            frame_id=frame["frame_id"], mask_file=str(mask_path),
            mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            candidate_skeleton_points=len(cloud["candidate_pixels_yx"]),
            valid_observed_points=len(points), mask_component_count=cloud["mask_component_count"],
            points_finite=bool(np.isfinite(points).all()),
            point_min_xyz_m=points.min(axis=0).tolist() if len(points) else None,
            point_max_xyz_m=points.max(axis=0).tolist() if len(points) else None,
            depth_overlay_range_m=depth_range,
            saved_nonzero_distortion_coefficients=bool(np.any(np.asarray(info.get("D", [])))),
            **verified[rgb_path.name],
        ))
    report = dict(
        purpose="Observed skeleton-point RGB-D datapath diagnostic only",
        capture=str(capture), mask_dir=str(mask_dir), frames=len(rows),
        capture_manifest_sha256=hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        prediction_provenance_verification=provenance,
        coordinate_frame="camera optical frame", units="meters",
        intrinsic_policy="crop-local K directly; no additional ROI offset",
        distortion_policy="pinhole diagnostic on image_rect_raw; no D correction; no calibration validation",
        depth_scale_policy="each capture frame's explicit depth_scale_m",
        neighborhood_size=neighborhood_size, minimum_valid_neighbors=minimum_valid,
        depth_neighbor_policy="mask-positive AND positive finite depth; median; no fallback",
        skeleton_policy="Lee thinning of predicted visible mask, all components retained, no bridging",
        point_order="row-major pixel order, NOT an ordered base-to-tip curve",
        synthetic_occlusion=False, physical_occlusion_test=False,
        curve_fit_performed=False, joint_angles_computed=False, fk_computed=False,
        limitations=["Predicted-mask false positives can admit background/occluder depth.",
                     "Fresh capture has no independent manual mask or 3D ground truth.",
                     "No hidden-body depth or shape is inferred and no occlusion robustness is claimed."],
        records=rows,
    )
    (output / "manifest.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-suffix", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--neighborhood-size", type=int, default=5)
    parser.add_argument("--minimum-valid", type=int, default=3)
    args = parser.parse_args()
    report = export_preview(args.capture, args.mask_dir, args.output, args.mask_suffix,
                            args.limit, args.neighborhood_size, args.minimum_valid)
    print(json.dumps({"output": str(args.output.resolve()), "frames": report["frames"],
                      "valid_points_total": sum(r["valid_observed_points"] for r in report["records"])}))


if __name__ == "__main__":
    main()
