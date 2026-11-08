# AUTHOR: Daeyun - HRM visible-body occlusion protocol and split handling.
"""Deterministic data and *visible-mask* occlusion feasibility protocol.

All labels remain OpenCV-generated pseudo-labels, not independent ground truth.
The extra 45deg_1 session is disjoint from the CURRENT split manifest only;
historical model-development use is unknown. Nothing here writes source data.

Occlusion is a synthetic opaque stripe crossing the complete body width. Its
requested fraction refers to the arc length of the longest baseline skeleton
path, not foreground area or ROI area. Rasterized actual coverage is recorded.
This is an image-space stress test, not a simulation of physical occluder depth.
"""

from __future__ import annotations

import heapq
import json
import re
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

import augment


ROOT = Path(__file__).resolve().parent
CONDITIONS = (
    "clean", "background", "mid_occlusion_05", "mid_occlusion_10",
    "mid_occlusion_20", "mid_occlusion_30", "mid_occlusion_bg_10",
    "mid_occlusion_bg_20", "low_light", "glare",
)
LABEL_PROVENANCE = "opencv_pseudo_label_not_independent_manual_ground_truth"


def _record(rgb, mask, session, split):
    rgb, mask = Path(rgb).resolve(), Path(mask).resolve()
    if not rgb.is_file() or not mask.is_file():
        raise FileNotFoundError(f"Missing RGB/mask pair: {rgb}, {mask}")
    match = re.fullmatch(r"(\d+)_(\d+)_(\d+)", rgb.stem)
    if not match:
        raise ValueError(f"Expected frame_seconds_nanoseconds filename: {rgb.name}")
    frame, seconds, nanos = map(int, match.groups())
    timestamp_ns = seconds * 1_000_000_000 + nanos
    return dict(rgb=str(rgb), mask=str(mask), session=session, group=session,
                split=split, frame=frame, frame_timestamp_ns=timestamp_ns,
                frame_timestamp_s=timestamp_ns / 1e9,
                label_provenance=LABEL_PROVENANCE)


def load_records(root=None):
    """Return {'train': [...], 'val': [...], 'test': [...]} with absolute paths.

    Training and validation are unchanged from datasets/split.json. Test is the
    nested NON-legacy 45deg_1 session (1670 RGB/mask pairs in this snapshot).
    Exact source timestamps must be disjoint across splits. Test may contain an
    RGB frame without depth; this API evaluates segmentation, not RGB-D replay.
    """
    root = Path(root or ROOT).resolve()
    data = root / "datasets"
    manifest = json.loads((data / "split.json").read_text())
    result = dict(train=[], val=[], test=[])
    for item in manifest["records"]:
        split = item["split"]
        if split not in result:
            raise ValueError(f"Unknown source split: {split}")
        result[split].append(_record(data / item["rgb"], data / item["mask"],
                                     item["group"], split))
    extra = (data / "20260927_HRM_shape_estimation_apriltag" /
             "shape_estimation_AprilTag_45deg_1" / "images")
    images = sorted((extra / "hrm_crop").glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No held-out candidate RGB images in {extra}")
    result["test"].extend(_record(rgb, extra / "hrm_crop_segmentation" / rgb.name,
                                  "shape_estimation_AprilTag_45deg_1", "test")
                          for rgb in images)
    seen = {}
    for split, records in result.items():
        records.sort(key=lambda r: r["frame_timestamp_ns"])
        for rec in records:
            key = rec["frame_timestamp_ns"]
            if key in seen:
                raise ValueError(f"Duplicate source timestamp across records: {seen[key]} / {rec}")
            seen[key] = rec
    return result


def read_pair(record):
    """Decode absolute-path records as RGB uint8 and visible uint8 {0,1}."""
    bgr = cv2.imread(str(record["rgb"]), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(record["mask"]), cv2.IMREAD_GRAYSCALE)
    if bgr is None or mask is None:
        raise ValueError(f"Cannot decode pair: {record}")
    if not np.isin(mask, [0, 1, 255]).all():
        raise ValueError(f"Nonbinary source mask: {record['mask']}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = (mask > 0).astype(np.uint8)
    augment._validate(image, mask)
    return image, mask


def background_paths(split, root=None):
    """Use only the matching background split; never fall back to train."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown background split: {split}")
    folder = Path(root or ROOT) / "datasets" / "backgrounds" / split
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def ordered_centerline(mask):
    """Longest 8-connected skeleton path, oriented bottom-to-top, in XY pixels.

    A weighted double sweep is exact on a tree and an approximation if the
    skeleton has cycles. Side branches are not part of the measured centerline.
    This baseline is for stress-test placement, not 3D shape ground truth.
    """
    skel = skeletonize(mask.astype(bool)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(skel, connectivity=8)
    if count <= 1:
        raise ValueError("Cannot place graded occlusion: empty skeleton")
    component = int(1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    yx = np.argwhere(labels == component)
    if len(yx) < 8:
        raise ValueError("Cannot place graded occlusion: skeleton shorter than 8 pixels")
    lookup = {tuple(p): i for i, p in enumerate(yx)}
    neighbors = [[] for _ in yx]
    for i, (y, x) in enumerate(yx):
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                       (0, 1), (1, -1), (1, 0), (1, 1)):
            j = lookup.get((y + dy, x + dx))
            if j is not None:
                neighbors[i].append((j, float(np.hypot(dy, dx))))

    def farthest(start):
        distance = np.full(len(yx), np.inf)
        previous = np.full(len(yx), -1, dtype=np.int32)
        distance[start] = 0
        queue = [(0., start)]
        while queue:
            cost, i = heapq.heappop(queue)
            if cost > distance[i]:
                continue
            for j, weight in neighbors[i]:
                candidate = cost + weight
                if candidate < distance[j]:
                    distance[j], previous[j] = candidate, i
                    heapq.heappush(queue, (candidate, j))
        return int(np.argmax(distance)), previous

    start, _ = farthest(int(np.argmax(yx[:, 0])))
    end, previous = farthest(start)
    indices = [end]
    while indices[-1] != start:
        indices.append(int(previous[indices[-1]]))
    points = yx[indices, ::-1].astype(np.float64)
    if points[0, 1] < points[-1, 1]:
        points = points[::-1].copy()
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return points, arc


def _base_metadata(mask, condition):
    area = int(mask.sum())
    return dict(condition=condition, label_provenance=LABEL_PROVENANCE,
                requested_centerline_occlusion_fraction=0.,
                actual_centerline_occlusion_fraction=0.,
                foreground_occlusion_fraction=0.,
                source_foreground_pixels=area,
                full_foreground_pixels=area, visible_foreground_pixels=area,
                has_actual_occlusion=False, endpoints_covered=[False, False],
                endpoint_regions_covered=[False, False],
                occluder_pixels=0, background_replaced=False,
                centerline_method="longest_8connected_skeleton_path",
                depth_policy="not_modified_RGB_segmentation_only")


def _middle_occlusion(image, mask, rng, fraction):
    if not 0 < fraction <= 0.4:
        raise ValueError("Middle occlusion fraction must be in (0,0.4]")
    points, arc = ordered_centerline(mask)
    length = float(arc[-1])
    delta = np.diff(arc)
    h, w = mask.shape
    yy, xx = np.mgrid[:h, :w]
    # Protect 8% of path length at BOTH ends, not just the endpoint pixels.
    protected_path = (arc <= 0.08 * length) | (arc >= 0.92 * length)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    endpoint_radius = max(3., float(max(distance[int(p[1]), int(p[0])]
                                      for p in (points[0], points[-1]))) + 2.)
    protected_pixels = np.zeros_like(mask, dtype=bool)
    for p in (points[0], points[-1]):
        protected_pixels |= ((xx - p[0]) ** 2 + (yy - p[1]) ** 2 <= endpoint_radius ** 2)

    # Different center candidates handle curved silhouettes. No silent no-op.
    for center_fraction in np.r_[rng.uniform(0.35, 0.65), 0.5, 0.4, 0.6]:
        center_s = center_fraction * length
        center = np.array([np.interp(center_s, arc, points[:, i]) for i in range(2)])
        span = max(3., 0.04 * length)
        a = np.array([np.interp(center_s - span, arc, points[:, i]) for i in range(2)])
        b = np.array([np.interp(center_s + span, arc, points[:, i]) for i in range(2)])
        tangent = b - a
        tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
        projection = np.abs((xx - center[0]) * tangent[0] + (yy - center[1]) * tangent[1])
        path_projection = np.abs((points - center) @ tangent)
        lo, hi = 0., min(float(np.min(path_projection[protected_path])) - 0.01,
                         0.5 * length)
        if hi <= 0:
            continue
        best = None
        for _ in range(18):
            width = (lo + hi) / 2.
            occluder = (projection <= width) & ~protected_pixels
            covered = occluder[points[:, 1].astype(int), points[:, 0].astype(int)]
            actual = float(np.sum(delta * (covered[:-1].astype(float) + covered[1:]) / 2.) / length)
            candidate = (abs(actual - fraction), occluder, covered, actual)
            if best is None or candidate[0] < best[0]:
                best = candidate
            if actual < fraction:
                lo = width
            else:
                hi = width
        _, occluder, covered, actual = best
        hidden_pixels = int(np.count_nonzero(occluder & (mask != 0)))
        if hidden_pixels == 0 or actual <= 0 or covered[protected_path].any():
            continue
        if abs(actual - fraction) > max(0.02, 2. / length):
            continue
        # Opaque textured stripe, not a transparent image corruption. No depth
        # is generated: RGB-D evaluation MUST exclude occluded true body depth.
        color = rng.uniform(15, 210, (1, 1, 3))
        texture = cv2.GaussianBlur(rng.normal(0, 7, mask.shape).astype(np.float32), (5, 5), 0)
        fill = np.clip(color + texture[..., None], 0, 255).astype(np.uint8)
        output, target = augment.apply_occluder(image, mask, occluder.astype(np.uint8), fill)
        metadata = _base_metadata(mask, "mid_occlusion")
        metadata.update(requested_centerline_occlusion_fraction=float(fraction),
                        actual_centerline_occlusion_fraction=actual,
                        foreground_occlusion_fraction=hidden_pixels / max(int(mask.sum()), 1),
                        full_foreground_pixels=int(mask.sum()),
                        visible_foreground_pixels=int(target.sum()),
                        has_actual_occlusion=True,
                        endpoints_covered=[bool(covered[0]), bool(covered[-1])],
                        endpoint_regions_covered=[bool(covered[arc <= .08 * length].any()),
                                                  bool(covered[arc >= .92 * length].any())],
                        centerline_length_pixels=length,
                        centerline_points=len(points),
                        center_fraction=float(center_fraction),
                        occluder_pixels=int(occluder.sum()))
        return output, target, metadata
    raise ValueError("Could not realize requested mid occlusion with protected endpoints")


def make_test_case(rgb, mask, seed, condition, backgrounds=()):
    """Fixed test transform; returns RGB, visible mask, JSON-safe metadata.

    Equal seed means equal occlusion center/color across grades and background
    variants. Geometry is unchanged; only opacity masks out target pixels. Any
    inability to produce the requested occlusion is raised, never called clean.
    """
    augment._validate(rgb, mask)
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition}; choose from {CONDITIONS}")
    image, target = rgb.copy(), mask.copy()
    meta = _base_metadata(mask, condition)
    bg_rng = np.random.default_rng(int(seed) ^ 0xBADC0FFE)
    rng = np.random.default_rng(int(seed))
    replace = condition == "background" or "_bg_" in condition
    if replace:
        image = augment._composite(image, target, list(backgrounds), bg_rng)
    if condition.startswith("mid_occlusion"):
        fraction = int(condition.rsplit("_", 1)[1]) / 100.
        image, target, meta = _middle_occlusion(image, target, rng, fraction)
    elif condition == "low_light":
        image = np.clip((image.astype(float) / 255.) ** 1.35 * 255. * 0.55, 0, 255).astype(np.uint8)
    elif condition == "glare":
        ys, xs = np.nonzero(mask)
        if not len(xs):
            raise ValueError("Glare test needs nonempty foreground")
        i = int(rng.integers(len(xs)))
        yy, xx = np.mgrid[:mask.shape[0], :mask.shape[1]]
        spot = np.exp(-((xx-xs[i])**2 + (yy-ys[i])**2) / (2 * (0.1 * max(mask.shape))**2))
        image = np.clip(image.astype(float) + 140 * spot[..., None], 0, 255).astype(np.uint8)
    meta.update(condition=condition, seed=int(seed), background_replaced=replace)
    return np.ascontiguousarray(image), np.ascontiguousarray(target), meta


def augment_pair(rgb, mask, rng, mode="train", backgrounds=()):
    """Training augmentation with explicit visible-mask, middle-occlusion labels.

    basic: modest affine/photometric/camera changes, no replacement or occlusion.
    train: same effects plus background/placement and 60% graded middle occlusion
           on nonnegative, non-clean branches (5% negative, 15% basic retained).
    clean: unchanged. Geometry augmentation here is for RGB segmentation ONLY;
    transformed RGB may not be paired with untransformed depth/intrinsics.
    """
    augment._validate(rgb, mask)
    if mode not in ("train", "basic", "clean"):
        raise ValueError("Training mode must be train/basic/clean")
    image, target = rgb.copy(), mask.copy()
    meta = _base_metadata(mask, mode)
    if mode == "clean":
        return image, target, meta
    choice = float(rng.random())
    if mode == "train" and choice < .05:
        image, target = augment.augment_sample(image, target, rng, list(backgrounds), "negative")
        meta.update(negative=True, full_foreground_pixels=0,
                    visible_foreground_pixels=0, background_replaced=True)
        return image, target, meta
    enhanced = mode == "train" and choice >= .20
    if enhanced and rng.random() < .25:
        image, target, placement = augment.place_foreground(image, target, rng, list(backgrounds),
                                                            return_metadata=True)
        meta.update(placement=placement, background_replaced=True)
    else:
        if rng.random() < .65:
            image, target = augment._affine(image, target, rng)
        if enhanced and rng.random() < .65:
            image = augment._composite(image, target, list(backgrounds), rng)
            meta["background_replaced"] = True
    # Geometry may change area; full_foreground_pixels refers to THIS transformed
    # scene immediately before opacity, not the original source image area.
    meta["full_foreground_pixels"] = int(target.sum())
    if enhanced and target.any() and rng.random() < .60:
        original_meta = meta
        image, target, meta = _middle_occlusion(image, target, rng, float(rng.uniform(.04, .32)))
        meta.update(condition="train", background_replaced=original_meta["background_replaced"])
        meta["source_foreground_pixels"] = original_meta["source_foreground_pixels"]
        if "placement" in original_meta:
            meta["placement"] = original_meta["placement"]
    if rng.random() < .25:
        image = augment._color(image, target, rng)
    if rng.random() < .65:
        image = augment._lighting(image, rng)
    if enhanced and rng.random() < .10:
        image = augment._smoke(image, rng)
    if rng.random() < .50:
        image = augment._camera(image, rng)
    meta["visible_foreground_pixels"] = int(target.sum())
    return np.ascontiguousarray(image), np.ascontiguousarray(target), meta
