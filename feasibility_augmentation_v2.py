# AUTHOR: Daeyun - HRM multiple-gap and projective augmentation protocol.
"""Stronger, explicit visible-HRM augmentation; preserves the first pilot.

All public transforms return (RGB, visible_mask, JSON_metadata, occluder_mask).
Both masks are uint8 {0,1} at input resolution. The fourth return is the known
SYNTHETIC opaque occluder map, not a learned detector and not ground truth that
may be supplied to a segmentation model at test time. It can invalidate true
HRM depth during a controlled missing-depth stress test.

Projective changes are 2D surface-plane appearance transforms only, NOT novel
3D viewpoints, depth renderings or changes to the HRM's articulated geometry.
Image and target are transformed together. Use only geometry-preserving test
conditions with original depth/intrinsics. Background split ownership remains
the caller's responsibility; no files or external datasets are read implicitly.
"""

from __future__ import annotations

import cv2
import numpy as np

import augment
from feasibility_data import LABEL_PROVENANCE, ordered_centerline


CONDITIONS_V2 = ("clean", "thin", "thick10", "thick20", "thick30", "multi3_15",
                 "multi3_30", "perspective", "bg_perspective", "bg_perspective_multi3")
GEOMETRY_PRESERVING_CONDITIONS_V2 = ("clean", "thin", "thick10", "thick20", "thick30",
                                     "multi3_15", "multi3_30")


def _metadata(mask, condition):
    count = int(mask.sum())
    return dict(condition=condition, label_provenance=LABEL_PROVENANCE,
                source_foreground_pixels=count, full_foreground_pixels=count,
                visible_foreground_pixels=count, background_replaced=False,
                geometry_changed=False, original_depth_geometry_compatible=True,
                geometry_claim="2D appearance only; no true 3D viewpoint synthesis",
                source_to_output_homography=np.eye(3).tolist(),
                requested_centerline_occlusion_fraction=0.,
                actual_centerline_occlusion_fraction=0.,
                foreground_occlusion_fraction=0., has_actual_occlusion=False,
                occluder_count=0, occluder_pixels=0, endpoints_covered=[False, False],
                endpoint_regions_covered=[False, False], negative=False,
                hard_negative_objects=0, centerline_method="longest_8connected_skeleton_path",
                depth_policy="invalidate known synthetic occluder ONLY in geometry-compatible missing-depth tests")


def _projective_foreground(rgb, mask, rng, backgrounds, replace_background, strong):
    """Homography on premultiplied RGB + nearest-mask, all body inside frame."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) < 16:
        raise ValueError("Projective HRM augmentation needs at least 16 foreground pixels")
    x0, x1 = float(xs.min() - 2), float(xs.max() + 2)
    y0, y1 = float(ys.min() - 2), float(ys.max() + 2)
    source = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    bw, bh = x1 - x0, y1 - y0
    # Convex trapezoids taper at one end and shear. This changes apparent plane
    # perspective, not physical 3D HRM centerline/articulation.
    taper_top, taper_bottom = rng.uniform(.55, 1.35, 2)
    shear_top, shear_bottom = rng.uniform(-.25, .25, 2) * bw
    destination = np.float64([
        [-bw * taper_top / 2 + shear_top, -bh / 2],
        [ bw * taper_top / 2 + shear_top, -bh / 2],
        [ bw * taper_bottom / 2 + shear_bottom, bh / 2],
        [-bw * taper_bottom / 2 + shear_bottom, bh / 2],
    ])
    angle = float(rng.uniform(-180, 180) if strong else rng.uniform(-20, 20))
    a = np.deg2rad(angle)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    destination = destination @ rotation.T
    extent = np.ptp(destination, axis=0)
    requested_scale = float(rng.uniform(.55, 1.40) if strong else rng.uniform(.85, 1.10))
    fit = min((w - 12) / extent[0], (h - 12) / extent[1])
    scale = min(requested_scale, fit)
    destination *= scale
    lower, upper = destination.min(0), destination.max(0)
    translation_low = np.array([6 - lower[0], 6 - lower[1]])
    translation_high = np.array([w - 6 - upper[0], h - 6 - upper[1]])
    # When one dimension exactly fits, floating-point roundoff may invert its
    # zero-width sampling interval by 1e-14. Clamp only this numerical case.
    if np.any(translation_high < translation_low - 1e-7):
        raise ValueError("Projective sprite exceeds available output extent")
    translation_high = np.maximum(translation_low, translation_high)
    if strong:
        translation = rng.uniform(translation_low, translation_high)
    else:
        desired = np.array([xs.mean(), ys.mean()]) + rng.uniform(-.03, .03, 2) * [w, h]
        translation = np.clip(desired, translation_low, translation_high)
    destination += translation
    transform = cv2.getPerspectiveTransform(source, destination.astype(np.float32))
    target = cv2.warpPerspective(mask, transform, (w, h), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = cv2.warpPerspective(mask.astype(np.float32), transform, (w, h))
    color = cv2.warpPerspective(rgb.astype(np.float32) * mask[..., None], transform, (w, h))
    color /= np.maximum(alpha[..., None], 1e-6)
    if replace_background:
        background = augment._background(mask.shape, list(backgrounds), rng)
    else:
        # Remove the original foreground so moving it does not leave a ghost.
        background = cv2.inpaint(rgb, mask * 255, 5., cv2.INPAINT_TELEA)
    image = np.where(target[..., None] != 0, np.clip(color, 0, 255), background).astype(np.uint8)
    if not target.any() or target[0].any() or target[-1].any() or target[:, 0].any() or target[:, -1].any():
        raise ValueError("Projective transform did not preserve full visible silhouette")
    info = dict(source_to_output_homography=transform.tolist(), geometry_changed=True,
                original_depth_geometry_compatible=False, all_foreground_inside_frame=True,
                rotation_degrees=angle, requested_scale=requested_scale, applied_scale=float(scale),
                perspective_taper_top=float(taper_top), perspective_taper_bottom=float(taper_bottom),
                geometric_effects=["planar_projective_taper", "rotation", "scale", "translation"],
                full_foreground_pixels=int(target.sum()), background_replaced=bool(replace_background))
    return image, target, info


def _fill_occluder(shape, rng):
    """Black, gray, white and colored opaque materials, independent of labels."""
    material = str(rng.choice(["black", "gray", "white", "color"]))
    if material == "black":
        color = np.repeat(rng.uniform(0, 30), 3)
    elif material == "gray":
        color = np.repeat(rng.uniform(60, 170), 3)
    elif material == "white":
        color = np.repeat(rng.uniform(210, 255), 3)
    else:
        color = rng.uniform(15, 235, 3)
    texture = cv2.GaussianBlur(rng.normal(0, 6, shape).astype(np.float32), (5, 5), 0)
    return np.clip(color[None, None, :] + texture[..., None], 0, 255).astype(np.uint8), material


def _occlude_stripes(rgb, mask, rng, count, fraction):
    """Opaque separated stripes; actual baseline centerline coverage measured."""
    points, arc = ordered_centerline(mask)
    length = float(arc[-1])
    path_x, path_y = points[:, 0].astype(int), points[:, 1].astype(int)
    delta = np.diff(arc)
    h, w = mask.shape
    yy, xx = np.mgrid[:h, :w]
    protected = (arc <= .07 * length) | (arc >= .93 * length)
    # Three disjoint arc windows; every stripe has its own local tangent.
    centers = (np.array([rng.uniform(.37, .63)]) if count == 1 else
               np.linspace(.24, .76, count) + rng.uniform(-.025, .025, count))
    boundaries = np.r_[.07, (centers[:-1] + centers[1:]) / 2, .93] * length
    union = np.zeros(mask.shape, dtype=bool)
    result, target = rgb.copy(), mask.copy()
    details = []
    for i, center in enumerate(centers):
        s = float(center * length)
        pivot = np.array([np.interp(s, arc, points[:, d]) for d in range(2)])
        span = max(2., .025 * length)
        before = np.array([np.interp(s - span, arc, points[:, d]) for d in range(2)])
        after = np.array([np.interp(s + span, arc, points[:, d]) for d in range(2)])
        tangent = after - before
        tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
        signed = (points - pivot) @ tangent
        projection = np.abs((xx - pivot[0]) * tangent[0] + (yy - pivot[1]) * tangent[1])
        forbidden = (arc <= boundaries[i]) | (arc >= boundaries[i + 1])
        hi = float(np.min(np.abs(signed[forbidden]))) - .01
        if hi <= 0:
            raise ValueError("Cannot place separated stripes on this centerline")
        lo, best = 0., None
        requested = fraction / count
        for _ in range(20):
            width = (lo + hi) / 2
            candidate = (projection <= width) & ~union
            covered = candidate[path_y, path_x]
            actual = float(np.sum(delta * (covered[:-1].astype(float) + covered[1:]) / 2) / length)
            error = abs(actual - requested)
            if best is None or error < best[0]:
                best = error, candidate, covered, actual, width
            if actual < requested:
                lo = width
            else:
                hi = width
        _, occluder, covered, actual, width = best
        if not np.any(occluder & (mask != 0)) or covered[protected].any():
            raise ValueError("Occlusion has no visible target removal or covers endpoints")
        if abs(actual - requested) > max(.015, 2 / length):
            raise ValueError("Unable to realize graded separated occlusion")
        fill, material = _fill_occluder(mask.shape, rng)
        result, target = augment.apply_occluder(result, target, occluder.astype(np.uint8), fill)
        union |= occluder
        details.append(dict(center_fraction=float(center), material=material,
                            width_pixels=float(2 * width),
                            actual_centerline_fraction=actual))
    covered = union[path_y, path_x]
    actual = float(np.sum(delta * (covered[:-1].astype(float) + covered[1:]) / 2) / length)
    removed = int(np.count_nonzero(union & (mask != 0)))
    info = dict(requested_centerline_occlusion_fraction=float(fraction),
                actual_centerline_occlusion_fraction=actual,
                foreground_occlusion_fraction=removed / max(int(mask.sum()), 1),
                full_foreground_pixels=int(mask.sum()), visible_foreground_pixels=int(target.sum()),
                has_actual_occlusion=True, occluder_count=count,
                occluder_pixels=int(union.sum()), occluders=details,
                endpoints_covered=[bool(covered[0]), bool(covered[-1])],
                endpoint_regions_covered=[bool(covered[arc <= .07 * length].any()),
                                          bool(covered[arc >= .93 * length].any())],
                centerline_length_pixels=length)
    return result, target, info, union.astype(np.uint8)


def _hard_negatives(rgb, mask, rng):
    """Distractor white/gray ellipses/rods in background, never over foreground."""
    image = rgb.copy()
    h, w = mask.shape
    safe = cv2.dilate(mask, np.ones((7, 7), np.uint8)) == 0
    count = int(rng.integers(1, 5))
    for _ in range(count):
        region = np.zeros_like(mask)
        center = (int(rng.integers(w)), int(rng.integers(h)))
        axes = (int(rng.integers(4, max(5, w // 5))), int(rng.integers(3, max(4, h // 12))))
        cv2.ellipse(region, center, axes, float(rng.uniform(0, 180)), 0, 360, 1, -1)
        color = np.repeat(rng.uniform(160, 250), 3)
        image[(region != 0) & safe] = color.astype(np.uint8)
    return image, count


def make_case_v2(rgb, mask, seed, condition, backgrounds=()):
    """Deterministic held-out stress condition; four returns, no input mutation."""
    augment._validate(rgb, mask)
    if condition not in CONDITIONS_V2:
        raise ValueError(f"Unknown v2 condition: {condition}")
    image, target = rgb.copy(), mask.copy()
    occ = np.zeros_like(mask)
    meta = _metadata(mask, condition)
    geometry_rng = np.random.default_rng(int(seed) ^ 0x617AC2)
    occlusion_rng = np.random.default_rng(int(seed) ^ 0x0CC10DE)
    if "perspective" in condition:
        image, target, info = _projective_foreground(image, target, geometry_rng, backgrounds,
                                                    condition.startswith("bg_"),
                                                    strong=condition.startswith("bg_"))
        meta.update(info)
    grade = {"thin": (1, .02), "thick10": (1, .10), "thick20": (1, .20),
             "thick30": (1, .30), "multi3_15": (3, .15), "multi3_30": (3, .30),
             "bg_perspective_multi3": (3, .30)}
    if condition in grade:
        count, fraction = grade[condition]
        image, target, info, occ = _occlude_stripes(image, target, occlusion_rng, count, fraction)
        meta.update(info)
    meta.update(seed=int(seed), condition=condition, visible_foreground_pixels=int(target.sum()))
    return np.ascontiguousarray(image), np.ascontiguousarray(target), meta, np.ascontiguousarray(occ)


def augment_v2(rgb, mask, rng, backgrounds=()):
    """Training: clean/negative retain, stronger geometry, diverse opaque strips.

    Source RGB+mask -> optional homography/background -> optional thin/thick/
    multiple opacity -> appearance effects. Source depths may NEVER accompany
    changed geometry without a new rendering. Metadata distinguishes the case.
    """
    augment._validate(rgb, mask)
    image, target = rgb.copy(), mask.copy()
    meta, occ = _metadata(mask, "train_v2"), np.zeros_like(mask)
    branch = float(rng.random())
    if branch < .05:
        image = augment._background(mask.shape, list(backgrounds), rng)
        target.fill(0)
        image, negatives = _hard_negatives(image, target, rng)
        image = augment._camera(augment._lighting(image, rng), rng)
        meta.update(negative=True, full_foreground_pixels=0, visible_foreground_pixels=0,
                    background_replaced=True, hard_negative_objects=negatives)
        return image, target, meta, occ
    if branch < .15:
        if rng.random() < .5:
            image = augment._camera(augment._lighting(image, rng), rng)
        return image, target, meta, occ
    replace = bool(rng.random() < .70)
    if rng.random() < .65:
        image, target, info = _projective_foreground(image, target, rng, backgrounds, replace, strong=True)
        meta.update(info)
    elif replace:
        image = augment._composite(image, target, list(backgrounds), rng)
        meta["background_replaced"] = True
    if rng.random() < .25:
        image, negatives = _hard_negatives(image, target, rng)
        meta["hard_negative_objects"] = negatives
    if rng.random() < .70:
        selection = int(rng.integers(3))
        count = 3 if selection == 2 else 1
        fraction = float(rng.uniform(.01, .045) if selection == 0 else rng.uniform(.08, .34))
        image, target, info, occ = _occlude_stripes(image, target, rng, count, fraction)
        meta.update(info)
    if rng.random() < .35:
        image = augment._color(image, target, rng)
    if rng.random() < .75:
        image = augment._lighting(image, rng)
    if rng.random() < .12:
        image = augment._smoke(image, rng)
    if rng.random() < .65:
        image = augment._camera(image, rng)
    meta["visible_foreground_pixels"] = int(target.sum())
    return np.ascontiguousarray(image), np.ascontiguousarray(target), meta, np.ascontiguousarray(occ)
