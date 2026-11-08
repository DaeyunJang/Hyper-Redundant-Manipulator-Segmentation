# AUTHOR: Daeyun - Base/tip-oriented HRM planar perspective augmentation.
"""Explicit base-near/tip-near planar-card pinhole tilt augmentation.

The source HRM is a flat textured card aligned to its ordered centerline's
base-to-tip axis. Tilting that CARD creates explicit longitudinal perspective
magnification/foreshortening, not a real articulated 3D HRM view. No new depth,
camera calibration, joint angles, or physical shape ground truth is produced.

API: (RGB uint8, visible uint8{0,1}, JSON metadata, opaque uint8{0,1}) for both
make_case_v3 and augment_v3. All altered-geometry cases forbid original-depth
reuse. Sources and V1/V2 implementations remain unchanged.
"""

from __future__ import annotations

import cv2
import numpy as np

import augment
from feasibility_data import ordered_centerline
from feasibility_augmentation_v2 import (_metadata, _occlude_stripes, _hard_negatives, _fill_occluder,
                                         augment_v2)


CONDITIONS_V3 = ("clean", "base_near25", "base_near45", "base_near60",
                 "tip_near25", "tip_near45", "tip_near60",
                 "bg_base_near45_multi3", "bg_tip_near45_multi3")
GEOMETRY_PRESERVING_CONDITIONS_V3 = ("clean",)


def _guided_stripes(rgb, mask, rng, count, fraction, guide_xy):
    """Grade opacity on the transformed SOURCE centerline, not a new medial axis.

    A near end can become wide enough that re-skeletonization creates corner
    branches unrelated to its semantic base. Projecting the original ordered
    centerline retains the source endpoints under the exact same homography.
    """
    guide = np.asarray(guide_xy, dtype=np.float64)
    original_arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(guide, axis=0), axis=1))]
    keep = np.r_[True, np.diff(original_arc) > 1e-6]
    guide, original_arc = guide[keep], original_arc[keep]
    length = float(original_arc[-1])
    if length < 8:
        raise ValueError("Projected source centerline too short for graded occlusion")
    arc = np.r_[np.arange(0., length, .5), length]
    points = np.column_stack([np.interp(arc, original_arc, guide[:, i]) for i in range(2)])
    h,w = mask.shape
    pixels = np.rint(points).astype(int)
    px,py = np.clip(pixels[:,0],0,w-1),np.clip(pixels[:,1],0,h-1)
    delta=np.diff(arc)
    yy,xx=np.mgrid[:h,:w]
    centers=(np.array([rng.uniform(.37,.63)]) if count==1 else
             np.linspace(.24,.76,count)+rng.uniform(-.025,.025,count))
    bounds=np.r_[.07,(centers[:-1]+centers[1:])/2,.93]*length
    union=np.zeros_like(mask,dtype=bool);result,target=rgb.copy(),mask.copy();details=[]
    protected=(arc<=.07*length)|(arc>=.93*length)
    for i,center_fraction in enumerate(centers):
        center_s=center_fraction*length
        center=np.array([np.interp(center_s,arc,points[:,j]) for j in range(2)])
        span=max(2.,.025*length)
        tangent=np.array([np.interp(center_s+span,arc,points[:,j])-np.interp(center_s-span,arc,points[:,j]) for j in range(2)])
        tangent/=max(float(np.linalg.norm(tangent)),1e-9)
        projection=np.abs((xx-center[0])*tangent[0]+(yy-center[1])*tangent[1])
        path_projection=np.abs((points-center)@tangent)
        forbidden=(arc<=bounds[i])|(arc>=bounds[i+1])
        lo,hi=0.,float(path_projection[forbidden].min())-.01
        if hi<=0:raise ValueError("Guided stripe cannot separate centerline arc windows")
        best=None
        for _ in range(20):
            width=(lo+hi)/2;occluder=(projection<=width)&~union
            covered=occluder[py,px]
            actual=float(np.sum(delta*(covered[:-1].astype(float)+covered[1:])/2)/length)
            item=(abs(actual-fraction/count),occluder,covered,actual,width)
            if best is None or item[0]<best[0]:best=item
            if actual<fraction/count:lo=width
            else:hi=width
        error,occluder,covered,actual,width=best
        if error>max(.015,2/length) or covered[protected].any() or not np.any(occluder&(mask!=0)):
            raise ValueError("Guided stripe grade/endpoint constraint could not be realized")
        fill,material=_fill_occluder(mask.shape,rng)
        result,target=augment.apply_occluder(result,target,occluder.astype(np.uint8),fill)
        union|=occluder
        details.append(dict(center_fraction=float(center_fraction),material=material,
                            width_pixels=float(2*width),actual_centerline_fraction=actual))
    covered=union[py,px]
    actual=float(np.sum(delta*(covered[:-1].astype(float)+covered[1:])/2)/length)
    removed=int(np.count_nonzero(union&(mask!=0)))
    info=dict(requested_centerline_occlusion_fraction=float(fraction),actual_centerline_occlusion_fraction=actual,
              foreground_occlusion_fraction=removed/max(int(mask.sum()),1),full_foreground_pixels=int(mask.sum()),
              visible_foreground_pixels=int(target.sum()),has_actual_occlusion=True,occluder_count=count,
              occluder_pixels=int(union.sum()),occluders=details,endpoints_covered=[bool(covered[0]),bool(covered[-1])],
              endpoint_regions_covered=[bool(covered[arc<=.07*length].any()),bool(covered[arc>=.93*length].any())],
              centerline_length_pixels=length,centerline_method="source_longest_skeleton_path_transformed_by_same_homography")
    return result,target,info,union.astype(np.uint8)


def _retry_stripes(rgb, mask, rng, count, fraction, guide_xy=None):
    """Resample placement, NEVER grade or labels, when a curved silhouette fails.

    Rejections are explicit provenance, not silent clean fallbacks. Retrying
    the same three stripes/total coverage avoids aborting a training epoch on
    rare local-tangent geometry; terminal failure remains a ValueError.
    """
    failures = []
    for attempt in range(1, 9):
        try:
            if guide_xy is None:
                image, target, meta, occluder = _occlude_stripes(rgb, mask, rng, count, fraction)
            else:
                image, target, meta, occluder = _guided_stripes(rgb, mask, rng, count, fraction, guide_xy)
            meta.update(occlusion_placement_attempts=attempt,
                        occlusion_placement_rejections=failures)
            return image, target, meta, occluder
        except ValueError as error:
            failures.append(str(error))
    raise ValueError(f"V3 stripe placement failed8attempts(count={count},fraction={fraction}): {failures}")


def _transform_points(points, transform):
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ transform.T
    if np.any(homogeneous[:, 2] <= 0) or not np.isfinite(homogeneous).all():
        raise ValueError("Planar-card projection crosses or reaches the virtual camera plane")
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def _planar_card(rgb, mask, rng, near_end, tilt_degrees, backgrounds, replace_background, strong):
    """Rotate a virtual source-aligned plane about its transverse axis.

    Local card coordinates are u across the body and v from base to tip, with
    origin at the two endpoints' midpoint. Virtual focal distance f=d=0.9*L.
    Projected coordinates are f*u/(d+s*v*sin(theta)),
    f*v*cos(theta)/(d+s*v*sin(theta)); s=+1 makes base near, s=-1 tip near.
    An output similarity transform then rotates/scales/places the entire card.
    """
    if near_end not in ("base", "tip") or not 0 < tilt_degrees < 75:
        raise ValueError("near_end must be base/tip and tilt must lie in (0,75) degrees")
    points, arc = ordered_centerline(mask)
    base, tip = points[0], points[-1]
    vector = tip - base
    length = float(np.linalg.norm(vector))
    if length < 16:
        raise ValueError("Base/tip separation under16px: virtual longitudinal card is not well defined")
    tangent = vector / length
    transverse = np.array([-tangent[1], tangent[0]])
    center = (base + tip) / 2.
    local = np.array([[transverse[0], transverse[1], -float(transverse @ center)],
                      [tangent[0], tangent[1], -float(tangent @ center)],
                      [0., 0., 1.]])
    theta = np.deg2rad(float(tilt_degrees))
    sign = 1. if near_end == "base" else -1.
    distance = focal = .9 * length
    tilt = np.array([[focal, 0., 0.], [0., focal * np.cos(theta), 0.],
                     [0., sign * np.sin(theta), distance]])
    # Preserve the source body's image-axis orientation until optional placement.
    basis = np.array([[transverse[0], tangent[0], 0.],
                      [transverse[1], tangent[1], 0.], [0., 0., 1.]])
    raw_transform = basis @ tilt @ local
    ys, xs = np.nonzero(mask)
    foreground_points = np.column_stack((xs, ys)).astype(np.float64)
    local_foreground = np.column_stack((foreground_points, np.ones(len(xs)))) @ local.T
    depths = distance + sign * local_foreground[:, 1] * np.sin(theta)
    if not np.isfinite(depths).all() or depths.min() <= .1 * distance:
        raise ValueError("Source silhouette is incompatible with positive virtual plane depth")
    warped_points = _transform_points(foreground_points, raw_transform)
    angle = float(rng.uniform(-180, 180)) if strong else 0.
    a = np.deg2rad(angle)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    rotated = warped_points @ rotation.T
    h, w = mask.shape
    extent = np.ptp(rotated, axis=0)
    if np.any(extent < 1):
        raise ValueError("Degenerate planar-card projected extent")
    requested_scale = float(rng.uniform(.65, 1.30)) if strong else 1.
    # Include a conservative interpolation safety margin around the full body.
    fit = min((w - 16) / extent[0], (h - 16) / extent[1])
    scale = min(requested_scale, fit)
    rotated *= scale
    low = np.array([8., 8.]) - rotated.min(0)
    high = np.array([w - 8., h - 8.]) - rotated.max(0)
    if np.any(high < low - 1e-7):
        raise ValueError("Full planar-card silhouette does not fit output canvas")
    high = np.maximum(high, low)
    translation = rng.uniform(low, high) if strong else np.clip(center, low, high)
    similarity = np.eye(3)
    similarity[:2, :2] = rotation * scale
    similarity[:2, 2] = translation
    transform = similarity @ raw_transform
    # Homography scale has no geometric meaning. Preserve positive camera-depth
    # denominators even when the image origin (outside the body) lies behind the
    # virtual plane; normalizing by a negative H33 would spuriously flip them.
    normalization = abs(float(transform[2, 2]))
    transform /= normalization if normalization > 1e-9 else np.linalg.norm(transform)
    target = cv2.warpPerspective(mask, transform, (w, h), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = cv2.warpPerspective(mask.astype(np.float32), transform, (w, h))
    premultiplied = cv2.warpPerspective(rgb.astype(np.float32) * mask[..., None], transform, (w, h))
    colors = premultiplied / np.maximum(alpha[..., None], 1e-6)
    background = (augment._background(mask.shape, list(backgrounds), rng) if replace_background else
                  cv2.inpaint(rgb, mask * 255, 5., cv2.INPAINT_TELEA))
    output = np.where(target[..., None] != 0, np.clip(colors, 0, 255), background).astype(np.uint8)
    if not target.any() or target[0].any() or target[-1].any() or target[:, 0].any() or target[:, -1].any():
        raise ValueError("Planar-card transform clipped or erased the foreground")
    transformed_endpoints = _transform_points(np.stack((base, tip)), transform)
    endpoint_depths = distance + sign * np.array([-length / 2, length / 2]) * np.sin(theta)
    endpoint_scales = scale * focal / endpoint_depths
    near_index = 0 if near_end == "base" else 1
    ratio = float(endpoint_scales[near_index] / endpoint_scales[1 - near_index])
    if ratio <= 1:
        raise ValueError("Near/far magnification sign inconsistency")
    meta = dict(geometry_changed=True, original_depth_geometry_compatible=False,
                geometry_claim="virtual planar textured-card pinhole tilt, NOT articulated 3D viewpoint/depth synthesis",
                source_to_output_homography=transform.tolist(),
                source_to_card_matrix=local.tolist(),
                card_pinhole_tilt_matrix=tilt.tolist(),
                output_similarity_matrix=similarity.tolist(),
                source_base_xy=base.tolist(), source_tip_xy=tip.tolist(),
                source_longitudinal_unit_xy=tangent.tolist(), source_transverse_unit_xy=transverse.tolist(),
                source_endpoint_distance_pixels=length,
                source_centerline_arc_length_pixels=float(arc[-1]),
                output_base_xy=transformed_endpoints[0].tolist(),
                output_tip_xy=transformed_endpoints[1].tolist(),
                near_end=near_end, tilt_degrees=float(tilt_degrees),
                virtual_focal_distance_pixels=focal,
                virtual_card_center_depth_pixels=distance,
                virtual_base_depth_pixels=float(endpoint_depths[0]),
                virtual_tip_depth_pixels=float(endpoint_depths[1]),
                base_transverse_magnification=float(endpoint_scales[0]),
                tip_transverse_magnification=float(endpoint_scales[1]),
                near_to_far_magnification_ratio=ratio,
                projected_endpoint_separation_pixels=float(np.linalg.norm(transformed_endpoints[1]-transformed_endpoints[0])),
                rotation_degrees=angle, requested_scale=requested_scale, applied_scale=float(scale),
                all_foreground_inside_frame=True, background_replaced=bool(replace_background),
                full_foreground_pixels=int(target.sum()),
                depth_policy="NO reuse of original depth/intrinsics after this homography")
    return output, target, meta


def make_case_v3(rgb, mask, seed, condition, backgrounds=()):
    """Deterministic condition with explicit near end and fixed virtual tilt."""
    augment._validate(rgb, mask)
    if condition not in CONDITIONS_V3:
        raise ValueError(f"Unknown V3 condition: {condition}")
    image, target, occ = rgb.copy(), mask.copy(), np.zeros_like(mask)
    meta = _metadata(mask, condition)
    meta.update(augmentation_family="v3_explicit_planar_card_tilt", seed=int(seed))
    if condition == "clean":
        return image, target, meta, occ
    near = "base" if "base_near" in condition else "tip"
    angle = 45 if condition.startswith("bg_") else int(condition[-2:])
    replace = condition.startswith("bg_")
    geometry_rng = np.random.default_rng(int(seed) ^ 0xB453712)
    occlusion_rng = np.random.default_rng(int(seed) ^ 0x0CC10DE)
    image, target, info = _planar_card(image, target, geometry_rng, near, angle, backgrounds,
                                      replace_background=replace, strong=replace)
    meta.update(info)
    if condition.endswith("multi3"):
        source_guide,_=ordered_centerline(mask)
        guide=_transform_points(source_guide,np.array(meta["source_to_output_homography"]))
        image, target, info, occ = _retry_stripes(image, target, occlusion_rng, 3, .30, guide)
        meta.update(info)
    meta["visible_foreground_pixels"] = int(target.sum())
    return np.ascontiguousarray(image), np.ascontiguousarray(target), meta, np.ascontiguousarray(occ)


def augment_v3(rgb, mask, rng, backgrounds=()):
    """Mix retained V2 training distribution with explicit longitudinal tilt.

    Forty percent reuse the frozen V2 pipeline; sixty percent get near-base or
    near-tip virtual tilt20..65deg, optional background/placement/distractors,
    single/thick/multiple opacity and appearance changes. Invalid geometry
    raises a descriptive ValueError; there is NO silent fallback to clean.
    """
    augment._validate(rgb, mask)
    if rng.random() < .40:
        image, target, meta, occ = augment_v2(rgb, mask, rng, backgrounds)
        meta["augmentation_family"] = "frozen_v2_replay_inside_v3_training"
        return image, target, meta, occ
    meta = _metadata(mask, "train_v3")
    meta["augmentation_family"] = "v3_explicit_planar_card_tilt"
    near = str(rng.choice(["base", "tip"]))
    angle = float(rng.uniform(20, 65))
    image, target, info = _planar_card(rgb, mask, rng, near, angle, backgrounds,
                                      replace_background=bool(rng.random() < .70),
                                      strong=bool(rng.random() < .75))
    meta.update(info)
    occ = np.zeros_like(mask)
    if rng.random() < .25:
        image, count = _hard_negatives(image, target, rng)
        meta["hard_negative_objects"] = count
    if rng.random() < .65:
        branch = int(rng.integers(3))
        count = 3 if branch == 2 else 1
        fraction = float(rng.uniform(.01, .045) if branch == 0 else rng.uniform(.08, .34))
        source_guide,_=ordered_centerline(mask)
        guide=_transform_points(source_guide,np.array(meta["source_to_output_homography"]))
        image, target, info, occ = _retry_stripes(image, target, rng, count, fraction, guide)
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
