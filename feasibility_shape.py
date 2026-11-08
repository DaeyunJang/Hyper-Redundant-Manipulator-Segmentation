# AUTHOR: Daeyun - Visible-fragment HRM reconstruction and paired angle/FK analysis.
# Lee skeletonization and numerical library algorithms retain their upstream attribution.
"""CPU-only visible RGB-D -> joint angles/FK diagnostic, no ROS/motor calls.

The production quartic reconstruction and joint-plane projection methods are
compiled directly from the current HRM postprocess.py AST, without executing
its imports or changing it. The C++ modified-DH FK is mirrored below and tested
against separately compiled original C++ method bodies.

This is NOT the unchanged production end-to-end pipeline. The offline front end
retains multiple visible components, uses Lee-skeleton graph paths and greedy
endpoint ordering, and omits the legacy 2D fit/extrapolation. Its purpose is a
paired clean/occluded sensitivity test with base and tip visible, not an amodal
shape ground truth. Across missing pixels, the production curve parameter uses
straight 3D chord distances between surviving observations: missing arc length
is approximated, NOT measured. There are no inserted hidden XYZ observations.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import heapq
import json
from pathlib import Path
import re

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from skimage.morphology import skeletonize

from feasibility_depth_preview import deproject_crop_pixels, mask_aware_depth_samples


HRM_ROOT = Path(__file__).resolve().parent.parent / "Hyper-Redundant-Manipulator"
DEFAULT_CONFIG = HRM_ROOT / "src/estimation_pkg/estimation_pkg/config.json"
POSTPROCESS_SOURCE = HRM_ROOT / "src/estimation_pkg/estimation_pkg/postprocess.py"
FK_SOURCE = HRM_ROOT / "src/robot_control_pkg/src/surgical_tool.cpp"
HW_SOURCE = HRM_ROOT / "src/robot_control_pkg/include/robot_control_pkg/hw_definition.hpp"
PRODUCTION_METHODS = (
    "fit_quartic_through_origin", "wrap_angle", "update_joint_kalman",
    "project_segment_directions_to_joint_planes", "reconstruct_3d_segments",
)


def production_core_class(source=POSTPROCESS_SOURCE):
    """Read-only extraction; no ROS/config lookups or original module imports."""
    source = Path(source)
    tree = ast.parse(source.read_text())
    rbsc = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RBSC")
    selected = [node for node in rbsc.body
                if isinstance(node, ast.FunctionDef) and node.name in PRODUCTION_METHODS]
    if {node.name for node in selected} != set(PRODUCTION_METHODS):
        raise RuntimeError("Production reconstruction API changed; re-audit before using this diagnostic.")
    rbsc.body = selected
    module = ast.Module(body=[rbsc], type_ignores=[])
    namespace = {"np": np, "Rotation": Rotation}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["RBSC"]


def _hardware_consistent(config):
    source = HW_SOURCE.read_text()
    mapping = {
        "NUM_OF_SEGMENTS": "num_of_segments", "NUM_OF_BENDING_JOINTS": "num_of_bending_joints",
        "NUM_OF_JOINT_PAIRS": "num_of_joint_pairs", "NUM_OF_PAN_JOINTS": "num_of_pan_joints",
        "NUM_OF_TILT_JOINTS": "num_of_tilt_joints", "PROXIMAL_OFFSET_LENGTH": "proximal_offset_length",
        "BENDING_JOINT_SPACING": "bending_joint_spacing", "DISTAL_OFFSET_LENGTH": "distal_offset_length",
    }
    for macro, key in mapping.items():
        match = re.search(r"^#define\s+" + macro + r"\s+([\d.]+)", source, re.MULTILINE)
        if match is None or not np.isclose(float(match[1]), float(config[key]), rtol=0, atol=1e-12):
            raise ValueError(f"Estimator config and robot-control hardware disagree: {key}/{macro}")


def modified_dh_transform(q, alpha, previous_length_m):
    """Mirror SurgicalTool::computeTransformationMatrix: Rx(alpha) Tx(l) Rz(q)."""
    cq, sq, ca, sa = np.cos(q), np.sin(q), np.cos(alpha), np.sin(alpha)
    return np.array([[cq, -sq, 0, previous_length_m],
                     [ca*sq, ca*cq, -sa, 0], [sa*sq, sa*cq, ca, 0], [0, 0, 0, 1]], float)


def forward_kinematics(q_rad, config):
    """q1 tilt about Base +Z, q2 pan; returns 18 frames and final tip 4x4."""
    q = np.asarray(q_rad, dtype=float)
    count = int(config["num_of_bending_joints"])
    if q.shape != (count,) or not np.isfinite(q).all():
        raise ValueError(f"Expected {count} finite relative joint angles in radians.")
    cumulative, frames = np.eye(4), []
    for index, angle in enumerate(q):
        alpha = 0. if index == 0 else (np.pi/2 if index % 2 else -np.pi/2)
        length = float(config["proximal_offset_length" if index == 0 else "bending_joint_spacing"]) * .001
        cumulative = cumulative @ modified_dh_transform(angle, alpha, length)
        frames.append(cumulative.copy())
    distal = np.eye(4)
    distal[0, 3] = float(config["distal_offset_length"]) * .001
    return np.asarray(frames), cumulative @ distal


def _longest_path(skeleton):
    """Weighted double-sweep path of one 8-connected Lee skeleton component."""
    yx = np.argwhere(skeleton)
    lookup = {tuple(pixel): index for index, pixel in enumerate(yx)}
    edges = [[] for _ in yx]
    for index, (y, x) in enumerate(yx):
        for dy, dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            neighbor = lookup.get((y+dy, x+dx))
            if neighbor is not None:
                edges[index].append((neighbor, float(np.hypot(dy, dx))))

    def sweep(start):
        distance = np.full(len(yx), np.inf)
        previous = np.full(len(yx), -1, int)
        distance[start] = 0.
        queue = [(0., start)]
        while queue:
            cost, index = heapq.heappop(queue)
            if cost > distance[index]:
                continue
            for neighbor, length in edges[index]:
                if cost+length < distance[neighbor]:
                    distance[neighbor], previous[neighbor] = cost+length, index
                    heapq.heappush(queue, (cost+length, neighbor))
        if not np.isfinite(distance).all():
            raise ValueError("Unexpected disconnected graph within a skeleton component.")
        return int(np.argmax(distance)), previous

    if len(yx) == 0:
        raise ValueError("Skeleton component has no observed pixel.")
    if len(yx) == 1:
        # A visible fragment may thin to one measured pixel. Retain that
        # observation when the caller explicitly permits short paths; do not
        # manufacture an endpoint, direction, or hidden connecting pixel.
        return yx.astype(np.int32)
    start, _ = sweep(int(np.argmax(yx[:, 0])))
    end, previous = sweep(start)
    path = [end]
    while path[-1] != start:
        path.append(int(previous[path[-1]]))
    return yx[path].astype(np.int32)


def order_visible_skeleton(mask, minimum_component_area=16, minimum_path_points=8):
    """Order visible paths without consulting clean/reference masks or depth.

    Start at the lowest-image endpoint (largest v), then greedily connect the
    nearest remaining component endpoint. This assumes an unbranched body with
    a visible distal end and no self-overlap; it is not a general amodal tracker.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2 or min(mask.shape) == 0 or not np.isin(mask, [0,1,255]).all():
        raise ValueError("Expected a same-resolution binary predicted body mask.")
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    paths, component_ids, discarded = [], [], []
    skeleton_image = np.zeros(mask.shape, np.uint8)
    skeleton_pixel_count = 0
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < minimum_component_area:
            discarded.append({"component": component, "reason": "small_area", "pixels": area})
            continue
        skel = skeletonize(labels == component, method="lee").astype(np.uint8)
        skeleton_pixel_count += int(skel.sum())
        # Lee thinning can exceptionally produce multiple pieces; retain each.
        skel_count, skel_labels = cv2.connectedComponents(skel, connectivity=8)
        for skel_component in range(1, skel_count):
            piece = skel_labels == skel_component
            if int(piece.sum()) < minimum_path_points:
                discarded.append({"component": component, "reason": "short_skeleton", "pixels": int(piece.sum())})
                continue
            path = _longest_path(piece)
            if len(path) < minimum_path_points:
                discarded.append({"component": component, "reason": "short_path", "pixels": len(path)})
                continue
            paths.append(path)
            component_ids.append(component)
            skeleton_image[path[:, 0], path[:, 1]] = 1
    if not paths:
        raise ValueError("No sufficiently large visible skeleton component remains.")
    choices = [(int(path[end, 0]), -int(path[end, 1]), index, end)
               for index, path in enumerate(paths) for end in (0, -1)]
    _, _, first, endpoint = max(choices)
    current = paths[first] if endpoint == 0 else paths[first][::-1]
    remaining = set(range(len(paths))) - {first}
    ordered, ids = [current], [np.full(len(current), component_ids[first], int)]
    gap_pixels = []
    while remaining:
        candidates = [(float(np.linalg.norm(paths[index][end]-current[-1])), index, end)
                      for index in sorted(remaining) for end in (0, -1)]
        gap, index, endpoint = min(candidates)
        current = paths[index] if endpoint == 0 else paths[index][::-1]
        gap_pixels.append(gap)
        ordered.append(current)
        ids.append(np.full(len(current), component_ids[index], int))
        remaining.remove(index)
    combined = np.concatenate(ordered)
    return combined, np.concatenate(ids), skeleton_image, dict(
        predicted_component_count=count-1, retained_path_count=len(paths),
        retained_path_pixel_counts=[len(path) for path in paths],
        minimum_path_points=minimum_path_points,
        discarded_components=discarded, intercomponent_gaps_pixels=gap_pixels,
        skeleton_pixels_before_branch_pruning=skeleton_pixel_count,
        ordered_path_pixels=len(combined),
        ordering="lowest-image-v endpoint, greedy nearest remaining component endpoint",
        branch_policy="longest weighted 8-connected path; cycles handled by approximate double sweep")


class ShapeEstimator:
    """Per-call independent shape state; no base averaging or Kalman history."""

    def __init__(self, config_path=DEFAULT_CONFIG, config=None, minimum_component_area=16,
                 minimum_path_points=8, endpoint_max_skipped=5, max_step_hardware_ratio=1.5,
                 max_base_origin_error_hardware_ratio=0.10):
        self.config = copy.deepcopy(config if config is not None else json.loads(Path(config_path).read_text()))
        _hardware_consistent(self.config)
        filters = self.config.setdefault("filters", {})
        filters.setdefault("joint_kalman", {})["enabled"] = False
        filters.setdefault("base_origin_initial_average", {})["enabled"] = False
        self.core_class = production_core_class()
        self.minimum_component_area = int(minimum_component_area)
        self.minimum_path_points = int(minimum_path_points)
        self.endpoint_max_skipped = int(endpoint_max_skipped)
        self.max_step_hardware_ratio = float(max_step_hardware_ratio)
        self.max_base_origin_error_hardware_ratio = float(max_base_origin_error_hardware_ratio)
        if min(self.minimum_component_area, self.minimum_path_points) < 1 or self.endpoint_max_skipped < 0:
            raise ValueError("Invalid component/path/endpoint settings.")
        if not np.isfinite(self.max_step_hardware_ratio) or self.max_step_hardware_ratio <= 0:
            raise ValueError("max_step_hardware_ratio must be positive finite.")
        if (not np.isfinite(self.max_base_origin_error_hardware_ratio)
                or self.max_base_origin_error_hardware_ratio <= 0):
            raise ValueError("max_base_origin_error_hardware_ratio must be positive finite.")
        self.rotation_camera_from_base = (
            Rotation.from_euler("y", self.config["base_in_camera_rotation_y_deg"], degrees=True)
            * Rotation.from_euler("z", self.config["base_in_camera_rotation_z_deg"], degrees=True))

    def metadata(self):
        return dict(
            production_methods=list(PRODUCTION_METHODS),
            production_source_sha256=hashlib.sha256(POSTPROCESS_SOURCE.read_bytes()).hexdigest(),
            fk_cpp_source_sha256=hashlib.sha256(FK_SOURCE.read_bytes()).hexdigest(),
            hardware_header_sha256=hashlib.sha256(HW_SOURCE.read_bytes()).hexdigest(),
            config=copy.deepcopy(self.config),
            minimum_component_area=self.minimum_component_area,
            minimum_path_points=self.minimum_path_points,
            endpoint_max_skipped=self.endpoint_max_skipped,
            max_single_observation_step_in_hardware_lengths=self.max_step_hardware_ratio,
            max_first_point_distance_from_fixed_base_in_hardware_lengths=self.max_base_origin_error_hardware_ratio,
            filters="joint Kalman and initial-base average OFF; every call independent",
            production_identical="3D quartic+origin constraint+arc scaling+19 chords+18 sequential DH projections",
            production_frontend_differences="multi-component Lee path ordering; mask-aware depth; no 2D fit/extrapolation",
            missing_geometry="surviving observations only; gap parameter distance approximated by endpoint 3D chord",
            known_synthetic_occlusion="depth availability only; does not clip segmentation prediction or skeleton",
            assumptions=["base and tip visible", "base is lowest-image-v endpoint",
                         "unbranched non-self-overlapping body", "pinhole crop-local K",
                         "source pixels/depth already aligned", "no independent shape ground truth"],
        )

    def estimate(self, mask, depth, crop_k, depth_scale_m=.001, occlusion_mask=None,
                 base_origin_camera_xyz=None):
        diagnostics = {}
        try:
            pixels, ids, skeleton, ordering = order_visible_skeleton(
                mask, self.minimum_component_area, self.minimum_path_points)
            diagnostics.update(ordering)
            predicted_mask = np.asarray(mask) > 0
            if np.shape(depth) != predicted_mask.shape:
                raise ValueError("Depth/mask shape mismatch.")
            if occlusion_mask is None:
                unavailable = np.zeros(predicted_mask.shape, bool)
            else:
                unavailable = np.asarray(occlusion_mask)
                if unavailable.shape != predicted_mask.shape or not np.isin(unavailable, [0,1,255]).all():
                    raise ValueError("Occlusion availability map must be same-size binary.")
                unavailable = unavailable > 0
            # Do NOT edit the predicted mask/skeleton above. This map declares
            # depth unavailable at synthetic occluder pixels. It is never used
            # to supply hidden XYZ, clean-mask geometry, or segmentation scores.
            support_mask = predicted_mask & ~unavailable
            depth_config = self.config.get("filters", {}).get("depth_neighborhood_median", {})
            kernel = int(depth_config.get("kernel_size", 5))
            minimum = int(depth_config.get("min_valid_samples", 3))
            if not depth_config.get("enabled", True):
                kernel, minimum = 1, 1
            observed_yx, depths, neighbor_counts, valid = mask_aware_depth_samples(
                depth, support_mask, pixels, depth_scale_m, kernel, minimum)
            diagnostics.update(observed_point_count=len(observed_yx), depth_candidate_count=len(pixels),
                               known_occlusion_pixels=int(unavailable.sum()),
                               query_pixels_hidden_by_occlusion=int(unavailable[pixels[:,0], pixels[:,1]].sum()),
                               depth_kernel=kernel, minimum_valid_depth_neighbors=minimum)
            if len(observed_yx) < 5:
                raise ValueError("At least five visible, valid depth observations are required.")
            retained = np.flatnonzero(valid)
            prefix, suffix = int(retained[0]), int(len(pixels)-1-retained[-1])
            diagnostics.update(skipped_base_endpoint_pixels=prefix, skipped_tip_endpoint_pixels=suffix)
            if max(prefix, suffix) > self.endpoint_max_skipped:
                raise ValueError("Insufficient depth at an endpoint; base/tip-visible assumption failed.")
            camera_points = deproject_crop_pixels(observed_yx, depths, crop_k)
            steps = np.linalg.norm(np.diff(camera_points, axis=0), axis=1)
            hardware_length = (float(self.config["proximal_offset_length"])
                               + (int(self.config["num_of_bending_joints"])-1)*float(self.config["bending_joint_spacing"])
                               + float(self.config["distal_offset_length"]))* .001
            diagnostics.update(hardware_length_m=hardware_length, max_observed_step_m=float(steps.max()))
            if np.any(steps > hardware_length*self.max_step_hardware_ratio):
                raise ValueError("An observed 3D step exceeds the declared whole-body length gate; possible background/depth jump.")
            if base_origin_camera_xyz is None:
                origin = camera_points[0].copy()
                diagnostics["base_origin_policy"] = "first valid observed endpoint in this frame"
            else:
                origin = np.asarray(base_origin_camera_xyz, dtype=float)
                if origin.shape != (3,) or not np.isfinite(origin).all():
                    raise ValueError("base_origin_camera_xyz must contain three finite coordinates.")
                diagnostics["base_origin_policy"] = "caller supplied fixed calibration origin, no clean shape points used"
                base_distance = float(np.linalg.norm(camera_points[0]-origin))
                diagnostics.update(first_observed_point_distance_from_fixed_base_m=base_distance,
                                   fixed_base_distance_limit_m=hardware_length*self.max_base_origin_error_hardware_ratio)
                if base_distance > hardware_length*self.max_base_origin_error_hardware_ratio:
                    raise ValueError("First observed point is too far from fixed base calibration; base component may be missing.")
            base_points = self.rotation_camera_from_base.inv().apply(camera_points-origin)
            diagnostics["tip_visibility_independently_verified"] = False
            core = self.core_class()
            core.config = copy.deepcopy(self.config)
            core.reconstruct_3d_segments(base_points, timestamp_sec=None)
            q = core.dh_joint_angles_projected_rad.copy()
            frames, tip = forward_kinematics(q, self.config)
            if not np.allclose(tip[:3,:3], core.dh_joint_frame_rotations[-1], atol=1e-10, rtol=0):
                raise ValueError("Estimator projected orientation and FK orientation disagree.")
            diagnostics.update(
                fitted_length_before_scaling_m=float(core.fitted_curve_length_before_hardware_scaling_3d),
                hardware_length_scale=float(core.hardware_length_scale),
                observed_polyline_length_m=float(core.raw_curve_length_3d),
                reconstructed_curve_length_m=float(core.curve_length_3d),
                maximum_projection_residual=float(np.max(np.abs(core.segment_projection_residuals))),
                gap_parameter_policy="direct 3D chord between consecutive surviving observations",
                kalman_enabled=False, hidden_observations_inserted=0,
                first_point_before_production_origin_constraint_m=base_points[0].tolist())
            return dict(success=True, status="ok", error=None, diagnostics=diagnostics,
                        q_rad=q, pan_relative_rad=core.pan_relative_rad.copy(),
                        tilt_relative_rad=core.tilt_relative_rad.copy(),
                        fk_tip_xyz_m=tip[:3,3].copy(), fk_tip_rotation=tip[:3,:3].copy(),
                        fk_tip_quaternion_xyzw=Rotation.from_matrix(tip[:3,:3]).as_quat(),
                        fk_joint_transforms=frames,
                        curve_tip_xyz_m=core.segment_points_xyz[-1].copy(),
                        curve_xyz_m=core.curve_dense_xyz.copy(),
                        boundary_xyz_m=core.segment_points_xyz.copy(),
                        center_xyz_m=core.segment_center_points_xyz.copy(),
                        segment_directions_xyz=core.segment_directions_xyz.copy(),
                        projected_segment_directions_xyz=core.segment_directions_projected_xyz.copy(),
                        base_origin_camera_xyz=origin.copy(), observed_camera_xyz_m=camera_points,
                        observed_base_xyz_m=base_points, observed_pixels_yx=observed_yx,
                        observed_component_ids=ids[valid], depth_neighbor_counts=neighbor_counts,
                        raw_parameter_s=core.raw_s_3d.copy(), skeleton=skeleton)
        except Exception as exc:
            return dict(success=False, status="failed", error=f"{type(exc).__name__}: {exc}",
                        diagnostics=diagnostics)
