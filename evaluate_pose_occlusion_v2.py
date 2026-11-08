#!/usr/bin/env python3
# AUTHOR: Daeyun - Paired visible-depth HRM joint-angle and FK sensitivity evaluation.
"""Paired occlusion sensitivity of image-derived q/FK, NOT ground-truth accuracy.

RGB-D geometry is unchanged. Synthetic opaque occluder pixels remove original
depth ONLY, never clip predicted masks. A clean OpenCV prediction is used solely
to place occluders. Each method's occluded shape is compared to that method's
clean shape on the SAME source frame, with its clean camera-space base origin
held fixed. This is a missing-depth simulation, not real-occluder rejection.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import zlib

os.environ.setdefault('MPLBACKEND', 'Agg')

import cv2
import numpy as np

from feasibility_backends import OpenCVBaseline


ROOT = Path(__file__).resolve().parent
ARCHIVE = (ROOT.parent / 'Hyper-Redundant-Manipulator/record/'
           '20260927_HRM_shape_estimation_apriltag/shape_estimation_AprilTag_45deg_1')
POSE_FIELDS = ('q_rmse_deg', 'q_p95_abs_deg', 'q_max_abs_deg',
               'fk_tip_delta_mm', 'fk_orientation_delta_deg', 'curve_tip_delta_mm')


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + '\n')


def spaced(records, count):
    if count == 0:
        return []
    if count < 0 or count >= len(records):
        return records
    return [records[index] for index in np.linspace(0, len(records) - 1, count, dtype=int)]


def load_live_records(capture):
    capture = Path(capture).resolve()
    manifest = json.loads((capture / 'manifest.json').read_text())
    records = []
    for frame in manifest['frames']:
        if int(frame['color_stamp_ns']) != int(frame['depth_stamp_ns']):
            raise ValueError('Live pose sensitivity requires exact RGB/depth source timestamps.')
        records.append({
            'dataset': 'live_black', 'frame_id': Path(frame['color_file']).stem,
            'source_stamp_ns': int(frame['color_stamp_ns']),
            'rgb': str(capture / frame['color_file']), 'depth': str(capture / frame['depth_file']),
            'crop_k': frame['crop_color_camera_info']['K'],
            'depth_scale_m': float(frame['depth_scale_m']),
            'frame_name': frame['frame_id'], 'calibration': frame['crop_color_camera_info'],
            'source_roi_xywh': frame['source_roi_xywh'], 'source_manifest': str(capture / 'manifest.json'),
        })
    return sorted(records, key=lambda record: record['source_stamp_ns'])


def load_archive_records(session):
    """Join RGB and depth by timestamp, NOT filename sequence counter.

    Archived45deg1 has one missing initial depth frame, so matching file sequence
    numbers would incorrectly shift every subsequent depth by one frame.
    """
    session = Path(session).resolve()
    rgb_dir, depth_dir = session / 'images/hrm_crop', session / 'images/hrm_crop_depth'
    with (rgb_dir / 'index.csv').open(newline='') as stream:
        colors = list(csv.DictReader(stream))
    with (depth_dir / 'index.csv').open(newline='') as stream:
        depths = list(csv.DictReader(stream))
    depth_by_stamp = {}
    for frame in depths:
        stamp = int(frame['source_time_ns'])
        if stamp in depth_by_stamp:
            raise ValueError('Duplicate archive depth timestamp.')
        depth_by_stamp[stamp] = frame
    records = []
    seen = set()
    for color in colors:
        stamp = int(color['source_time_ns'])
        if stamp in seen:
            raise ValueError('Duplicate archive RGB timestamp.')
        seen.add(stamp)
        if stamp not in depth_by_stamp:
            continue
        depth = depth_by_stamp[stamp]
        if (color['width'], color['height'], color['frame_id']) != (
                depth['width'], depth['height'], depth['frame_id']):
            raise ValueError('Archive RGB/depth resolution or frame mismatch.')
        records.append({
            'dataset': 'archive_45deg_1', 'frame_id': Path(color['filename']).stem,
            'source_stamp_ns': stamp, 'rgb': str(rgb_dir / color['filename']),
            'depth': str(depth_dir / depth['filename']),
            'crop_k': json.loads(depth['k']), 'depth_scale_m': float(depth['depth_scale_m_per_unit']),
            'frame_name': color['frame_id'],
            'calibration': {key: json.loads(depth[key]) for key in ('k', 'p', 'r', 'd')},
            'source_roi_xywh': [int(depth[key]) for key in ('roi_x', 'roi_y', 'roi_width', 'roi_height')],
            'source_manifest': str(session / 'session.json'),
        })
    return sorted(records, key=lambda record: record['source_stamp_ns']), {
        'source_rgb_frames': len(colors), 'source_depth_frames': len(depths),
        'exact_timestamp_pairs': len(records), 'rgb_without_depth': len(colors) - len(records),
        'join_rule': 'exact source_time_ns; file sequence counter is NOT used',
    }


def wrapped_joint_delta_deg(current, reference):
    current, reference = np.asarray(current, dtype=float), np.asarray(reference, dtype=float)
    if current.shape != (18,) or reference.shape != (18,):
        raise ValueError('Exactly18 ordered joint angles required.')
    if not np.isfinite(current).all() or not np.isfinite(reference).all():
        raise ValueError('Joint angles must be finite.')
    return np.rad2deg(np.arctan2(np.sin(current - reference), np.cos(current - reference)))


def rotation_geodesic_deg(current, reference):
    current, reference = np.asarray(current, dtype=float), np.asarray(reference, dtype=float)
    for matrix in (current, reference):
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(matrix), 1., atol=1e-5)):
            raise ValueError('Expected proper orthonormal3x3 rotation.')
    if np.allclose(current, reference, rtol=0., atol=1e-12):
        return 0.  # avoid reporting roundoff as a nonzero clean-vs-itself angle
    return float(np.rad2deg(np.arccos(np.clip((np.trace(reference.T @ current) - 1.) / 2., -1., 1.))))


def paired_pose_metrics(current, reference):
    delta = wrapped_joint_delta_deg(current['q_rad'], reference['q_rad'])
    metrics = {
        'q_rmse_deg': float(np.sqrt(np.mean(delta ** 2))),
        'q_p95_abs_deg': float(np.percentile(np.abs(delta), 95)),
        'q_max_abs_deg': float(np.max(np.abs(delta))),
        'fk_tip_delta_mm': float(np.linalg.norm(np.asarray(current['fk_tip_xyz_m']) - reference['fk_tip_xyz_m']) * 1000.),
        'fk_orientation_delta_deg': rotation_geodesic_deg(current['fk_tip_rotation'], reference['fk_tip_rotation']),
        'curve_tip_delta_mm': float(np.linalg.norm(np.asarray(current['curve_tip_xyz_m']) - reference['curve_tip_xyz_m']) * 1000.),
    }
    if not np.isfinite(list(metrics.values())).all():
        raise ValueError('Nonfinite paired pose sensitivity.')
    return metrics, delta


def aggregate_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['dataset'], row['method'], row['condition']].append(row)
    summary = []
    for (dataset, method, condition), group in groups.items():
        pair = [row for row in group if row['paired_valid']]
        result = {'dataset': dataset, 'method': method, 'condition': condition,
                  'requested_frames': len(group),
                  'generation_valid_frames': sum(row['generation_success'] for row in group),
                  'segmentation_call_success_frames': sum(row['segmentation_success'] for row in group),
                  'clean_shape_valid_frames': sum(row['clean_shape_success'] for row in group),
                  'occluded_shape_valid_frames': sum(row['shape_success'] for row in group),
                  'paired_valid_frames': len(pair),
                  'shape_output_fraction_all_requested': sum(row['shape_success'] for row in group) / len(group),
                  'paired_fraction_all_requested': len(pair) / len(group)}
        for field in POSE_FIELDS:
            values = [row[field] for row in pair]
            for statistic, function in (('mean', np.mean), ('median', np.median),
                                        ('p95', lambda x: np.percentile(x, 95)), ('max', np.max)):
                result[f'{field}_{statistic}'] = float(function(values)) if values else None
        summary.append(result)
    return summary


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def segmentation(backend, rgb, synchronize):
    synchronize()
    start = time.perf_counter()
    error = None
    try:
        mask = np.asarray(backend.predict_rgb(rgb))
        if mask.shape != rgb.shape[:2] or not np.isin(mask, [0, 1, 255]).all():
            raise ValueError('Invalid segmentation mask shape/values.')
        mask = (mask > 0).astype(np.uint8)
    except Exception as exc:
        mask = np.zeros(rgb.shape[:2], np.uint8)
        error = f'{type(exc).__name__}: {exc}'
    synchronize()
    return mask, (time.perf_counter() - start) * 1000., error


def shape_estimate(estimator, mask, depth, record, occmap=None, base=None):
    start = time.perf_counter()
    try:
        result = estimator.estimate(mask, depth, record['crop_k'],
                                    depth_scale_m=record['depth_scale_m'],
                                    occlusion_mask=occmap, base_origin_camera_xyz=base)
    except Exception as exc:
        result = {'success': False, 'status': 'exception', 'error': f'{type(exc).__name__}: {exc}'}
    return result, (time.perf_counter() - start) * 1000.


def occluded_shape_estimate(estimator, mask, depth, record, occmap, clean_reference):
    """No automatic rebasing when this method's clean calibration failed."""
    if not clean_reference.get('success'):
        return {'success': False, 'status': 'clean_base_unavailable',
                'error': 'Clean paired base origin unavailable; occluded shape not evaluated.'}, 0.
    return shape_estimate(estimator, mask, depth, record, occmap,
                          clean_reference['base_origin_camera_xyz'])


def export_arrays(path, result, reference=None):
    arrays = {key: value for key, value in result.items()
              if isinstance(value, np.ndarray) and value.dtype.kind in 'buifc'}
    if reference is not None:
        for key in ('q_rad', 'fk_tip_xyz_m', 'fk_tip_rotation', 'curve_tip_xyz_m',
                    'curve_xyz_m', 'base_origin_camera_xyz'):
            if key in reference:
                arrays[f'clean_{key}'] = np.asarray(reference[key])
    if arrays:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)


def pose_baseline_row(record, method, result):
    row = {'dataset': record['dataset'], 'source_stamp_ns': record['source_stamp_ns'],
           'frame_id': record['frame_id'], 'method': method,
           'shape_success': bool(result.get('success')), 'error': result.get('error')}
    for i in range(18):
        row[f'q{i + 1:02d}_rad'] = float(result['q_rad'][i]) if result.get('success') else None
    for key, count in (('fk_tip_xyz_m', 3), ('fk_tip_rotation', 9),
                       ('curve_tip_xyz_m', 3), ('base_origin_camera_xyz', 3)):
        values = np.asarray(result[key]).reshape(-1) if result.get('success') else [None] * count
        for i, value in enumerate(values):
            row[f'{key}_{i}'] = None if value is None else float(value)
    return row


def comparison_sheet(rgb, condition, method_results):
    from evaluate_live_feasibility import tile
    panels = [tile(rgb, f'Observed: {condition}', 'Paired CLEAN difference, NOT ground-truth error')]
    for method, mask, result, values in method_results:
        if result.get('success') and values:
            subtitle = f"dq RMS {values['q_rmse_deg']:.2f} deg | FK {values['fk_tip_delta_mm']:.2f} mm"
        elif result.get('success'):
            subtitle = 'Output valid; clean reference unavailable'
        else:
            subtitle = 'NO SHAPE: ' + str(result.get('error', 'unavailable'))[:38]
        panels.append(tile(rgb, method, subtitle, mask))
    while len(panels) % 2:
        panels.append(np.zeros_like(panels[0]))
    return np.vstack([np.hstack(panels[i:i + 2]) for i in range(0, len(panels), 2)])


def evaluate(args):
    import torch
    from feasibility_augmentation_v2 import GEOMETRY_PRESERVING_CONDITIONS_V2, make_case_v2
    from feasibility_shape import ShapeEstimator
    from run_feasibility import UNetBackend
    from evaluate_live_feasibility import write_rgb, write_mask

    conditions = args.conditions or list(GEOMETRY_PRESERVING_CONDITIONS_V2)
    if 'clean' not in conditions:
        conditions = ['clean', *conditions]
    if any(condition not in GEOMETRY_PRESERVING_CONDITIONS_V2 for condition in conditions):
        raise ValueError('RGB-D pose evaluation forbids perspective/affine image transforms.')
    archive, archive_stats = load_archive_records(args.archive)
    live = load_live_records(args.capture)
    records = spaced(live, args.live_limit) + spaced(archive, args.archive_limit)
    if not records:
        raise ValueError('No selected RGB-D frames.')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    backends = {'opencv': OpenCVBaseline(), 'unet_v1': UNetBackend(args.v1, args.device),
                'unet_v2': UNetBackend(args.v2, args.device)}
    # New live images and held-out archive frames must not enter fitting.
    for method, backend in backends.items():
        if hasattr(backend, 'checkpoint'):
            seen = {Path(path).name for path in backend.checkpoint.get('training_images', [])}
            overlap = seen & {Path(record['rgb']).name for record in records}
            if overlap:
                raise ValueError(f'{method} training/evaluation image overlap: {list(overlap)[:3]}')
    estimator = ShapeEstimator(minimum_path_points=args.minimum_path_points)
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    synchronize = torch.cuda.synchronize if args.device.startswith('cuda') else lambda: None
    warm = cv2.cvtColor(cv2.imread(records[0]['rgb']), cv2.COLOR_BGR2RGB)
    for backend in backends.values():
        segmentation(backend, warm, synchronize)
    dump(output / 'protocol.json', {
        'purpose': __doc__, 'archive_source': str(Path(args.archive).resolve()),
        'archive_pair_audit': archive_stats, 'capture': str(Path(args.capture).resolve()),
        'records': records, 'conditions': conditions, 'seed': args.seed,
        'methods': list(backends), 'v1_checkpoint': str(Path(args.v1).resolve()),
        'v2_checkpoint': str(Path(args.v2).resolve()),
        'shape_estimator_metadata': estimator.metadata(),
        'source_hashes': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ('evaluate_pose_occlusion_v2.py', 'feasibility_shape.py',
                                       'feasibility_augmentation_v2.py', 'feasibility_backends.py')},
        'synthetic_depth_rule': 'Known generated occluder map invalidates depth query and neighborhood support ONLY. Never clips model mask.',
        'base_rule': 'Per-method per-frame clean base origin reused for ALL occlusion variants. No rebase to surviving first point.',
        'sequence_rule': 'Independent paired frames; no Kalman, no temporal/base initialization averaging.',
        'no_ground_truth': True,
    })
    rows, joint_rows, baselines, cases = [], [], [], []
    dataset_counts = defaultdict(int)
    examples = {}
    for dataset in {record['dataset'] for record in records}:
        members = [record for record in records if record['dataset'] == dataset]
        examples[dataset] = {members[index]['frame_id'] for index in
                             np.linspace(0, len(members) - 1, min(3, len(members)), dtype=int)}
    for record in records:
        bgr = cv2.imread(record['rgb'])
        depth = np.load(record['depth']) if Path(record['depth']).suffix == '.npy' else cv2.imread(record['depth'], cv2.IMREAD_UNCHANGED)
        if bgr is None or depth is None or bgr.shape[:2] != depth.shape:
            raise ValueError(f'Invalid RGB-D pair {record["rgb"]}, {record["depth"]}')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        seed = args.seed + zlib.crc32(f"{record['dataset']}:{record['source_stamp_ns']}".encode())
        placement = OpenCVBaseline().predict_rgb(rgb)
        clean = {}
        for method, backend in backends.items():
            mask, seg_ms, seg_error = segmentation(backend, rgb, synchronize)
            result, shape_ms = shape_estimate(estimator, mask, depth, record)
            clean[method] = (mask, result, seg_ms, shape_ms, seg_error)
            baselines.append(pose_baseline_row(record, method, result))
        for condition in conditions:
            generation_error = None
            try:
                observed, _unused_target, meta, occmap = make_case_v2(rgb, placement, seed, condition)
                if observed.shape != rgb.shape or occmap.shape != depth.shape:
                    raise ValueError('Synthetic condition changed RGB-D pixel geometry.')
                meta['placement_reference'] = 'clean OpenCV prediction, not ground truth; discarded target never enters inference'
            except Exception as exc:
                generation_error = f'{type(exc).__name__}: {exc}'
                observed, occmap, meta = rgb, None, {'condition': condition, 'error': generation_error}
            rendered = []
            for method, backend in backends.items():
                clean_mask, reference, clean_seg_ms, clean_shape_ms, clean_seg_error = clean[method]
                if generation_error:
                    mask = np.zeros(depth.shape, np.uint8)
                    result = {'success': False, 'error': generation_error, 'status': 'generation_error'}
                    seg_ms, shape_ms, seg_error = 0., 0., None
                elif condition == 'clean':
                    mask, result, seg_ms, shape_ms, seg_error = clean[method]
                else:
                    mask, seg_ms, seg_error = segmentation(backend, observed, synchronize)
                    result, shape_ms = occluded_shape_estimate(estimator, mask, depth, record, occmap, reference)
                paired = bool(result.get('success') and reference.get('success') and not generation_error and not seg_error and not clean_seg_error)
                values = {field: None for field in POSE_FIELDS}
                delta = np.full(18, np.nan)
                metric_error = None
                if paired:
                    try:
                        values, delta = paired_pose_metrics(result, reference)
                    except Exception as exc:
                        paired = False
                        metric_error = f'{type(exc).__name__}: {exc}'
                stem = f"{record['dataset']}/{condition}/{method}/{record['frame_id']}"
                npz_relative = f'shape_arrays/{stem}.npz'
                export_arrays(output / npz_relative, result, reference)
                actual_fraction = meta.get('actual_centerline_occlusion_fraction')
                row = {'dataset': record['dataset'], 'source_stamp_ns': record['source_stamp_ns'],
                       'frame_id': record['frame_id'], 'method': method, 'condition': condition,
                       'generation_success': generation_error is None,
                       'segmentation_success': seg_error is None and generation_error is None,
                       'shape_success': bool(result.get('success')),
                       'clean_shape_success': bool(reference.get('success')),
                       'paired_valid': paired, **values,
                       'segmentation_latency_ms': seg_ms, 'shape_latency_ms': shape_ms,
                       'predicted_foreground_pixels': int(mask.sum()),
                       'placement_reference_actual_occlusion': actual_fraction,
                       'error': generation_error or seg_error or result.get('error') or metric_error,
                       'clean_error': clean_seg_error or reference.get('error'),
                       'shape_npz': npz_relative if (output / npz_relative).is_file() else None}
                rows.append(row)
                for joint in range(18):
                    joint_rows.append({'dataset': record['dataset'], 'source_stamp_ns': record['source_stamp_ns'],
                                       'frame_id': record['frame_id'], 'method': method, 'condition': condition,
                                       'joint_id': joint + 1, 'joint_axis_name': 'tilt' if joint % 2 == 0 else 'pan',
                                       'paired_valid': paired,
                                       'clean_q_rad': float(reference['q_rad'][joint]) if reference.get('success') else None,
                                       'occluded_q_rad': float(result['q_rad'][joint]) if result.get('success') else None,
                                       'wrapped_delta_deg': float(delta[joint]) if paired else None})
                cases.append({'dataset': record['dataset'], 'frame_id': record['frame_id'],
                              'method': method, 'condition': condition, 'augmentation': meta,
                              'shape_diagnostics': result.get('diagnostics', {}),
                              'shape_status': result.get('status'), 'error': row['error'],
                              'base_origin_camera_xyz': result.get('base_origin_camera_xyz'),
                              'paired_clean_base_origin_camera_xyz': reference.get('base_origin_camera_xyz')})
                rendered.append((method, mask, result, values if paired else None))
                if record['frame_id'] in examples[record['dataset']]:
                    write_mask(output / f'example_masks/{stem}.png', mask)
            if record['frame_id'] in examples[record['dataset']] and generation_error is None:
                write_rgb(output / f"examples/{record['dataset']}/{condition}/{record['frame_id']}.png",
                          comparison_sheet(observed, condition, rendered))
            # If interrupted later, preserve already-completed rows on every source frame.
        save_csv(output / 'per_frame.csv', rows)
        save_csv(output / 'per_joint_delta.csv', joint_rows)
        save_csv(output / 'baseline_pose.csv', baselines)
        dataset_counts[record['dataset']] += 1
        print(json.dumps({'source_done': dict(dataset_counts), 'source_stamp_ns': record['source_stamp_ns'],
                          'rows': len(rows)}), flush=True)
    summary = aggregate_rows(rows)
    save_csv(output / 'summary.csv', summary)
    dump(output / 'cases.json', cases)
    dump(output / 'results.json', {
        'purpose': __doc__, 'datasets': dict(dataset_counts), 'methods': list(backends),
        'conditions': conditions, 'summary': summary,
        'rows': len(rows), 'paired_valid_rows': sum(row['paired_valid'] for row in rows),
        'failure_rows': [row for row in rows if not row['paired_valid']],
        'caveats': [
            'All angle/pose metrics are paired deviations from the SAME method on unoccluded RGB-D, NOT ground-truth accuracy.',
            'A small difference can also mean the method is consistently wrong; independent GT is required for accuracy.',
            'Live straight stationary black-background and held-out-candidate archive dynamic poses reported separately.',
            'Historic exposure of archived evaluation session to earlier development is not known; not clinical/independent external validation.',
            'Base origin is supplied from clean paired image per method. Real operation must maintain/reliably detect this origin.',
            'Exact synthetic occluder map is oracle information used ONLY to simulate missing depth. Real valid occluder depth rejection not tested.',
            'No RGB-D perspective/homography transform, no occlusion clipping of predicted masks, no hidden depth reuse.',
            'Shared shape frontend differs from original largest-component 2D polynomial path; all methods use the SAME new frontend.',
            'No temporal filtering/base averaging; production polynomial/projection and kinematic convention reused.',
            'Output availability is algorithmic return availability, not proof of correct body reconstruction.',
            'Conditional metrics retain denominator counts; generation/segmentation/shape failures remain in all-requested rows.',
            'segmentation_success means a valid API return with binary input-resolution mask, NOT accuracy or a nonempty correct HRM mask.',
            'q1 tilt, q2 pan alternating18; radians preserved; wrapped angular differences reported in degrees.',
            'Curve point cloud lies on visible surface; center-axis radius/calibration error not corrected here.',
        ],
    })
    print(json.dumps({'output': str(output), 'rows': len(rows), 'paired_valid': sum(row['paired_valid'] for row in rows)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', default=str(ROOT / 'runs/occlusion_pose_20261024/live_capture_black'))
    parser.add_argument('--archive', default=str(ARCHIVE))
    parser.add_argument('--live-limit', type=int, default=40, help='0 skips live; negative selects all')
    parser.add_argument('--archive-limit', type=int, default=64, help='0 skips archive; negative selects all')
    parser.add_argument('--v1', default=str(ROOT / 'runs/feasibility_20261024/unet_robust/best.pt'))
    parser.add_argument('--v2', default=str(ROOT / 'runs/occlusion_pose_20261024/unet_v2/best.pt'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True)
    parser.add_argument('--conditions', nargs='+')
    parser.add_argument('--seed', type=int, default=61024)
    parser.add_argument('--minimum-path-points', type=int, default=8,
                        help='Minimum retained skeleton path length; original protocol uses8.')
    args = parser.parse_args()
    if args.minimum_path_points < 1:
        parser.error('--minimum-path-points must be at least1.')
    evaluate(args)


if __name__ == '__main__':
    main()
