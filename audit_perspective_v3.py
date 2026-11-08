"""Independent, CPU-only audit of saved perspective evaluation PNGs.

This does not import the training/evaluation runner or its metric function.
Binary metrics use NumPy/SciPy rather than the runner's OpenCV implementation.
Only saved RGB/mask assets and provenance are read: no camera depth or models.
Agreement with visible pseudo-labels is not independent anatomical accuracy.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage


def recompute_metrics(prediction, target):
    p, t = prediction.astype(bool), target.astype(bool)
    tp = int(np.count_nonzero(p & t))
    fp = int(np.count_nonzero(p & ~t))
    fn = int(np.count_nonzero(~p & t))

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else 1.

    labels, count = ndimage.label(t, structure=np.ones((3, 3), bool))
    recalls = []
    for component in range(1, count + 1):
        component_mask = labels == component
        area = int(np.count_nonzero(component_mask))
        if area >= 32:
            recalls.append(float(np.count_nonzero(p & component_mask) / area))
    boundary_p = p & ~ndimage.binary_erosion(p, structure=np.ones((3, 3)), border_value=0)
    boundary_t = t & ~ndimage.binary_erosion(t, structure=np.ones((3, 3)), border_value=0)
    nearby_p = ndimage.maximum_filter(boundary_p, size=5, mode='constant', cval=0)
    nearby_t = ndimage.maximum_filter(boundary_t, size=5, mode='constant', cval=0)
    bp = ratio(np.count_nonzero(boundary_p & nearby_t), np.count_nonzero(boundary_p))
    br = ratio(np.count_nonzero(boundary_t & nearby_p), np.count_nonzero(boundary_t))
    return dict(tp=tp, fp=fp, fn=fn,
                predicted_pixels=int(np.count_nonzero(p)), target_pixels=int(np.count_nonzero(t)),
                dice=ratio(2 * tp, 2 * tp + fp + fn), iou=ratio(tp, tp + fp + fn),
                precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                boundary_f1_2px=2 * bp * br / (bp + br) if bp + br else 0.,
                visible_components=len(recalls), minimum_component_recall=min(recalls) if recalls else 1.,
                all_visible_components_recall_ge_80=bool(recalls and min(recalls) >= .8))


def audit(evaluation):
    root = Path(evaluation).resolve()
    issues, issue_count = [], 0
    checked_metrics, checked_predictions, checked_sources = 0, 0, set()
    maximum_difference = 0.

    def check(condition, message):
        nonlocal issue_count
        if not condition:
            issue_count += 1
            if len(issues) < 100:
                issues.append(message)

    def asset(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'Asset escapes evaluation folder: {relative}')
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def binary(relative):
        path = asset(relative)
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError(f'Expected uint8 grayscale mask: {path}')
        check(bool(np.isin(image, [0, 255]).all()), f'Nonbinary PNG: {relative}')
        return image != 0

    hashes = {}
    def digest(path):
        path = Path(path)
        if path not in hashes:
            hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        return hashes[path]

    cases = json.loads((root / 'cases.json').read_text())
    summary_json = json.loads((root / 'summary.json').read_text())
    with (root / 'per_frame.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    with (root / 'summary.csv').open(newline='') as stream:
        summaries = list(csv.DictReader(stream))
    indexed_rows = {}
    for row in rows:
        key = (row['condition'], row['source'], int(row['frame_timestamp_ns']), row['method'])
        check(key not in indexed_rows, f'Duplicate CSV row: {key}')
        indexed_rows[key] = row
    recomputed, matched_keys, case_keys = [], set(), set()
    source_sets, method_sets = defaultdict(set), []
    for record in cases:
        source, meta = record['source'], record['metadata']
        condition, name, stamp = meta['condition'], Path(source['rgb']).name, int(source['frame_timestamp_ns'])
        case_key = (condition, name, stamp)
        check(case_key not in case_keys, f'Duplicate case: {case_key}')
        case_keys.add(case_key)
        source_sets[condition].add((name, stamp))
        check(record['original_depth_reused'] is False, f'Original depth reused: {case_key}')
        if meta.get('geometry_changed'):
            check(meta.get('original_depth_geometry_compatible') is False,
                  f'Changed geometry incorrectly permits original depth: {case_key}')
        for kind in ('rgb', 'mask'):
            check(digest(source[kind]) == record[f'source_{kind}_sha256'],
                  f'Source {kind} hash mismatch: {case_key}')
            checked_sources.add(source[kind])
        rgb_path = asset(record['rgb_path'])
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
        target, occluder = binary(record['mask_path']), binary(record['occluder_path'])
        check(rgb is not None and rgb.dtype == np.uint8 and rgb.shape == (*target.shape, 3),
              f'RGB shape/type mismatch: {case_key}')
        check(occluder.shape == target.shape, f'Occluder shape mismatch: {case_key}')
        check(hashlib.sha256(occluder.astype(np.uint8).tobytes()).hexdigest() == record['occluder_sha256'],
              f'Occluder raw-pixel hash mismatch: {case_key}')
        check(json.loads((rgb_path.parent / 'provenance.json').read_text()) == record,
              f'Per-asset/global provenance mismatch: {case_key}')
        method_sets.append(set(record['prediction_paths']))
        for method, prediction_path in record['prediction_paths'].items():
            key = (*case_key, method)
            check(key in indexed_rows, f'Missing per-frame CSV row: {key}')
            if key not in indexed_rows:
                continue
            row = indexed_rows[key]
            matched_keys.add(key)
            prediction = binary(prediction_path)
            if prediction.shape != target.shape:
                raise ValueError(f'Prediction shape mismatch: {key}')
            result = recompute_metrics(prediction, target)
            checked_predictions += 1
            for metric, expected in result.items():
                if isinstance(expected, bool):
                    matches = row[metric].lower() == str(expected).lower()
                else:
                    difference = abs(float(row[metric]) - expected)
                    maximum_difference = max(maximum_difference, difference)
                    matches = bool(np.isclose(float(row[metric]), expected, rtol=1e-10, atol=1e-12))
                check(matches, f'Metric mismatch: {key}, {metric}: CSV={row[metric]}, pixels={expected}')
                checked_metrics += 1
            if row['status'] != 'ok':
                check(not prediction.any(), f'Failed inference did not save empty fallback: {key}')
            recomputed.append(dict(method=method, condition=condition, status=row['status'],
                                   latency_ms=float(row['latency_ms']), **result))
    check(matched_keys == set(indexed_rows), 'CSV contains unmatched/missing case rows')
    check(bool(method_sets) and all(value == method_sets[0] for value in method_sets),
          'Cases have unequal model sets')
    sets = list(source_sets.values())
    check(bool(sets) and all(value == sets[0] for value in sets), 'Conditions have unequal source sets')
    check(bool(sets) and len(sets[0]) == summary_json['records'], 'Summary source count mismatch')

    groups = defaultdict(list)
    for row in recomputed:
        groups[row['method'], row['condition']].append(row)
    summary_keys = set()
    json_summaries = {(row['method'], row['condition']): row for row in summary_json['results']}
    check(len(json_summaries) == len(summary_json['results']), 'Duplicate JSON summaries')
    for saved in summaries:
        key = (saved['method'], saved['condition'])
        check(key not in summary_keys, f'Duplicate CSV summary: {key}')
        summary_keys.add(key)
        group = groups.get(key, [])
        check(bool(group), f'Summary group has no predictions: {key}')
        if not group:
            continue
        expected = dict(frames=len(group), errors=sum(row['status'] != 'ok' for row in group),
                        latency_median_ms=float(np.median([row['latency_ms'] for row in group])))
        for metric in ('dice', 'iou', 'boundary_f1_2px', 'minimum_component_recall'):
            values = [row[metric] for row in group]
            expected[metric + '_mean'] = float(np.mean(values))
            expected[metric + '_p10'] = float(np.percentile(values, 10))
        for field, value in expected.items():
            check(np.isclose(float(saved[field]), value, rtol=1e-10, atol=1e-12),
                  f'CSV summary mismatch: {key}, {field}')
            check(key in json_summaries and np.isclose(float(json_summaries.get(key, {}).get(field, np.nan)),
                                                     value, rtol=1e-10, atol=1e-12),
                  f'JSON summary mismatch: {key}, {field}')
    check(summary_keys == set(groups) == set(json_summaries), 'Summary group sets differ')
    return dict(passed=issue_count == 0, evaluation=str(root), cases=len(cases), csv_rows=len(rows),
                checked_prediction_pngs=checked_predictions, checked_metrics=checked_metrics,
                checked_unique_source_files=len(checked_sources), conditions=len(source_sets),
                maximum_absolute_metric_difference=maximum_difference, issue_count=issue_count,
                issues_first_100=issues,
                audit_method='Independent NumPy/SciPy recomputation from all prediction/target PNGs; no model execution',
                limits='Checks artifact integrity and pseudo-label metric arithmetic, not real-world anatomical accuracy or physical-view generalization')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--output', required=True, help='New JSON audit path; never overwrite an existing report')
    args = parser.parse_args()
    report = audit(args.evaluation)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
