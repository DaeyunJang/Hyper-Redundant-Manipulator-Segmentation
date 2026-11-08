#!/usr/bin/env python3
"""Qualitative RGB-only inference on a finite fresh camera capture.

There is NO ground-truth mask in this capture. No Dice, IoU, accuracy, recall,
or downstream 3D performance is computed. OpenCV's clean predicted mask is
used solely to place synthetic occluders, never as a target or SAM prompt.
The synthetic images must not be paired with unchanged depth for 3D replay.
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

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(__file__).resolve().parent / '.ultralytics'))

import cv2
import numpy as np

from feasibility_backends import OpenCVBaseline
from feasibility_data import make_test_case


CONDITIONS = ('clean', 'mid_occlusion_10', 'mid_occlusion_20')
DISPLAY_NAMES = {
    'opencv': 'OpenCV current',
    'unet_basic': 'U-Net basic augmentation',
    'unet_robust': 'U-Net robust augmentation',
    'mobile_sam_roi': 'MobileSAM fixed ROI',
    'unet_robust_mobile_sam': 'U-Net robust -> MobileSAM',
}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def write_rgb(path, rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f'Could not write {path}')


def write_mask(path, mask):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), (np.asarray(mask) > 0).astype(np.uint8) * 255):
        raise OSError(f'Could not write {path}')


def build_cases(rgb, seed, conditions=CONDITIONS):
    """Yield observed RGB plus generation provenance, without output targets.

    Reference is solely a clean OpenCV prediction for occluder placement. In
    particular, neither reference nor visible pseudo-mask is passed to SAM.
    Generation errors are explicit and never silently turned into clean cases.
    """
    placement = OpenCVBaseline().predict_rgb(rgb)
    for condition in conditions:
        if condition == 'clean':
            yield condition, rgb.copy(), {
                'condition': condition, 'synthetic_occlusion': False,
                'has_ground_truth': False, 'rgb_only': True,
                'requested_centerline_occlusion_fraction': 0.,
                'actual_centerline_occlusion_fraction': 0.,
                'occluder_placement_reference': None,
            }, None
            continue
        try:
            observed, _unused_pseudo_mask, metadata = make_test_case(
                rgb, placement, seed, condition)
            metadata.update({
                'synthetic_occlusion': True, 'has_ground_truth': False,
                'rgb_only': True,
                'occluder_placement_reference': 'OpenCV prediction on clean live RGB, not ground truth',
                'occlusion_ratio_interpretation': 'Fraction relative to OpenCV-predicted centerline, not independently verified physical HRM extent',
                'label_provenance': 'NO EVALUATION LABEL: returned pseudo-mask discarded',
                'depth_policy': 'Depth intentionally unused. Do not combine this modified RGB with original unmasked depth.',
            })
            yield condition, observed, metadata, None
        except Exception as exc:
            yield condition, None, None, f'{type(exc).__name__}: {exc}'


def infer_methods(rgb, backends, synchronize=lambda: None):
    """Infer all backends; hybrid prompts come ONLY from observed-RGB U-Net.

    Returned masks are untouched by occluder/placement masks. Latency includes
    CPU preprocessing, inference and CPU mask extraction, not file I/O. Hybrid
    latency adds the already-timed U-Net dependency to its own SAM stage.
    """
    predictions, statuses, timings = {}, {}, {}
    for name, backend in backends.items():
        synchronize()
        start = time.perf_counter()
        exception = None
        try:
            if name == 'unet_robust_mobile_sam':
                predicted = backend.predict_rgb(rgb, prompt_mask=predictions['unet_robust'])
            else:
                predicted = backend.predict_rgb(rgb)
            predicted = np.asarray(predicted)
            if predicted.shape != rgb.shape[:2] or not np.isin(predicted, [0, 1, 255]).all():
                raise ValueError('Expected input-resolution binary mask')
            predicted = (predicted > 0).astype(np.uint8)
            status = getattr(backend, 'last_status', None) or ('ok' if predicted.any() else 'empty_prediction')
        except Exception as exc:
            exception = f'{type(exc).__name__}: {exc}'
            predicted = np.zeros(rgb.shape[:2], dtype=np.uint8)
            status = 'inference_exception'
        synchronize()
        elapsed = (time.perf_counter() - start) * 1000.
        if name == 'unet_robust_mobile_sam':
            elapsed += timings['unet_robust']
        predictions[name] = predicted
        statuses[name] = {'status': status, 'exception': exception,
                          'model_confidence': getattr(backend, 'last_confidence', None),
                          'prompts': getattr(backend, 'last_prompts', None)}
        timings[name] = elapsed
    return predictions, statuses, timings


def tile(rgb, label, subtitle='', mask=None):
    view = rgb.copy()
    if mask is not None:
        foreground = np.asarray(mask) > 0
        view[foreground] = (0.55 * view[foreground] + 0.45 * np.array([0, 235, 80])).astype(np.uint8)
        contours, _ = cv2.findContours(foreground.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(view, contours, -1, (255, 220, 0), 1, cv2.LINE_AA)
    view = cv2.copyMakeBorder(view, 66, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(view, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .53, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(view, subtitle[:55], (8, 46), cv2.FONT_HERSHEY_SIMPLEX, .38, (210, 210, 210), 1, cv2.LINE_AA)
    return view


def sheet(rgb, condition, predictions, statuses, timings):
    panels = [tile(rgb, f'Observed RGB: {condition}', 'NO ground-truth mask; green = predicted HRM')]
    for name, prediction in predictions.items():
        subtitle = f"{statuses[name]['status']} | {timings[name]:.1f} ms"
        panels.append(tile(rgb, DISPLAY_NAMES.get(name, name), subtitle, prediction))
    while len(panels) % 3:
        panels.append(np.zeros_like(panels[0]))
    return np.vstack([np.hstack(panels[i:i + 3]) for i in range(0, len(panels), 3)])


def evaluate(args):
    import torch
    from feasibility_backends import SAMUNetRefinement, UltralyticsSAM
    from run_feasibility import UNetBackend

    capture = Path(args.capture).resolve()
    source_manifest = capture / 'manifest.json'
    metadata = json.loads(source_manifest.read_text())
    frames = metadata['frames'][:args.limit] if args.limit else metadata['frames']
    if not frames:
        raise ValueError('Capture has no frames.')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    backends = {'opencv': OpenCVBaseline(),
                'unet_basic': UNetBackend(args.basic, args.device),
                'unet_robust': UNetBackend(args.robust, args.device)}
    if args.sam:
        sam_device = args.device.removeprefix('cuda:') if args.device.startswith('cuda:') else ('0' if args.device == 'cuda' else args.device)
        backends['mobile_sam_roi'] = UltralyticsSAM(args.sam, device=sam_device)
        backends['unet_robust_mobile_sam'] = SAMUNetRefinement(args.sam, device=sam_device)
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    synchronize = torch.cuda.synchronize if args.device.startswith('cuda') else lambda: None
    first_rgb = cv2.cvtColor(cv2.imread(str(capture / frames[0]['color_file'])), cv2.COLOR_BGR2RGB)
    # Warm-up is excluded from timings, contains no mask labels, no fitting.
    infer_methods(first_rgb, backends, synchronize)
    rows, cases, errors = [], [], []
    examples = defaultdict(list)
    for frame_index, frame in enumerate(frames):
        bgr = cv2.imread(str(capture / frame['color_file']))
        if bgr is None:
            raise OSError(f"Cannot read {frame['color_file']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        name = Path(frame['color_file']).stem
        for condition, observed, case, error in build_cases(rgb, args.seed + frame_index, args.conditions):
            if error:
                errors.append({'source': frame['color_file'], 'condition': condition,
                               'stage': 'synthetic_generation', 'error': error})
                continue
            relative_input = f'inputs/{condition}/{name}.png'
            write_rgb(output / relative_input, observed)
            predictions, statuses, timings = infer_methods(observed, backends, synchronize)
            for method, prediction in predictions.items():
                relative_mask = f'predicted_masks/{method}/{condition}/{name}.png'
                write_mask(output / relative_mask, prediction)
                status = statuses[method]
                rows.append({'frame_index': frame_index, 'source_color_file': frame['color_file'],
                             'source_stamp_ns': frame['color_stamp_ns'],
                             'condition': condition, 'method': method, 'status': status['status'],
                             'exception': status['exception'], 'latency_ms': timings[method],
                             'predicted_foreground_pixels': int(prediction.sum()),
                             'prediction_empty': bool(not prediction.any()),
                             'has_ground_truth': False,
                             'requested_occlusion_fraction': case['requested_centerline_occlusion_fraction'],
                             'placement_reference_occlusion_fraction': case['actual_centerline_occlusion_fraction'],
                             'input_rgb': relative_input, 'predicted_mask': relative_mask})
                if status['exception']:
                    errors.append({'source': frame['color_file'], 'condition': condition,
                                   'method': method, 'stage': 'inference', 'error': status['exception']})
            rendered = sheet(observed, condition, predictions, statuses, timings)
            write_rgb(output / f'overlays/{condition}/{name}.png', rendered)
            cases.append({'source_color_file': frame['color_file'], 'source_stamp_ns': frame['color_stamp_ns'],
                          **case, 'backend_status': statuses})
            if frame_index in {0, len(frames) // 2, len(frames) - 1}:
                examples[condition].append(rendered)
        print(f'Live feasibility {frame_index + 1}/{len(frames)} frames complete', flush=True)
    if rows:
        with (output / 'inference_status.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    timing = []
    for method in backends:
        group = [row for row in rows if row['method'] == method]
        values = [row['latency_ms'] for row in group if row['exception'] is None]
        timing.append({'method': method, 'cases': len(group),
                       'exception_count': sum(row['exception'] is not None for row in group),
                       'empty_prediction_count': sum(row['prediction_empty'] for row in group),
                       'latency_median_ms': float(np.median(values)) if values else None,
                       'latency_p95_ms': float(np.percentile(values, 95)) if values else None})
    for condition, images in examples.items():
        write_rgb(output / f'{condition}_overview.png', np.vstack(images))
        write_rgb(output / f'{condition}_first_frame.png', images[0])
    dump(output / 'cases.json', cases)
    dump(output / 'report.json', {
        'purpose': __doc__, 'capture': str(capture), 'source_manifest_sha256': hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        'source_frames': len(frames), 'conditions': args.conditions, 'seed': args.seed,
        'inference_rows': len(rows), 'generation_failures': sum(error['stage'] == 'synthetic_generation' for error in errors),
        'errors': errors, 'inference_status_and_timing': timing,
        'models': {name: backend.metadata() if hasattr(backend, 'metadata') else {'checkpoint': backend.path}
                   for name, backend in backends.items()},
        'caveats': [
            'Fresh static straight-HRM sequence; no human ground truth and no deliberate physical occlusion.',
            'NO accuracy metrics. Empty prediction is an output diagnostic, not an independently validated failure rate.',
            'Synthetic occluder placement and its percentage use an OpenCV-predicted clean centerline, not true physical extent.',
            'SAM fixed ROI is unchanged for all frames. Hybrid prompts use U-Net predictions on the same observed frame only.',
            'Inference results are not clipped by synthetic occluder, original prediction or a ground-truth mask.',
            'RGB only: original aligned depth was not read or used after synthetic occlusion.',
            'No skeleton/3D/angle/FK/control pipeline was evaluated or changed.',
            'Per-frame inference timing excludes warm-up and disk I/O; hybrid includes U-Net dependency. Not ROS end-to-end rate.',
        ],
    })
    print(json.dumps({'output': str(output), 'rows': len(rows), 'errors': len(errors)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', default='runs/feasibility_20261024/live_capture')
    parser.add_argument('--basic', required=True)
    parser.add_argument('--robust', required=True)
    parser.add_argument('--sam')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--seed', type=int, default=731)
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS))
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative (0 means all frames).')
    evaluate(args)


if __name__ == '__main__':
    main()
