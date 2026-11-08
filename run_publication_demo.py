#!/usr/bin/env python3
# AUTHOR: Daeyun - Portable, paired visible-depth U-Net evaluation entry point.
"""Replay the packaged V2 RGB-D examples without ROS or the full recording tree.

The companion Hyper-Redundant-Manipulator repository must be a sibling folder:
feasibility_shape reads its reconstruction/FK source and hardware definitions.
All reported errors are changes from the same frame's clean estimate, NOT
ground-truth accuracy. This wrapper reuses the existing evaluation unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_pose_occlusion_v2 import evaluate


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'examples/publication_demo')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True, help='New output directory; never overwrites a prior run')
    parser.add_argument('--limit', type=int, default=-1, help='Negative: all 104 frames; positive: evenly spaced subset')
    parser.add_argument('--include-opencv', action='store_true', help='Also run the existing OpenCV comparator')
    args = parser.parse_args()
    if args.limit == 0:
        parser.error('--limit must be positive, or negative for all frames.')
    bundle = args.bundle.resolve()
    protocol_path = bundle / 'metadata/pose_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    evaluate(argparse.Namespace(
        record_index=str(bundle / 'rgbd/index.json'), record_limit=args.limit,
        shape_protocol=str(protocol_path),
        methods=['opencv', 'unet_v2'] if args.include_opencv else ['unet_v2'],
        v1=None, v2=str(bundle / 'models/unet_v2.pt'), device=args.device,
        output=args.output, conditions=['clean', 'multi3_30'],
        seed=int(protocol['seed']),
        minimum_path_points=int(protocol['shape_estimator_metadata']['minimum_path_points']),
    ))


if __name__ == '__main__':
    main()
