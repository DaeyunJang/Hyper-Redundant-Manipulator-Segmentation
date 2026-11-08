#!/usr/bin/env python3
"""Finite, subscriber-only RGB-D capture for segmentation feasibility.

Launch the camera independently. This script never starts robot nodes, sends
commands, changes camera parameters, or treats algorithm masks as ground truth.
Saved depth is aligned to the color image; both are cropped with the same ROI.
"""

import argparse
from collections import deque
import json
from pathlib import Path
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


def stamp_ns(msg):
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def info_dict(msg):
    return {
        'stamp_ns': stamp_ns(msg), 'frame_id': msg.header.frame_id,
        'width': msg.width, 'height': msg.height,
        'distortion_model': msg.distortion_model,
        'K': list(msg.k), 'D': list(msg.d), 'R': list(msg.r), 'P': list(msg.p),
        'binning_x': msg.binning_x, 'binning_y': msg.binning_y,
        'roi': {key: getattr(msg.roi, key) for key in
                ('x_offset', 'y_offset', 'height', 'width', 'do_rectify')},
    }


class Capture(Node):
    def __init__(self, args):
        super().__init__('segmentation_feasibility_capture')
        self.args = args
        self.output = Path(args.output).resolve()
        if self.output.exists() and any(self.output.iterdir()):
            raise ValueError(f'Refusing to overwrite a nonempty capture folder: {self.output}')
        self.output.mkdir(parents=True, exist_ok=True)
        for child in ('hrm_crop', 'hrm_crop_depth'):
            (self.output / child).mkdir()
        self.bridge = CvBridge()
        self.started = time.monotonic()
        self.colors = deque(maxlen=100)
        self.depths = deque(maxlen=100)
        self.camera_infos = deque(maxlen=100)
        self.rows = []
        self.last_stamp_ns = None
        self.done = False
        self.failure = None
        self.received = {'color': 0, 'depth': 0, 'camera_info': 0}
        self.subscriptions_ = [
            self.create_subscription(Image, args.color_topic, self.on_color,
                                     qos_profile_sensor_data),
            self.create_subscription(Image, args.depth_topic, self.on_depth,
                                     qos_profile_sensor_data),
            self.create_subscription(CameraInfo, args.info_topic, self.on_info,
                                     qos_profile_sensor_data),
        ]
        self.timer_ = self.create_timer(0.05, self.process)

    def on_color(self, msg):
        self.received['color'] += 1
        self.colors.append((time.monotonic(), time.time_ns(), msg))

    def on_depth(self, msg):
        self.received['depth'] += 1
        self.depths.append((time.time_ns(), msg))

    def on_info(self, msg):
        self.received['camera_info'] += 1
        self.camera_infos.append(msg)

    def process(self):
        if self.done:
            return
        now = time.monotonic()
        if now - self.started > self.args.timeout_sec:
            self.failure = f'Timeout after {self.args.timeout_sec} seconds.'
            self.done = True
            return
        if now - self.started < self.args.warmup_sec:
            self.colors.clear()
            return
        if not self.depths or not self.camera_infos:
            return
        while self.colors:
            receipt_mono, receipt_ns, color = self.colors[0]
            if now - receipt_mono < 0.12:
                return  # allow same-frame depth to arrive before matching
            self.colors.popleft()
            color_stamp = stamp_ns(color)
            if self.last_stamp_ns is not None and (
                    color_stamp - self.last_stamp_ns < self.args.interval_sec * 1e9):
                continue
            depth_receipt_ns, depth = min(
                self.depths, key=lambda entry: abs(stamp_ns(entry[1]) - color_stamp))
            delta_ns = stamp_ns(depth) - color_stamp
            if abs(delta_ns) > self.args.max_pair_delta_ms * 1e6:
                continue
            info = min(self.camera_infos, key=lambda msg: abs(stamp_ns(msg) - color_stamp))
            if color.header.frame_id != depth.header.frame_id:
                continue
            if (color.width, color.height) != (depth.width, depth.height):
                raise ValueError('Aligned depth and color dimensions disagree.')
            if (info.width, info.height) != (color.width, color.height):
                raise ValueError('Color camera intrinsics dimensions disagree.')
            if info.header.frame_id != color.header.frame_id:
                raise ValueError('Color camera intrinsics frame_id disagrees.')
            x, y, width, height = self.args.roi
            if x + width > color.width or y + height > color.height:
                raise ValueError(f'ROI {self.args.roi} exceeds color frame dimensions.')
            bgr = self.bridge.imgmsg_to_cv2(color, desired_encoding='bgr8')
            depth_array = self.bridge.imgmsg_to_cv2(depth, desired_encoding='passthrough')
            crop = bgr[y:y + height, x:x + width]
            depth_crop = depth_array[y:y + height, x:x + width]
            index = len(self.rows)
            name = f'{index:04d}_{color_stamp}'
            color_file = f'hrm_crop/{name}.png'
            if not cv2.imwrite(str(self.output / color_file), crop):
                raise OSError('Color PNG write failed.')
            if depth.encoding.upper() == '16UC1' and depth_crop.dtype == np.uint16:
                depth_file = f'hrm_crop_depth/{name}.png'
                depth_scale_m = 0.001
                if not cv2.imwrite(str(self.output / depth_file), depth_crop):
                    raise OSError('Depth PNG write failed.')
            elif depth.encoding.upper() == '32FC1' and depth_crop.dtype == np.float32:
                depth_file = f'hrm_crop_depth/{name}.npy'
                depth_scale_m = 1.0
                np.save(self.output / depth_file, depth_crop)
            else:
                raise ValueError(f'Unsupported depth encoding/dtype {depth.encoding}/{depth_crop.dtype}')
            full_info = info_dict(info)
            crop_info = info_dict(info)
            crop_info.update({'width': width, 'height': height})
            crop_info['K'][2] -= x
            crop_info['K'][5] -= y
            crop_info['P'][2] -= x
            crop_info['P'][6] -= y
            crop_info['roi'] = {'x_offset': 0, 'y_offset': 0, 'width': width,
                                'height': height, 'do_rectify': False}
            self.rows.append({
                'index': index, 'color_file': color_file, 'depth_file': depth_file,
                'color_stamp_ns': color_stamp, 'depth_stamp_ns': stamp_ns(depth),
                'depth_minus_color_ms': delta_ns / 1e6,
                'color_receipt_unix_ns': receipt_ns,
                'depth_receipt_unix_ns': depth_receipt_ns,
                'color_source_encoding': color.encoding,
                'saved_color_encoding': 'bgr8 (OpenCV PNG; RGB appearance preserved)',
                'depth_encoding': depth.encoding, 'depth_dtype': str(depth_crop.dtype),
                'depth_scale_m': depth_scale_m,
                'depth_valid_fraction': float(np.mean(np.isfinite(depth_crop) & (depth_crop > 0))),
                'frame_id': color.header.frame_id,
                'source_roi_xywh': [x, y, width, height],
                'source_color_camera_info': full_info,
                'crop_color_camera_info': crop_info,
            })
            if index == 0:
                cv2.imwrite(str(self.output / 'first_full_color.png'), bgr)
            self.last_stamp_ns = color_stamp
            self.get_logger().info(f'Captured pair {index + 1}/{self.args.count}, delta={delta_ns / 1e6:.3f} ms')
            if len(self.rows) >= self.args.count:
                self.done = True
            return

    def save_manifest(self):
        manifest = {
            'purpose': 'Fresh HRM RGB-D feasibility inference; no human ground-truth mask.',
            'user_reported_condition': self.args.condition,
            'physical_occlusion_supplied': False,
            'capture_topics': {'color': self.args.color_topic, 'aligned_depth': self.args.depth_topic,
                               'color_camera_info': self.args.info_topic},
            'requested_pairs': self.args.count, 'saved_pairs': len(self.rows),
            'received_messages': self.received, 'failure': self.failure,
            'warmup_sec': self.args.warmup_sec, 'minimum_frame_interval_sec': self.args.interval_sec,
            'maximum_pair_delta_ms': self.args.max_pair_delta_ms,
            'crop_intrinsics_note': 'Crop K/P already have ROI offset removed. Use roi_offset=(0,0) with crop K.',
            'depth_alignment_note': 'Camera publishes aligned_depth_to_color. No inference of hardware synchronization beyond header matching.',
            'depth_units_note': 'ROS RealSense 16UC1 depth is millimeters; 32FC1 is meters.',
            'frames': self.rows,
        }
        (self.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--condition', default='HRM straight, camera physically connected.',
                        help='User-reported acquisition context, not geometric ground truth.')
    parser.add_argument('--roi', nargs=4, type=int, default=[220, 121, 379, 337],
                        metavar=('X', 'Y', 'WIDTH', 'HEIGHT'))
    parser.add_argument('--color-topic', default='/camera/camera/color/image_rect_raw')
    parser.add_argument('--depth-topic', default='/camera/camera/aligned_depth_to_color/image_raw')
    parser.add_argument('--info-topic', default='/camera/camera/color/camera_info')
    parser.add_argument('--interval-sec', type=float, default=0.25)
    parser.add_argument('--max-pair-delta-ms', type=float, default=50.0)
    parser.add_argument('--warmup-sec', type=float, default=3.0)
    parser.add_argument('--timeout-sec', type=float, default=30.0)
    args = parser.parse_args()
    if args.count < 1 or min(args.roi[:2]) < 0 or min(args.roi[2:]) < 1:
        parser.error('Count and ROI sizes must be positive; ROI origin must be nonnegative.')
    rclpy.init()
    node = Capture(args)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, Exception) as exc:
        node.failure = f'{type(exc).__name__}: {exc}'
    finally:
        node.save_manifest()
        node.destroy_node()
        rclpy.shutdown()
    print(json.dumps({'output': str(node.output), 'saved_pairs': len(node.rows), 'failure': node.failure}))
    return 0 if len(node.rows) == args.count and node.failure is None else 2


if __name__ == '__main__':
    raise SystemExit(main())
