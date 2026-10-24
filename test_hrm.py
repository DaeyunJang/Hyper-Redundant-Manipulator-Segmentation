"""Run: python -m unittest -v test_hrm.py (no GPU or network required)."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

import hrm
from augment import MODES, apply_occluder, augment_sample, place_foreground


class SegmentationTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((80, 120, 3), 35, np.uint8)
        self.mask = np.zeros((80, 120), np.uint8)
        self.mask[10:70, 50:65] = 1
        self.image[self.mask > 0] = 230

    def test_mask_encoding_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cv2.imwrite(str(root / "rgb.png"), self.image)
            record = {"rgb": "rgb.png", "mask": "mask.png"}
            for scale in (1, 255):
                cv2.imwrite(str(root / "mask.png"), self.mask * scale)
                _, target = hrm.read_pair(root, record)
                np.testing.assert_array_equal(target, self.mask)
            cv2.imwrite(str(root / "mask.png"), self.mask * 127)
            with self.assertRaisesRegex(ValueError, "Non-binary"):
                hrm.read_pair(root, record)

    def test_occluder_removes_exactly_covered_labels(self):
        occluder = np.zeros_like(self.mask)
        occluder[30:45] = 1
        image, target = apply_occluder(self.image, self.mask, occluder, [210, 20, 30])
        np.testing.assert_array_equal(target, self.mask & (1 - occluder))
        self.assertTrue((image[occluder > 0] == [210, 20, 30]).all())
        self.assertEqual(int(self.mask.sum()), 900)

    def test_augmentation_reproducibility_and_semantics(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                a, am = augment_sample(self.image, self.mask, np.random.default_rng(4), [], mode)
                b, bm = augment_sample(self.image, self.mask, np.random.default_rng(4), [], mode)
                np.testing.assert_array_equal(a, b)
                np.testing.assert_array_equal(am, bm)
                self.assertEqual(a.dtype, np.uint8)
                self.assertTrue(np.isin(am, [0, 1]).all())
                if mode in ("background", "lighting", "smoke", "color"):
                    np.testing.assert_array_equal(am, self.mask)
                if mode == "occlusion":
                    self.assertGreater(am.sum(), 0)
                    self.assertLess(am.sum(), self.mask.sum())
                if mode == "negative":
                    self.assertEqual(am.sum(), 0)

    def test_letterbox_and_inference_restore_original_coordinates(self):
        class BrightnessModel(torch.nn.Module):
            def forward(self, x):
                return ((x[:, :1] * float(hrm.STD[0]) + float(hrm.MEAN[0])) - 0.5) * 40
        output = hrm.infer_frame(BrightnessModel(), self.image, 160, torch.device("cpu"))
        self.assertEqual(output.shape, self.mask.shape)
        intersection = np.count_nonzero(output & self.mask)
        union = np.count_nonzero(output | self.mask)
        self.assertGreater(intersection / union, 0.97)
        _, labels, _ = hrm.letterbox(self.image, 160, self.mask)
        self.assertTrue(np.isin(labels, [0, 1]).all())

    def test_placement_covers_frame_and_preserves_visible_foreground(self):
        quadrants, edges, angles, scales = set(), set(), [], []
        clipped = 0
        h, w = self.mask.shape
        for seed in range(192):
            _, target, metadata = place_foreground(
                self.image, self.mask, np.random.default_rng(seed), [], return_metadata=True)
            ys, xs = np.nonzero(target)
            self.assertGreater(len(xs), 0)
            quadrants.add((xs.mean() >= w / 2, ys.mean() >= h / 2))
            for name, pixels in zip(("top", "bottom", "left", "right"),
                                    (target[0], target[-1], target[:, 0], target[:, -1])):
                if pixels.any():
                    edges.add(name)
            self.assertGreaterEqual(metadata["visible_fraction"], 0.6)
            self.assertLessEqual(metadata["visible_fraction"], 1)
            self.assertEqual(metadata["foreground_pixels"], int(target.sum()))
            clipped += metadata["clipped"]
            angles.append(metadata["angle_degrees"])
            scales.append(metadata["scale"])
        self.assertEqual(len(quadrants), 4)
        self.assertEqual(len(edges), 4)
        self.assertGreater(clipped, 20)
        self.assertLess(clipped, 100)
        self.assertLess(min(angles), 15)
        self.assertGreater(max(angles), 345)
        self.assertLess(min(scales), 0.75)
        self.assertGreater(max(scales), 1.15)

    def test_placement_aligns_rgb_mask_without_source_background_bleed(self):
        red = np.array([230, 20, 10], dtype=np.uint8)
        blue = np.array([5, 30, 220], dtype=np.uint8)
        first = np.zeros_like(self.image)
        first[self.mask > 0] = red
        second = first.copy()
        second[self.mask == 0] = [255, 0, 255]
        background = np.broadcast_to(blue, self.image.shape).copy()
        original_mask = self.mask.copy()
        for seed in range(12):
            with patch("augment._background", return_value=background.copy()):
                a, target = place_foreground(first, self.mask, np.random.default_rng(seed), [])
            with patch("augment._background", return_value=background.copy()):
                b, other = place_foreground(second, self.mask, np.random.default_rng(seed), [])
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(target, other)
            self.assertTrue((a[target == 0] == blue).all())
            core = cv2.erode(target, np.ones((5, 5), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
            self.assertGreater(core.sum(), 0)
            self.assertLessEqual(np.abs(a[core > 0].astype(int) - red).max(), 1)
        np.testing.assert_array_equal(self.mask, original_mask)
        self.assertTrue((first[self.mask == 0] == 0).all())
        self.assertTrue((second[self.mask == 0] == [255, 0, 255]).all())
        _, empty = place_foreground(first, np.zeros_like(self.mask), np.random.default_rng(0), [])
        self.assertEqual(empty.sum(), 0)

    def test_export_saves_reproducible_paired_masks_and_preserves_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records = []
            for index in range(2):
                cv2.imwrite(str(root / f"rgb{index}.png"), self.image)
                cv2.imwrite(str(root / f"mask{index}.png"), self.mask * 255)
                records.append({"rgb": f"rgb{index}.png", "mask": f"mask{index}.png",
                                "group": "session_01", "split": "train"})
            (root / "split.json").write_text(json.dumps({"records": records}))
            backgrounds = root / "backgrounds"
            (backgrounds / "train").mkdir(parents=True)
            cv2.imwrite(str(backgrounds / "train/bg.png"), np.full_like(self.image, 90))
            args = argparse.Namespace(data=root, backgrounds=backgrounds, split="train",
                                      count=2, variants=2, modes=["placement", "combined"],
                                      seed=42, output=root / "export")
            hrm.export_augmented(args)
            manifest = json.loads((args.output / "manifest.json").read_text())
            self.assertEqual(manifest["pairs"], 8)
            for entry in manifest["records"]:
                image = cv2.imread(str(args.output / entry["rgb"]))
                mask = cv2.imread(str(args.output / entry["mask"]), cv2.IMREAD_GRAYSCALE)
                self.assertEqual(image.shape[:2], mask.shape)
                self.assertTrue(np.isin(mask, [0, 255]).all())
                source = next(record for record in records if record["rgb"] == entry["source_rgb"])
                rgb, target = hrm.read_pair(root, source)
                expected, labels = augment_sample(rgb, target, np.random.default_rng(entry["seed"]),
                                                 hrm.backgrounds(backgrounds, "train"), entry["mode"])
                np.testing.assert_array_equal(image, cv2.cvtColor(expected, cv2.COLOR_RGB2BGR))
                np.testing.assert_array_equal(mask, labels * 255)
            with self.assertRaises(FileExistsError):
                hrm.export_augmented(args)

    def test_loss_handles_empty_and_positive_masks(self):
        target = torch.zeros(2, 1, 32, 32)
        target[1, :, 8:24, 10:20] = 1
        logits = torch.zeros_like(target, requires_grad=True)
        loss = hrm.loss_fn(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_metrics_perfect_and_empty_predictions(self):
        class Identity(torch.nn.Module):
            def forward(self, x):
                return x
        target = torch.from_numpy(self.mask[None, None].astype(np.float32))
        for y in (target, torch.zeros_like(target)):
            report = hrm.score_model(Identity(), [(y * 40 - 20, y)], torch.device("cpu"))
            self.assertAlmostEqual(report["dice"], 1)
            self.assertAlmostEqual(report["iou"], 1)
            self.assertAlmostEqual(report["boundary_f1_2px"], 1)

    def test_split_uses_whole_sessions_and_detects_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for directory in ("image_rgb", "image_segmentation"):
                (root / directory).mkdir()
            for second in (100, 101, 200, 201, 300, 301):
                name = f"0001_{second}_000000001.png"
                cv2.imwrite(str(root / "image_rgb" / name), self.image)
                cv2.imwrite(str(root / "image_segmentation" / name), self.mask * 255)
            hrm.prepare(argparse.Namespace(data=root, session_gap=2, val_group=None, test_group=None, overwrite=False))
            self.assertEqual(len(hrm.read_records(root, "train")), 4)
            self.assertEqual(len(hrm.read_records(root, "val")), 2)
            with self.assertRaises(ValueError):
                hrm.read_records(root, "test")
            manifest = json.loads((root / "split.json").read_text())
            manifest["records"][0]["split"] = "val"
            (root / "split.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "leakage"):
                hrm.read_records(root, "val")

    def test_evaluation_rejects_reassigned_training_images(self):
        with tempfile.TemporaryDirectory() as folder:
            record = {"rgb": "image_rgb/old_training.png", "mask": "unused", "group": "session_01", "split": "val"}
            (Path(folder) / "split.json").write_text(json.dumps({"records": [record]}))
            args = argparse.Namespace(device="cpu", checkpoint="unused.pt", data=folder, split="val", limit=0)
            checkpoint = {"training_images": [record["rgb"]]}
            with patch("hrm.load_checkpoint", return_value=(None, checkpoint)):
                with self.assertRaisesRegex(ValueError, "Evaluation leakage"):
                    hrm.evaluate(args)


if __name__ == "__main__":
    unittest.main()
