"""Offline segmentation backends with a common RGB -> uint8 {0, 1} contract.

No ROS messages, camera controls, motor commands, model downloads, or dependency
installation occur here. The OpenCV backend reproduces the current HRM mask,
including its largest-component limitation. SAM is an explicitly prompted
zero-shot comparator, not a custom HRM semantic detector.
"""
import os
from pathlib import Path

import cv2
import numpy as np


def _validate_rgb(rgb):
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8
            or rgb.ndim != 3 or rgb.shape[2] != 3
            or min(rgb.shape[:2]) == 0):
        raise ValueError("Expected a nonempty HWC uint8 RGB image.")


class OpenCVBaseline:
    """Faithful RBSC.pre-skeleton mask, without importing ROS dependencies.

    Source: estimation_pkg/postprocess.py RBSC.postprocess and RBSC.smoothing.
    Preserve the broad first HSV interval exactly despite its misleading 'red'
    source name. Empty detections become an empty mask and a failure status;
    they must not be dropped from an evaluation denominator.
    """

    name = "opencv_current"

    def __init__(self, binary_threshold=120):
        self.binary_threshold = int(binary_threshold)
        if not 0 <= self.binary_threshold <= 255:
            raise ValueError("binary_threshold must be in [0, 255].")
        self.last_status = None

    def predict_rgb(self, rgb):
        _validate_rgb(rgb)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        excluded = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 50, 50]), np.array([255, 220, 220])),
            cv2.inRange(hsv, np.array([100, 70, 50]), np.array([130, 255, 255])),
        )
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray[excluded != 0] = 0
        _, binary = cv2.threshold(gray, self.binary_threshold, 255, cv2.THRESH_BINARY)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if count <= 1:
            self.last_status = "no_body_detected"
            return np.zeros(rgb.shape[:2], dtype=np.uint8)
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        body = (labels == largest).astype(np.uint8) * 255
        closed = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filled = np.zeros_like(closed)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        blurred = cv2.GaussianBlur(filled, (15, 15), 0)
        _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        self.last_status = "ok" if smoothed.any() else "empty_after_smoothing"
        return (smoothed > 0).astype(np.uint8)

    def metadata(self):
        return {
            "backend": self.name, "input_encoding": "RGB", "output_values": [0, 1],
            "binary_threshold": self.binary_threshold,
            "largest_component_only": True, "closing_kernel": 15,
            "gaussian_kernel": 15, "post_blur_threshold": 127,
            "source": "estimation_pkg/postprocess.py:RBSC.postprocess/RBSC.smoothing",
            "empty_detection_policy": "empty prediction retained in evaluation",
        }


class UltralyticsSAM:
    """SAM/MobileSAM zero-shot comparison with one fixed normalized box.

    Default prompt is the ENTIRE crop, not a per-frame GT-derived bounding box.
    The same box fractions are used for every frame, including occluded frames.
    No image label, reference mask, test IoU, or clean paired image is accepted
    by predict_rgb. No interactive correction or temporal memory is used.

    An existing checkpoint is required so network/disk changes stay explicit.
    MobileSAM: official ultralytics API, mobile_sam.pt (~41 MB).
    Official reference: https://docs.ultralytics.com/models/mobile-sam/
    """

    name = "mobile_sam_fixed_roi_box"

    def __init__(self, checkpoint, device="0", box_fraction=(0., 0., 1., 1.)):
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Download the official SAM weights explicitly first: {path}")
        box = np.asarray(box_fraction, dtype=np.float32)
        if (box.shape != (4,) or not np.isfinite(box).all()
                or (box < 0).any() or (box > 1).any()
                or box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError("box_fraction must be normalized [x0,y0,x1,y1] with positive area.")
        # Keep third-party preference files out of the ROS user's defaults.
        os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).parent / ".ultralytics"))
        Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
        from ultralytics import SAM

        self.checkpoint = path
        self.device = device
        self.box_fraction = box
        self.model = SAM(str(path))
        self.last_status = None
        self.last_confidence = None

    def predict_rgb(self, rgb):
        _validate_rgb(rgb)
        height, width = rgb.shape[:2]
        box = self.box_fraction * np.array([width - 1, height - 1, width - 1, height - 1])
        return self._predict_with_prompts(rgb, box.tolist())

    def _predict_with_prompts(self, rgb, box, points=None):
        height, width = rgb.shape[:2]
        if self.model.predictor is not None:
            self.model.predictor.reset_image()
        prompts = {"bboxes": box}
        if points is not None:
            # One object, multiple positive points (not one object per point).
            prompts.update(points=[points], labels=[[1] * len(points)])
        # Ultralytics numpy inputs follow OpenCV's BGR convention. Its SAM
        # predictor converts BGR -> RGB internally; do not pass RGB unchanged.
        results = self.model.predict(
            source=np.ascontiguousarray(rgb[..., ::-1]), **prompts,
            device=self.device, imgsz=1024, conf=0.0, save=False, verbose=False,
        )
        self.last_confidence = None
        if not results or results[0].masks is None or len(results[0].masks.data) == 0:
            self.last_status = "no_mask"
            return np.zeros((height, width), dtype=np.uint8)
        result = results[0]
        masks = result.masks.data.detach().cpu().numpy()
        selected = 0
        if result.boxes is not None and len(result.boxes.conf):
            confidence = result.boxes.conf.detach().cpu().numpy()
            selected = int(np.argmax(confidence))
            self.last_confidence = float(confidence[selected])
        mask = masks[selected] > 0.5
        if mask.shape != (height, width):
            # Current Ultralytics SAM returns original-resolution masks. This
            # guard is deliberately strict: naive resizing a letterboxed mask
            # can shift coordinates and silently invalidate the comparison.
            raise ValueError(f"SAM mask shape {mask.shape} != input {(height, width)}")
        self.last_status = "ok" if mask.any() else "empty_mask"
        return mask.astype(np.uint8)

    def metadata(self):
        return {
            "backend": self.name, "checkpoint": str(self.checkpoint),
            "prompt": "one fixed normalized ROI box per frame",
            "box_fraction_xyxy": self.box_fraction.tolist(),
            "ground_truth_prompts": False, "temporal_tracking": False,
            "mask_selection": "highest model confidence if multiple, otherwise single mask",
            "input_encoding": "RGB (converted to BGR for ultralytics)",
            "image_encoder_input": 1024, "output_values": [0, 1],
            "ultralytics_minimum_model_confidence": 0.0,
            "limitations": "prompt-dependent zero-shot comparator; not an HRM-trained detector",
        }


def predicted_mask_prompts(prompt_mask, min_component_pixels=16, max_components=8, box_margin=4):
    """Deterministic prompts from a PREDICTED binary mask, never from test GT.

    Keep up to max_components largest nontrivial visible components so a midbody
    gap does not discard one half as in the classical largest-component rule.
    Each interior point is the distance-transform maximum of that component.
    Return None for empty/tiny predictions; do not fall back to oracle/full ROI.
    """
    mask = np.asarray(prompt_mask)
    if (mask.ndim != 2 or min(mask.shape) == 0
            or not np.isin(mask, [0, 1, 255]).all()):
        raise ValueError("prompt_mask must be a nonempty-shape binary predicted mask (0/1/255).")
    if min_component_pixels < 1 or max_components < 1 or box_margin < 0:
        raise ValueError("Invalid prompt component/margin settings.")
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_component_pixels]
    candidates.sort(key=lambda i: (-int(stats[i, cv2.CC_STAT_AREA]), i))
    selected = candidates[:max_components]
    if not selected:
        return None
    kept = np.isin(labels, selected)
    ys, xs = np.nonzero(kept)
    height, width = mask.shape
    box = [max(0, int(xs.min()) - box_margin), max(0, int(ys.min()) - box_margin),
           min(width - 1, int(xs.max()) + box_margin), min(height - 1, int(ys.max()) + box_margin)]
    points = []
    for component_id in selected:
        # Padding gives finite, meaningful distances even for border/full masks.
        component = (labels == component_id).astype(np.uint8)
        distance = cv2.distanceTransform(np.pad(component, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        y, x = np.unravel_index(np.argmax(distance), distance.shape)
        points.append([int(x), int(y)])
    return {"box": box, "points": points, "retained_components": len(selected),
            "retained_prompt_pixels": int(kept.sum())}


class SAMUNetRefinement(UltralyticsSAM):
    """Hybrid U-Net-prompted SAM; caller must supply U-Net PREDICTIONS only.

    SAM gets one union box and interior positive points for visible components.
    The returned SAM mask is NOT clipped by the U-Net mask or the synthetic
    occluder map. Thus hallucinations/occluder inclusion remain measurable.
    This is a separate hybrid, not evidence of standalone SAM identification.
    """

    name = "unet_prompted_mobile_sam"

    def __init__(self, checkpoint, device="0", min_component_pixels=16, max_components=8, box_margin=4):
        super().__init__(checkpoint, device=device)
        self.min_component_pixels = int(min_component_pixels)
        self.max_components = int(max_components)
        self.box_margin = int(box_margin)
        if self.min_component_pixels < 1 or self.max_components < 1 or self.box_margin < 0:
            raise ValueError("Invalid prompt component/margin settings.")
        self.last_prompts = None

    def predict_rgb(self, rgb, prompt_mask):
        _validate_rgb(rgb)
        if np.shape(prompt_mask) != rgb.shape[:2]:
            raise ValueError("Predicted prompt mask must match RGB's original pixel dimensions.")
        self.last_prompts = predicted_mask_prompts(
            prompt_mask, self.min_component_pixels, self.max_components, self.box_margin)
        if self.last_prompts is None:
            self.last_status = "empty_unet_prompt"
            self.last_confidence = None
            return np.zeros(rgb.shape[:2], dtype=np.uint8)
        return self._predict_with_prompts(rgb, self.last_prompts["box"], self.last_prompts["points"])

    def metadata(self):
        result = super().metadata()
        result.pop("box_fraction_xyxy")
        result.update({
            "prompt": "union box plus one interior positive point per U-Net predicted component",
            "prompt_source": "U-Net prediction on the SAME observed RGB frame only; never GT",
            "minimum_prompt_component_pixels": self.min_component_pixels,
            "maximum_prompt_components": self.max_components,
            "box_margin_pixels": self.box_margin,
            "point_rule": "maximum interior Euclidean distance transform, deterministic first tie",
            "output_clipped_by_unet_or_occluder": False,
            "empty_prediction_policy": "empty output, no SAM call and no fallback prompt",
            "limitations": "hybrid depends on U-Net target localization; can include occluder/hidden region",
        })
        return result
