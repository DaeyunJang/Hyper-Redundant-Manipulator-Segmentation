# AUTHOR: Daeyun - HRM-specific paired RGB/visible-mask augmentation.
"""RGB/binary-mask augmentation, using only NumPy and OpenCV.

Masks describe the *visible* HRM: opaque occluders remove labels; translucent
smoke does not. Background paths must come from the current dataset split.
An empty background list uses procedural textures, not simulated surgery.
All randomness comes from the caller's Generator, including stress modes.
"""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


MODES = ("train", "background", "lighting", "smoke", "occlusion", "color", "placement", "combined", "negative")


def _validate(image, mask):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an H x W x 3 RGB uint8 array")
    if mask.dtype != np.uint8 or mask.shape != image.shape[:2]:
        raise ValueError("mask must be an H x W uint8 array matching image")
    if np.any((mask != 0) & (mask != 1)):
        raise ValueError("mask values must be 0 or 1; convert 0/255 masks first")


def _uint8(image):
    return np.clip(image, 0, 255).astype(np.uint8)


def _field(shape, rng, grid=5):
    """Smooth random field in [0, 1], independent of image resolution."""
    values = rng.uniform(0, 1, (grid, grid)).astype(np.float32)
    return np.clip(cv2.resize(values, (shape[1], shape[0]),
                              interpolation=cv2.INTER_CUBIC), 0, 1)


@lru_cache(maxsize=16)
def _read_background(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode background image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _background(shape, paths, rng):
    h, w = shape
    if not paths:
        # Deliberately generic colors/textures; these are not clinical images.
        color = rng.uniform(20, 225, (1, 1, 3))
        texture = (_field(shape, rng) - 0.5)[..., None] * rng.uniform(20, 100)
        noise = rng.normal(0, 3, (h, w, 1))
        return _uint8(color + texture + noise)
    image = _read_background(str(paths[int(rng.integers(len(paths)))]))
    bh, bw = image.shape[:2]
    scale = max(w / bw, h / bh) * rng.uniform(1, 1.35)
    nw, nh = max(w, int(np.ceil(bw * scale))), max(h, int(np.ceil(bh * scale)))
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x, y = int(rng.integers(nw - w + 1)), int(rng.integers(nh - h + 1))
    image = image[y:y + h, x:x + w].copy()
    return image[:, ::-1].copy() if rng.random() < 0.5 else image


def _composite(image, mask, paths, rng):
    background = _background(mask.shape, paths, rng)
    # Feather only inward. No original black pixels outside the mask survive.
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    alpha = np.minimum(distance / 1.25, 1.0)[..., None]
    return _uint8(image * alpha + background * (1.0 - alpha))


def place_foreground(image, mask, rng, background_paths, *, return_metadata=False):
    """Rotate/scale only the HRM and place it on a new background.

    Angles span 0–360 degrees; requested scale is 0.65–1.35, capped to fit the
    rotated sprite. About 30% of placements deliberately cross a random frame
    edge while retaining >=60% of the transformed foreground. The remaining
    placements are fully visible. RGB is warped premultiplied by the mask, so
    source pixels outside the mask cannot introduce dark/color fringes.

    Returns ``(RGB uint8, mask uint8)``; with ``return_metadata=True`` also
    returns angle, scale, source-to-output affine matrix and visible fraction.
    An empty input mask produces a background-only image. Tiny inputs retain
    their original geometry if rotation would erase the foreground.
    """
    _validate(image, mask)
    h, w = mask.shape
    background = _background(mask.shape, background_paths, rng)
    ys, xs = np.nonzero(mask)
    metadata = dict(angle_degrees=0.0, scale=1.0, requested_scale=1.0,
                    visible_fraction=1.0 if len(xs) else 0.0, clipped=False,
                    offset_xy=[0, 0], transform=[[1., 0., 0.], [0., 1., 0.]],
                    full_foreground_pixels=len(xs), foreground_pixels=len(xs))
    if not len(xs) or min(h, w) <= 8:
        result = np.where(mask[..., None] != 0, image, background)
        output = (result, mask.copy())
        return (*output, metadata) if return_metadata else output

    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    # A small transparent border keeps interpolation clear of crop boundaries.
    crop_mask = np.pad(mask[y0:y1, x0:x1], 2)
    crop_rgb = np.pad(image[y0:y1, x0:x1], ((2, 2), (2, 2), (0, 0)))
    ch, cw = crop_mask.shape
    angle, requested = float(rng.uniform(0, 360)), float(rng.uniform(0.65, 1.35))
    sine, cosine = abs(np.sin(np.deg2rad(angle))), abs(np.cos(np.deg2rad(angle)))
    extent_x, extent_y = cosine * cw + sine * ch, sine * cw + cosine * ch
    scale = min(requested, (w - 6) / extent_x, (h - 6) / extent_y)
    sw, sh = int(np.ceil(extent_x * scale)) + 4, int(np.ceil(extent_y * scale)) + 4
    transform = cv2.getRotationMatrix2D(((cw - 1) / 2, (ch - 1) / 2), angle, scale)
    transform[:, 2] += [(sw - cw) / 2, (sh - ch) / 2]
    sprite_mask = cv2.warpAffine(crop_mask, transform, (sw, sh), flags=cv2.INTER_NEAREST)
    full_area = int(sprite_mask.sum())
    if not full_area:
        # Extremely small/thin masks can disappear under nearest-neighbor warp.
        output = (np.where(mask[..., None] != 0, image, background), mask.copy())
        return (*output, metadata) if return_metadata else output
    alpha = cv2.warpAffine(crop_mask.astype(np.float32), transform, (sw, sh))
    premult = cv2.warpAffine(crop_rgb.astype(np.float32) * crop_mask[..., None],
                            transform, (sw, sh))
    # Match image support to the nearest-neighbor ground truth exactly.
    alpha *= sprite_mask
    premult *= sprite_mask[..., None]

    def overlap(left, top):
        x_start, y_start = max(0, left), max(0, top)
        x_end, y_end = min(w, left + sw), min(h, top + sh)
        destination = (slice(y_start, y_end), slice(x_start, x_end))
        source = (slice(y_start - top, y_end - top), slice(x_start - left, x_end - left))
        return destination, source

    left, top = int(rng.integers(w - sw + 1)), int(rng.integers(h - sh + 1))
    minimum_area = min(full_area, max(32, int(0.001 * h * w)))
    if rng.random() < 0.3:
        for _ in range(24):
            edge = int(rng.integers(4))
            candidate_x, candidate_y = int(rng.integers(w - sw + 1)), int(rng.integers(h - sh + 1))
            span = sw if edge < 2 else sh
            cut = int(rng.integers(1, max(2, int(span * 0.45))))
            if edge < 2:
                candidate_x = -cut if edge == 0 else w - sw + cut
            else:
                candidate_y = -cut if edge == 2 else h - sh + cut
            _, source = overlap(candidate_x, candidate_y)
            visible = int(sprite_mask[source].sum())
            if max(minimum_area, 0.6 * full_area) <= visible <= 0.98 * full_area:
                left, top = candidate_x, candidate_y
                break
    destination, source = overlap(left, top)
    labels = np.zeros_like(mask)
    labels[destination] = sprite_mask[source]
    background[destination] = _uint8(premult[source] + background[destination] *
                                      (1 - alpha[source][..., None]))
    # Convert crop-local coordinates back to source-image coordinates.
    transform[:, 2] += np.array([left, top]) - transform[:, :2] @ np.array([x0 - 2, y0 - 2])
    visible = int(labels.sum())
    metadata.update(angle_degrees=angle, scale=float(scale), requested_scale=requested,
                    visible_fraction=visible / full_area, clipped=visible < full_area,
                    offset_xy=[left, top], transform=transform.tolist(),
                    full_foreground_pixels=full_area, foreground_pixels=visible)
    output = (background, labels)
    return (*output, metadata) if return_metadata else output


def _affine(image, mask, rng):
    h, w = mask.shape
    transform = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2),
                                      rng.uniform(-15, 15), rng.uniform(0.9, 1.1))
    transform[:, 2] += rng.uniform(-0.07, 0.07, 2) * [w, h]
    image = cv2.warpAffine(image, transform, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine(mask, transform, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if rng.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def _color(image, mask, rng):
    # Multiplicative color preserves the original surface shading and texture.
    tint = rng.uniform(0.4, 1.1, (1, 1, 3))
    result = _uint8(image.astype(np.float32) * tint)
    return np.where(mask[..., None] != 0, result, image).astype(np.uint8)


def _lighting(image, rng):
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx, yy = xx / max(w - 1, 1), yy / max(h - 1, 1)
    cx, cy = rng.uniform(-0.15, 1.15, 2)
    width = rng.uniform(0.12, 0.5)
    spot = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * width ** 2))
    shade = 0.7 + 0.6 * _field((h, w), rng, 3)
    gain = rng.uniform(0.65, 1.35) * shade * (1 + rng.uniform(-0.5, 0.8) * spot)
    color_temperature = np.array([rng.uniform(0.85, 1.15), 1,
                                  rng.uniform(0.85, 1.15)], dtype=np.float32)
    value = image.astype(np.float32) / 255
    value = np.power(value, rng.uniform(0.75, 1.4))
    value *= gain[..., None] * color_temperature
    value += spot[..., None] * rng.uniform(0, 0.12)
    return _uint8(value * 255)


def _smoke(image, rng):
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    spread = rng.uniform(0.25, 0.65) * max(h, w)
    cloud = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * spread ** 2))
    alpha = cloud * (0.3 + 0.7 * _field((h, w), rng, 7)) * rng.uniform(0.3, 0.7)
    smoke_color = rng.uniform(170, 235) + rng.uniform(-8, 8, (1, 1, 3))
    return _uint8(image * (1 - alpha[..., None]) + smoke_color * alpha[..., None])


def apply_occluder(image, mask, occluder, color_or_image):
    """Overlay an opaque binary region and remove exactly its covered labels.

    ``color_or_image`` is an RGB triplet or an H x W x 3 RGB image. Inputs are
    not modified. Both the returned image and 0/1 mask have dtype uint8.
    """
    _validate(image, mask)
    if occluder.shape != mask.shape or np.any((occluder != 0) & (occluder != 1)):
        raise ValueError("occluder must match mask shape and contain only 0 or 1")
    fill = np.asarray(color_or_image)
    if fill.shape not in ((3,), image.shape):
        raise ValueError("occluder fill must be an RGB triplet or an RGB image")
    covered = occluder.astype(bool)
    result, labels = image.copy(), mask.copy()
    result[covered] = _uint8(fill) if fill.shape == (3,) else _uint8(fill)[covered]
    labels[covered] = 0
    return result, labels


def _occlude(image, mask, rng):
    ys, xs = np.nonzero(mask)
    area = len(xs)
    if area < 4:
        return image, mask
    h, w = mask.shape
    for _ in range(12):
        center = int(rng.integers(area))
        radius = np.sqrt(area / np.pi) * rng.uniform(0.25, 0.65)
        angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)
        angles += rng.uniform(-0.12, 0.12, len(angles))
        radii = radius * rng.uniform(0.65, 1.3, len(angles))
        polygon = np.stack((xs[center] + np.cos(angles) * radii,
                            ys[center] + np.sin(angles) * radii), axis=-1)
        occluder = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(occluder, [np.rint(polygon).astype(np.int32)], 1)
        fraction = np.count_nonzero(occluder & mask) / area
        if 0.025 <= fraction <= 0.35:
            color = rng.uniform(25, 215, (1, 1, 3))
            texture = (_field((h, w), rng) - 0.5)[..., None] * 35
            return apply_occluder(image, mask, occluder, _uint8(color + texture))
    return image, mask


def _camera(image, rng):
    if rng.random() < 0.5:
        image = cv2.GaussianBlur(image, (3, 3), rng.uniform(0.3, 0.9))
    if rng.random() < 0.6:
        image = _uint8(image + rng.normal(0, rng.uniform(1, 7), image.shape))
    if rng.random() < 0.35:
        # RGB->BGR round trip prevents a subtle channel swap after JPEG decode.
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(65, 96))])
        if not ok:
            raise RuntimeError("JPEG augmentation encoding failed")
        image = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return image


def augment_sample(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                   background_paths: list[Path], mode: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """Return augmented RGB uint8 and visible-foreground uint8 {0, 1} arrays.

    ``train`` places the HRM at random positions/orientations in composites;
    original scenes retain modest affine geometry. Approximately 15% of
    nonnegative samples receive only modest geometry (or remain original).
    Five percent of training samples contain only an augmented background.
    ``placement`` changes only foreground pose and background; ``combined``
    adds every appearance effect. Other stress modes preserve geometry, and
    ``negative`` produces a background with an empty foreground mask.
    Every mode is reproducible given a seeded rng.
    Source arrays are never modified. Invalid/corrupt background files raise.
    """
    _validate(image, mask)
    if mode not in MODES:
        raise ValueError(f"Unknown augmentation mode {mode!r}; choose from {MODES}")
    image, mask = image.copy(), mask.copy()
    train = mode == "train"
    if mode == "negative" or (train and rng.random() < 0.05):
        image = _background(mask.shape, background_paths, rng)
        image = _camera(_lighting(image, rng), rng)
        return np.ascontiguousarray(image), np.zeros_like(mask)
    if train:
        if rng.random() < 0.15:
            if rng.random() < 0.8:
                image, mask = _affine(image, mask, rng)
            return image, mask
        if rng.random() < 0.9:
            image, mask = place_foreground(image, mask, rng, background_paths)
        elif rng.random() < 0.8:
            image, mask = _affine(image, mask, rng)
    elif mode in ("placement", "combined"):
        image, mask = place_foreground(image, mask, rng, background_paths)
    elif mode == "background":
        image = _composite(image, mask, background_paths, rng)

    def use(effect, probability):
        return rng.random() < probability if train else mode in (effect, "combined")

    if use("color", 0.45):
        image = _color(image, mask, rng)
    if use("occlusion", 0.35):
        image, mask = _occlude(image, mask, rng)
    if use("lighting", 0.8):
        image = _lighting(image, rng)
    if use("smoke", 0.35):
        image = _smoke(image, rng)
    if (train and rng.random() < 0.4) or mode == "combined":
        image = _camera(image, rng)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)
