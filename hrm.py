"""Binary HRM segmentation: prepare, preview, export, train, evaluate, predict, benchmark.

PNG masks remain 0/255 on disk; training targets are float32 0/1.
Only the white body is annotated, not individual links or black attachments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from augment import MODES, augment_sample

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save image: {path}")


def read_pair(root, record):
    image = cv2.imread(str(Path(root) / record["rgb"]), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(Path(root) / record["mask"]), cv2.IMREAD_UNCHANGED)
    if image is None or mask is None:
        raise ValueError(f"Unreadable pair: {record['rgb']}, {record['mask']}")
    if mask.ndim != 2 or image.shape[:2] != mask.shape:
        raise ValueError(f"Expected aligned RGB + single-channel mask: {record['rgb']}")
    if not np.isin(mask, [0, 1, 255]).all():
        raise ValueError(f"Non-binary mask: {record['mask']}; values={np.unique(mask)}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (mask > 0).astype(np.uint8)


def prepare(args):
    """Audit every pair; reconstruct sessions from acquisition timestamps."""
    root = Path(args.data)
    rgb = {p.name: p for p in (root / "image_rgb").iterdir() if p.suffix.lower() in EXTENSIONS}
    masks = {p.name: p for p in (root / "image_segmentation").iterdir() if p.suffix.lower() in EXTENSIONS}
    if not rgb or rgb.keys() != masks.keys():
        raise ValueError(f"Pair mismatch: RGB-only={list(rgb.keys()-masks.keys())[:5]}, "
                         f"mask-only={list(masks.keys()-rgb.keys())[:5]}")
    timestamps = {}
    for name in rgb:
        match = re.fullmatch(r"\d+_(\d+)_(\d+)", Path(name).stem)
        if not match:
            raise ValueError(f"Cannot infer recording session from {name}. "
                             "Use frame_seconds_nanoseconds filenames or supply an edited split.json.")
        timestamps[name] = int(match[1]) + int(match[2]) / 1e9
    records, sessions, shapes, values, fractions = [], {}, set(), set(), []
    last, group_index = -float("inf"), 0
    for name in tqdm(sorted(rgb, key=timestamps.get), desc="Auditing pairs"):
        stamp = timestamps[name]
        if stamp - last > args.session_gap:
            group_index += 1
        group = f"session_{group_index:02d}"
        last = stamp
        record = {"rgb": f"image_rgb/{name}", "mask": f"image_segmentation/{name}", "group": group}
        image, mask = read_pair(root, record)
        raw = cv2.imread(str(masks[name]), cv2.IMREAD_UNCHANGED)
        values.update(int(v) for v in np.unique(raw))
        shapes.add(tuple(image.shape))
        fractions.append(float(mask.mean()))
        sessions.setdefault(group, {"count": 0, "start": stamp})
        sessions[group]["count"] += 1
        sessions[group]["end"] = stamp
        records.append(record)
    if len(sessions) < 2:
        raise ValueError("Need at least two recording sessions for an independent validation split.")
    val_group = args.val_group or list(sessions)[-1]
    if val_group not in sessions or (args.test_group and args.test_group not in sessions):
        raise ValueError(f"Unknown group; available: {list(sessions)}")
    if val_group == args.test_group:
        raise ValueError("Validation and test must use different sessions.")
    for record in records:
        record["split"] = ("val" if record["group"] == val_group else
                           "test" if record["group"] == args.test_group else "train")
    if not any(r["split"] == "train" for r in records):
        raise ValueError("No training session remains.")
    summary = {"pairs": len(records), "image_shapes": sorted(shapes), "mask_values": sorted(values),
               "foreground_fraction_min_median_max": [min(fractions), float(np.median(fractions)), max(fractions)],
               "sessions": sessions,
               "split_counts": {s: sum(r["split"] == s for r in records) for s in ("train", "val", "test")},
               "note": "No independent test set unless --test-group is supplied. Hold out real deployment images separately."}
    path = root / "split.json"
    if path.exists() and not args.overwrite:
        raise FileExistsError(f"{path} exists; use --overwrite to intentionally regenerate splits.")
    write_json(path, {"version": 1, "summary": summary, "records": records})
    print(json.dumps(summary, indent=2))
    print(f"Saved {path}")


def read_records(root, split):
    path = Path(root) / "split.json"
    if not path.exists():
        raise FileNotFoundError(f"Run python hrm.py prepare --data {root} first.")
    manifest = json.loads(path.read_text())
    records = manifest["records"]
    group_splits = {}
    seen = set()
    for r in records:
        if r["rgb"] in seen:
            raise ValueError(f"Duplicate manifest entry: {r['rgb']}")
        seen.add(r["rgb"])
        group_splits.setdefault(r["group"], set()).add(r["split"])
    if any(len(s) > 1 for s in group_splits.values()):
        raise ValueError("Recording-session leakage in split.json.")
    selected = [r for r in records if r["split"] == split]
    if not selected:
        raise ValueError(f"No {split} records in {path}; an independent test set is not created by default.")
    return selected


def backgrounds(root, split):
    """Never reuse train backgrounds in validation/test."""
    folder = Path(root) / split
    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not paths:
        print(f"No images in {folder}: using generic procedural backgrounds (not surgical scenes).")
    return paths


def augmentation_provenance(paths):
    """Changing code or the ordered background pool changes seeded examples."""
    source = Path(__file__).with_name("augment.py").read_bytes()
    return {"augment_py_sha256": hashlib.sha256(source).hexdigest(),
            "numpy": np.__version__, "opencv": cv2.__version__,
            "backgrounds_in_order": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}


def letterbox(image, size, mask=None):
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    top, left = (size - nh) // 2, (size - nw) // 2
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[top:top+nh, left:left+nw] = cv2.resize(image, (nw, nh), interpolation=interpolation)
    target = None
    if mask is not None:
        target = np.zeros((size, size), dtype=np.uint8)
        target[top:top+nh, left:left+nw] = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return canvas, target, (top, left, nh, nw)


def to_tensor(image):
    normalized = (image.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))


class HRMDataset(Dataset):
    def __init__(self, root, records, size, background_paths=(), mode="clean", seed=42):
        self.root, self.records, self.size = root, records, size
        self.background_paths, self.mode, self.seed = background_paths, mode, seed

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image, mask = read_pair(self.root, record)
        if self.mode != "clean":
            seed = (np.random.randint(0, 2**32 - 1) if self.mode == "train" else
                    self.seed + zlib.crc32(record["rgb"].encode()))
            image, mask = augment_sample(image, mask, np.random.default_rng(seed),
                                         self.background_paths, mode=self.mode)
        image, mask, _ = letterbox(image, self.size, mask)
        return to_tensor(image), torch.from_numpy(mask[None].astype(np.float32))


def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(0)


def loader(dataset, args, shuffle=False):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
                      pin_memory=torch.cuda.is_available(), worker_init_fn=worker_seed,
                      persistent_workers=args.workers > 0, generator=torch.Generator().manual_seed(args.seed))


def build_model(encoder="resnet18", pretrained=False):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=1, decoder_channels=(128, 64, 32, 16, 16))


def loss_fn(logits, target):
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    intersection = (probability * target).sum((1, 2, 3))
    dice = (2 * intersection + 1) / (probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return 0.5 * bce + 0.5 * (1 - dice.mean())


def pick_device(name):
    device = torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; install a CUDA PyTorch wheel or pass --device cpu.")
    return device


@torch.inference_mode()
def score_model(model, batches, device, threshold=0.5):
    model.eval()
    totals = torch.zeros(11, device=device, dtype=torch.float64)
    for image, target in batches:
        image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(image)
        prediction = logits.float().sigmoid() >= threshold
        truth = target > 0.5
        tp = (prediction & truth).sum()
        fp = (prediction & ~truth).sum()
        fn = (~prediction & truth).sum()
        # Inner boundaries, matched within two pixels on the network input grid.
        pred_float = prediction.float()
        pb = pred_float - (-F.max_pool2d(-pred_float, 3, 1, 1))
        tb = target - (-F.max_pool2d(-target, 3, 1, 1))
        pmatch = (pb * F.max_pool2d(tb, 5, 1, 2)).sum()
        tmatch = (tb * F.max_pool2d(pb, 5, 1, 2)).sum()
        batch_size = image.shape[0]
        totals += torch.stack([tp, fp, fn, loss_fn(logits, target) * batch_size,
                               totals.new_tensor(batch_size), pmatch, pb.sum(), tmatch, tb.sum(),
                               totals.new_tensor(target.numel()), prediction.sum()])
    tp, fp, fn, loss, count, pmatch, pcount, tmatch, tcount, pixels, foreground = totals.tolist()
    safe = lambda numerator, denominator: numerator / denominator if denominator else 1.0
    precision, recall = safe(tp, tp+fp), safe(tp, tp+fn)
    bp, br = safe(pmatch, pcount), safe(tmatch, tcount)
    return {"images": int(count), "loss": loss / max(count, 1),
            "dice": safe(2*tp, 2*tp+fp+fn), "iou": safe(tp, tp+fp+fn),
            "precision": precision, "recall": recall,
            "predicted_foreground_fraction": foreground / max(pixels, 1),
            "false_positive_pixels_per_image": fp / max(count, 1),
            "boundary_f1_2px": 2*bp*br / (bp+br) if bp+br else 0.0}


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["encoder"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def evenly_spaced(records, limit):
    if not limit or limit >= len(records):
        return records
    return [records[i] for i in np.linspace(0, len(records)-1, limit, dtype=int)]


def train(args):
    device = pick_device(args.device)
    out = Path(args.output)
    if any((out / name).exists() for name in ("best.pt", "last.pt", "config.json", "history.csv")):
        raise FileExistsError(f"{out} already contains a run; choose a new --output to preserve it.")
    out.mkdir(parents=True, exist_ok=True)
    train_records, val_records = read_records(args.data, "train"), read_records(args.data, "val")
    train_backgrounds = backgrounds(args.backgrounds, "train")
    val_backgrounds = backgrounds(args.backgrounds, "val")
    training = loader(HRMDataset(args.data, train_records, args.size, train_backgrounds,
                                "clean" if args.no_augment else "train", args.seed), args, shuffle=True)
    validation = loader(HRMDataset(args.data, val_records, args.size), args)
    stress_records = evenly_spaced(val_records, args.stress_samples)
    stress = loader(HRMDataset(args.data, stress_records, args.size, val_backgrounds, "combined", args.seed), args)
    model = build_model(args.encoder, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    config = {k: v for k, v in vars(args).items() if k != "func"}
    config.update({"parameters": sum(p.numel() for p in model.parameters()),
                   "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                   "torch": str(torch.__version__), "train_images": len(train_records), "val_images": len(val_records),
                   "selection": "mean(clean validation Dice, fixed synthetic validation Dice)",
                   "mask_encoding": "PNG 0/255; float32 tensor 0/1",
                   "augmentation_train": augmentation_provenance(train_backgrounds),
                   "augmentation_val": augmentation_provenance(val_backgrounds)})
    write_json(out / "config.json", config)
    shutil.copyfile(Path(__file__).with_name("augment.py"), out / "augment_snapshot.py")
    write_json(out / "split.json", json.loads((Path(args.data) / "split.json").read_text()))
    if (Path(args.backgrounds) / "sources.json").exists():
        write_json(out / "background_sources.json", json.loads((Path(args.backgrounds) / "sources.json").read_text()))
    best, stale = -1, 0
    print(f"Training {args.encoder} on {config['device_name']}; {config['parameters']:,} parameters", flush=True)
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        running_loss = 0.0
        for step, (image, target) in enumerate(training, 1):
            image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(image)
                loss = loss_fn(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * image.shape[0]
            if step % 100 == 0:
                print(f"  epoch {epoch}, batch {step}/{len(training)}, loss={loss.item():.4f}", flush=True)
        scheduler.step()
        clean_metrics = score_model(model, validation, device)
        stress_metrics = score_model(model, stress, device)
        selection = (clean_metrics["dice"] + stress_metrics["dice"]) / 2
        row = {"epoch": epoch, "train_loss": running_loss / len(training.dataset),
               "val_dice": clean_metrics["dice"], "val_iou": clean_metrics["iou"],
               "stress_dice": stress_metrics["dice"], "selection_score": selection,
               "seconds": time.perf_counter() - start}
        with (out / "history.csv").open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        checkpoint = {"model": model.state_dict(), "encoder": args.encoder, "size": args.size,
                      "epoch": epoch, "threshold": 0.5, "clean": clean_metrics, "synthetic": stress_metrics,
                      "selection_score": selection, "config": config,
                      "training_images": [r["rgb"] for r in train_records]}
        torch.save(checkpoint, out / "last.pt")
        if selection > best:
            best, stale = selection, 0
            torch.save(checkpoint, out / "best.pt")
            write_json(out / "best_metrics.json", {k: v for k, v in checkpoint.items() if k not in ("model", "training_images")})
        else:
            stale += 1
        print(json.dumps(row), flush=True)
        if args.patience and stale >= args.patience:
            print(f"Early stop after {stale} epochs without validation improvement.")
            break
    print(f"Best checkpoint: {out / 'best.pt'} (validation selection={best:.4f})")


def evaluate(args):
    device = pick_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    records = evenly_spaced(read_records(args.data, args.split), args.limit)
    overlap = set(checkpoint["training_images"]) & {r["rgb"] for r in records}
    if overlap:
        raise ValueError(f"Evaluation leakage: {len(overlap)} selected images were used to train this checkpoint.")
    paths = backgrounds(args.backgrounds, args.split)
    report = {"checkpoint": str(args.checkpoint), "split": args.split, "size": checkpoint["size"],
              "threshold": args.threshold, "seed": args.seed,
              "augmentation": augmentation_provenance(paths),
              "note": "Synthetic stress is not real surgical-environment validation.", "conditions": {}}
    for mode in args.modes:
        batches = loader(HRMDataset(args.data, records, checkpoint["size"], paths, mode, args.seed), args)
        metrics = score_model(model, batches, device, args.threshold)
        report["conditions"][mode] = metrics
        print(mode, json.dumps(metrics), flush=True)
    write_json(args.output, report)


def overlay(image, mask):
    result = image.copy()
    selected = mask > 0
    result[selected] = (0.55 * result[selected] + 0.45 * np.array([0, 220, 60])).astype(np.uint8)
    return result


def preview(args):
    records = evenly_spaced(read_records(args.data, args.split), args.count)
    paths = backgrounds(args.backgrounds, args.split)
    rows = []
    for index, record in enumerate(records):
        image, mask = read_pair(args.data, record)
        aug, target = augment_sample(image, mask, np.random.default_rng(args.seed+index), paths, args.mode)
        tiles = [image, aug, overlay(aug, target), cv2.cvtColor(target * 255, cv2.COLOR_GRAY2RGB)]
        for tile, label in zip(tiles, ["Original", args.mode, "Target overlay", "Target 0/255"]):
            cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        rows.append(np.hstack(tiles))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_image(args.output, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    print(f"Saved {args.output}")


def export_augmented(args):
    """Materialize inspectable PNG pairs without altering online training data."""
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"Preserving {out}; choose a new --output for another export.")
    if len(set(args.modes)) != len(args.modes):
        raise ValueError("Export modes must be unique.")
    records = evenly_spaced(read_records(args.data, args.split), args.count)
    paths = backgrounds(args.backgrounds, args.split)
    (out / "image_rgb").mkdir(parents=True)
    (out / "image_segmentation").mkdir()
    entries = []
    for index, record in enumerate(tqdm(records, desc="Exporting augmentation")):
        image, mask = read_pair(args.data, record)
        for mode in args.modes:
            for variant in range(args.variants):
                seed = args.seed + zlib.crc32(f"{record['rgb']}:{mode}:{variant}".encode())
                aug, target = augment_sample(image, mask, np.random.default_rng(seed), paths, mode)
                name = f"{index+1:04d}_{mode}_{variant+1:02d}.png"
                save_image(out / "image_rgb" / name, cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))
                save_image(out / "image_segmentation" / name, target * 255)
                entries.append({"rgb": f"image_rgb/{name}", "mask": f"image_segmentation/{name}",
                                "source_rgb": record["rgb"], "source_mask": record["mask"],
                                "group": record["group"], "split": args.split, "mode": mode, "seed": seed})
    manifest = {"purpose": "Reproducible examples, not saved historical training batches; training augments online.",
                "pairs": len(entries), "original_images": len(records), "split": args.split,
                "mask_encoding": "uint8 PNG: background=0, visible body=255",
                "augmentation": augmentation_provenance(paths), "records": entries}
    source = Path(args.backgrounds) / "sources.json"
    if source.exists():
        shutil.copyfile(source, out / "background_sources.json")
        manifest["background_sources"] = "background_sources.json"
    for license_file in Path(args.backgrounds).glob("LICENSE*"):
        if license_file.is_file():
            shutil.copyfile(license_file, out / license_file.name)
    write_json(out / "manifest.json", manifest)
    tiles = []
    for entry in evenly_spaced(entries, 24):
        tile = cv2.resize(cv2.imread(str(out / entry["rgb"])), (256, 228))
        tile = cv2.copyMakeBorder(tile, 24, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(tile, Path(entry["rgb"]).stem, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1)
        tiles.append(tile)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    save_image(out / "preview.jpg", np.vstack([np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)]))
    print(f"Saved {len(entries)} RGB/mask pairs and preview to {out}")


@torch.inference_mode()
def infer_frame(model, rgb, size, device, threshold=0.5):
    image, _, (top, left, nh, nw) = letterbox(rgb, size)
    tensor = to_tensor(image)[None].to(device)
    with torch.autocast(device.type, enabled=device.type == "cuda"):
        probability = model(tensor).float().sigmoid()[0, 0].cpu().numpy()
    probability = cv2.resize(probability[top:top+nh, left:left+nw], (rgb.shape[1], rgb.shape[0]))
    return (probability >= threshold).astype(np.uint8)


def predict(args):
    device = pick_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    path = Path(args.source)
    images = sorted(p for p in path.iterdir() if p.suffix.lower() in EXTENSIONS) if path.is_dir() else ([path] if path.suffix.lower() in EXTENSIONS else [])
    if images:
        for p in tqdm(images, desc="Predicting"):
            bgr = cv2.imread(str(p))
            if bgr is None:
                raise ValueError(f"Unreadable image: {p}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask = infer_frame(model, rgb, checkpoint["size"], device, args.threshold)
            save_image(out / f"{p.stem}_mask.png", mask * 255)
            save_image(out / f"{p.stem}_overlay.jpg", cv2.cvtColor(overlay(rgb, mask), cv2.COLOR_RGB2BGR))
        print(f"Masks (uint8 0/255) and overlays saved to {out}")
        return
    capture = cv2.VideoCapture(int(args.source) if args.source.isdecimal() else args.source)
    if not capture.isOpened():
        raise ValueError(f"Cannot open video/camera: {args.source}")
    writer, frames, elapsed = None, 0, 0.0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok or (args.max_frames and frames >= args.max_frames):
                break
            start = time.perf_counter()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask = infer_frame(model, rgb, checkpoint["size"], device, args.threshold)
            view = cv2.cvtColor(overlay(rgb, mask), cv2.COLOR_RGB2BGR)
            dt = time.perf_counter() - start
            elapsed += dt
            frames += 1
            cv2.putText(view, f"Processing: {1/dt:.1f} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if writer is None:
                fps = capture.get(cv2.CAP_PROP_FPS)
                fps = fps if np.isfinite(fps) and fps > 0 else 30.0
                writer = cv2.VideoWriter(str(out / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (view.shape[1], view.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError("Video writer could not open overlay.mp4")
            writer.write(view)
            if args.save_masks:
                save_image(out / f"{frames:08d}_mask.png", mask * 255)
            if args.show:
                cv2.imshow("HRM segmentation (Q to stop)", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    print(f"Saved {frames} frames to {out}; processing FPS={frames/max(elapsed, 1e-9):.1f} (excludes capture/write/display).")


@torch.inference_mode()
def benchmark(args):
    device = pick_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    rgb = cv2.imread(args.source)
    if rgb is None:
        raise ValueError(f"Unreadable benchmark image: {args.source}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    image, _, _ = letterbox(rgb, checkpoint["size"])
    tensor = to_tensor(image)[None].to(device)
    sync = lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = {}
    for kind in ("model", "pipeline"):
        samples = []
        for i in range(args.warmup + args.iterations):
            sync()
            start = time.perf_counter()
            if kind == "model":
                with torch.autocast(device.type, enabled=device.type == "cuda"):
                    model(tensor)
            else:
                infer_frame(model, rgb, checkpoint["size"], device)
            sync()
            if i >= args.warmup:
                samples.append((time.perf_counter() - start) * 1000)
        times[kind] = {"median_ms": float(np.median(samples)), "p95_ms": float(np.percentile(samples, 95)),
                       "mean_fps": 1000 / float(np.mean(samples))}
    report = {"device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
              "torch": str(torch.__version__), "size": checkpoint["size"], "batch_size": 1,
              "amp": device.type == "cuda", "iterations": args.iterations,
              "peak_allocated_mb": torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else None,
              "timing": times, "note": "Pipeline includes CPU preprocessing, transfers, model, resizing and threshold. Excludes camera/decode, overlay, disk I/O, display."}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", default="datasets")
    common.add_argument("--backgrounds", default="datasets/backgrounds")
    common.add_argument("--seed", type=int, default=42)
    runtime = argparse.ArgumentParser(add_help=False)
    runtime.add_argument("--device", default="auto")
    runtime.add_argument("--workers", type=int, default=4)
    runtime.add_argument("--batch-size", type=int, default=8)

    p = sub.add_parser("prepare", parents=[common])
    p.add_argument("--session-gap", type=float, default=2.0)
    p.add_argument("--val-group")
    p.add_argument("--test-group")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=prepare)

    p = sub.add_parser("preview", parents=[common])
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--mode", choices=MODES, default="combined")
    p.add_argument("--count", type=int, default=6)
    p.add_argument("--output", default="runs/augmentation_preview.jpg")
    p.set_defaults(func=preview)

    p = sub.add_parser("export-augmented", parents=[common])
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--modes", nargs="+", choices=MODES, default=["placement", "combined"])
    p.add_argument("--count", type=int, default=50, help="Number of source images")
    p.add_argument("--variants", type=int, default=3, help="Variants per source image and mode")
    p.add_argument("--output", default="datasets/augmented_random")
    p.set_defaults(func=export_augmented)

    p = sub.add_parser("train", parents=[common, runtime])
    p.add_argument("--encoder", choices=["resnet18", "resnet34"], default="resnet18")
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--stress-samples", type=int, default=256, help="Fixed evenly spaced validation samples; 0 means all")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--output", default="runs/hrm")
    p.set_defaults(func=train)

    p = sub.add_parser("evaluate", parents=[common, runtime])
    p.add_argument("--checkpoint", default="runs/hrm/best.pt")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--modes", nargs="+", choices=["clean", *MODES[1:]], default=["clean", *MODES[1:]])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output", default="runs/evaluation.json")
    p.set_defaults(func=evaluate)

    p = sub.add_parser("predict", parents=[runtime])
    p.add_argument("--checkpoint", default="runs/hrm/best.pt")
    p.add_argument("--source", required=True, help="Image, image directory, video, or webcam number")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--show", action="store_true")
    p.add_argument("--save-masks", action="store_true")
    p.add_argument("--output", default="runs/predict")
    p.set_defaults(func=predict, seed=42)

    p = sub.add_parser("benchmark", parents=[runtime])
    p.add_argument("--checkpoint", default="runs/hrm/best.pt")
    p.add_argument("--source", required=True)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--output", default="runs/benchmark.json")
    p.set_defaults(func=benchmark, seed=42)

    args = parser.parse_args()
    for name in ("batch_size", "epochs", "count", "iterations", "variants"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_','-')} must be positive")
    for name in ("workers", "warmup", "patience", "stress_samples", "limit", "max_frames"):
        if hasattr(args, name) and getattr(args, name) < 0:
            parser.error(f"--{name.replace('_','-')} must be nonnegative")
    for name in ("lr", "session_gap"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_','-')} must be positive")
    if hasattr(args, "size") and (args.size < 64 or args.size % 32):
        parser.error("--size must be a multiple of 32 and at least 64")
    if hasattr(args, "threshold") and not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    args.func(args)


if __name__ == "__main__":
    main()
