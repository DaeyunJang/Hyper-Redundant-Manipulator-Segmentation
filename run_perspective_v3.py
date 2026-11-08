# AUTHOR: Daeyun - Matched-budget HRM perspective-augmentation experiment.
"""Matched-budget V3 planar depth-tilt augmentation experiment.

Never pairs projectively warped RGB with original camera depth. Exported assets
are paper-auditable visible-mask development tests, not novel physical views.
V2 source records, validation conditions and optimizer budget are held fixed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("MPLBACKEND", "Agg")
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import hrm
from feasibility_data import load_records, read_pair, background_paths, LABEL_PROVENANCE
from feasibility_augmentation_v2 import make_case_v2
from feasibility_augmentation_v3 import augment_v3, make_case_v3, CONDITIONS_V3
from run_occlusion_v2 import Validation
from run_feasibility import spaced, source_seed, digest, metrics, UNetBackend, labeled_tile

ROOT = Path(__file__).resolve().parent
V2_RUN = ROOT / "runs/occlusion_pose_20261024"
CONDITIONS = (*CONDITIONS_V3, "multi3_30", "bg_perspective_multi3")


def dump(path, value):
    hrm.write_json(Path(path), value)


def case(rgb, mask, seed, condition, backgrounds):
    function = make_case_v3 if condition in CONDITIONS_V3 else make_case_v2
    return function(rgb, mask, seed, condition, backgrounds)


class Training(Dataset):
    def __init__(self, records, size):
        self.records, self.size = records, size
        self.backgrounds = background_paths("train")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rgb, mask = read_pair(self.records[index])
        rng = np.random.default_rng(np.random.randint(0, 2**32 - 1))
        rgb, mask, _, _ = augment_v3(rgb, mask, rng, self.backgrounds)
        rgb, mask, _ = hrm.letterbox(rgb, self.size, mask)
        return hrm.to_tensor(rgb), torch.from_numpy(mask[None].astype(np.float32))


def train(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    previous = json.loads((V2_RUN / "unet_v2/config.json").read_text())
    split = json.loads((V2_RUN / "unet_v2/split.json").read_text())
    for key in ("epochs", "batch_size", "size", "seed"):
        if getattr(args, key) != previous[key]:
            raise ValueError(f"Matched-budget run requires {key}={previous[key]}")
    data = Training(split["train"], args.size)
    val = Validation(split["val"], args.size, args.seed)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=4,
                        pin_memory=True, persistent_workers=True, worker_init_fn=hrm.worker_seed,
                        generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val, batch_size=args.batch_size)
    device = torch.device(args.device)
    model = hrm.build_model("resnet18", pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    config = {k: v for k, v in vars(args).items() if k != "func"}
    config.update(training_images=len(data), validation_images=len(split["val"]),
                  validation_conditions=val.conditions,
                  checkpoint_selection="same V2 validation micro foreground Dice; no test selection",
                  matched_to=str(V2_RUN / "unet_v2"), label_provenance=LABEL_PROVENANCE,
                  comparison_caveat="Matched data/budget/validation; one random seed, development test, not blind or multi-run evidence",
                  torch=str(torch.__version__),
                  source_hashes={name: digest(ROOT / name) for name in
                                 ("run_perspective_v3.py", "feasibility_augmentation_v3.py",
                                  "feasibility_augmentation_v2.py", "run_occlusion_v2.py",
                                  "feasibility_data.py", "hrm.py")})
    dump(output / "config.json", config)
    dump(output / "split.json", split)
    best, started = -1., time.perf_counter()
    with (output / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "val_dice", "val_iou", "seconds"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            tick = time.perf_counter()
            model.train()
            loss_sum = 0.
            for rgb, target in loader:
                rgb, target = rgb.to(device, non_blocking=True), target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=device.type == "cuda"):
                    loss = hrm.loss_fn(model(rgb), target)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item() * len(rgb)
            scheduler.step()
            score = hrm.score_model(model, val_loader, device)
            row = dict(epoch=epoch, train_loss=loss_sum / len(data), val_dice=score["dice"],
                       val_iou=score["iou"], seconds=time.perf_counter() - tick)
            writer.writerow(row)
            stream.flush()
            print(json.dumps(row), flush=True)
            if score["dice"] > best:
                best = score["dice"]
                torch.save(dict(model=model.state_dict(), encoder="resnet18", size=args.size,
                                epoch=epoch, threshold=.5, config=config,
                                training_images=[r["rgb"] for r in split["train"]], validation=score),
                           output / "best.pt")
    dump(output / "completed.json", dict(best_validation_dice=best, seconds=time.perf_counter() - started,
                                         checkpoint_sha256=digest(output / "best.pt")))


def save_png(path, pixels, rgb=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR) if rgb else pixels
    if not cv2.imwrite(str(path), encoded):
        raise IOError(f"Cannot write PNG: {path}")


def valid_prediction(prediction, shape):
    pred = np.asarray(prediction)
    if pred.shape != shape or not np.isin(pred, [0, 1]).all():
        raise ValueError("Prediction must match target shape and contain finite binary 0/1")
    return pred.astype(np.uint8)


def evaluate(args):
    from feasibility_backends import OpenCVBaseline
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    records = spaced(load_records()["test"], args.limit)
    methods = {"opencv": OpenCVBaseline(), "unet_v2": UNetBackend(args.v2, args.device),
               "unet_v3": UNetBackend(args.v3, args.device)}
    source_names = {Path(r["rgb"]).name for r in records}
    for name, backend in methods.items():
        if name != "opencv":
            if {Path(p).name for p in backend.checkpoint["training_images"]} & source_names:
                raise ValueError("Training / evaluation source overlap")
    warm, _ = read_pair(load_records()["val"][0])
    for backend in methods.values():
        backend.predict_rgb(warm)
    results, cases = [], []
    examples = set(np.linspace(0, len(records) - 1, 5, dtype=int))
    with (output / "per_frame.csv").open("w", newline="") as stream:
        writer = None
        for index, record in enumerate(records):
            original, mask = read_pair(record)
            for condition in CONDITIONS:
                seed = source_seed(record, args.seed)
                rgb, target, meta, occ = case(original, mask, seed, condition, background_paths("test"))
                folder = Path("assets") / condition / Path(record["rgb"]).stem
                save_png(output / folder / "rgb.png", rgb, rgb=True)
                save_png(output / folder / "visible_pseudo_mask.png", target * 255)
                save_png(output / folder / "synthetic_occluder.png", occ * 255)
                provenance = dict(source=record, source_rgb_sha256=digest(record["rgb"]),
                                  source_mask_sha256=digest(record["mask"]), metadata=meta,
                                  rgb_path=str(folder / "rgb.png"), mask_path=str(folder / "visible_pseudo_mask.png"),
                                  occluder_path=str(folder / "synthetic_occluder.png"),
                                  occluder_sha256=hashlib.sha256(occ.tobytes()).hexdigest(),
                                  original_depth_reused=False, prediction_paths={})
                tiles = [labeled_tile(rgb, np.zeros_like(target), condition),
                         labeled_tile(rgb, target, "Visible pseudo-label")]
                for name, backend in methods.items():
                    status = "ok"
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    tick = time.perf_counter()
                    try:
                        pred = valid_prediction(backend.predict_rgb(rgb), target.shape)
                    except Exception as error:
                        status = f"{type(error).__name__}: {error}"
                        pred = np.zeros_like(target)
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - tick) * 1000
                    prediction_path = folder / f"{name}_mask.png"
                    save_png(output / prediction_path, pred * 255)
                    provenance["prediction_paths"][name] = str(prediction_path)
                    row = dict(method=name, condition=condition, source=Path(record["rgb"]).name,
                               frame_timestamp_ns=record["frame_timestamp_ns"], status=status,
                               latency_ms=elapsed, **metrics(pred, target))
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    results.append(row)
                    if index in examples:
                        tiles.append(labeled_tile(rgb, pred, f"{name}: Dice={row['dice']:.3f}", target))
                dump(output / folder / "provenance.json", provenance)
                cases.append(provenance)
                if index in examples:
                    save_png(output / f"{condition}_example_{index:03d}.png", np.hstack(tiles), rgb=True)
            if (index + 1) % 16 == 0:
                stream.flush()
                print(f"Perspective evaluation {index + 1}/{len(records)}", flush=True)
    summary = []
    for name in methods:
        for condition in CONDITIONS:
            group = [r for r in results if r["method"] == name and r["condition"] == condition]
            row = dict(method=name, condition=condition, frames=len(group), errors=sum(r["status"] != "ok" for r in group))
            for metric in ("dice", "iou", "boundary_f1_2px", "minimum_component_recall"):
                row[metric + "_mean"] = float(np.mean([r[metric] for r in group]))
                row[metric + "_p10"] = float(np.percentile([r[metric] for r in group], 10))
            row["latency_median_ms"] = float(np.median([r["latency_ms"] for r in group]))
            summary.append(row)
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    dump(output / "cases.json", cases)
    dump(output / "summary.json", dict(records=len(records), results=summary,
         model_hashes={"v2": digest(args.v2), "v3": digest(args.v3)}, label_provenance=LABEL_PROVENANCE,
         selection="fixed five example frame indices, not ranked by model score",
         test_role="development stress test, previously examined sequence, no blind final accuracy claim",
         depth_policy="RGB/mask only: never reuse original depth for warped RGB",
         comparison="same source training/validation records, epoch budget and validation selection; single seed"))
    print(json.dumps(dict(output=str(output), rows=len(results), assets=len(cases))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("train")
    command.add_argument("--output", required=True)
    command.add_argument("--epochs", type=int, default=16)
    command.add_argument("--batch-size", type=int, default=8)
    command.add_argument("--size", type=int, default=384)
    command.set_defaults(func=train)
    command = sub.add_parser("evaluate")
    command.add_argument("--output", required=True)
    command.add_argument("--v2", default=str(V2_RUN / "unet_v2/best.pt"))
    command.add_argument("--v3", required=True)
    command.add_argument("--limit", type=int, default=128)
    command.set_defaults(func=evaluate)
    for command in sub.choices.values():
        command.add_argument("--device", default="cuda")
        command.add_argument("--seed", type=int, default=173)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    args.func(args)


if __name__ == "__main__":
    main()
