# AUTHOR: Daeyun - HRM multiple-occlusion training and comparison workflow.
"""Train and compare visible-body segmentation with projective/gap augmentation.

RGB-only appearance test. No warped RGB is reused with unchanged physical depth.
The first pilot remains intact. Current test sequence has already been examined
in the first pilot: report as development stress test, NOT a new blind test.
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
from feasibility_augmentation_v2 import augment_v2, make_case_v2, CONDITIONS_V2 as CONDITIONS
from run_feasibility import spaced, source_seed, digest, metrics, UNetBackend, labeled_tile


def dump(path, obj):
    hrm.write_json(Path(path), obj)


class Training(Dataset):
    def __init__(self, records, size):
        self.records, self.size = records, size
        self.backgrounds = background_paths("train")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image, mask = read_pair(self.records[index])
        rng = np.random.default_rng(np.random.randint(0, 2**32-1))
        image, mask, _, _ = augment_v2(image, mask, rng, self.backgrounds)
        image, mask, _ = hrm.letterbox(image, self.size, mask)
        return hrm.to_tensor(image), torch.from_numpy(mask[None].astype(np.float32))


class Validation(Dataset):
    def __init__(self, records, size, seed):
        self.items = []
        bgs = background_paths("val")
        self.conditions = ["clean", "multi3_30", "bg_perspective_multi3"]
        for record in records:
            image, mask = read_pair(record)
            for condition in self.conditions:
                rgb, target, _, _ = make_case_v2(image, mask, source_seed(record, seed), condition, bgs)
                rgb, target, _ = hrm.letterbox(rgb, size, target)
                self.items.append((hrm.to_tensor(rgb), torch.from_numpy(target[None].astype(np.float32))))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def train(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    records = load_records()
    train_records = spaced(records["train"], args.train_limit)
    val_records = spaced(records["val"], args.val_limit)
    data = Training(train_records, args.size)
    val = Validation(val_records, args.size, args.seed)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=4,
                        pin_memory=True, persistent_workers=True, worker_init_fn=hrm.worker_seed,
                        generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val, batch_size=args.batch_size)
    device = torch.device(args.device)
    model = hrm.build_model("resnet18", pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    config = {k:v for k,v in vars(args).items() if k != "func"}
    config.update(training_images=len(train_records), validation_images=len(val_records),
                  validation_conditions=val.conditions, checkpoint_selection="validation micro foreground Dice",
                  test_role="previously examined development stress test, not blind final test",
                  label_provenance=LABEL_PROVENANCE,
                  comparison_caveat="V1 and V2 differ in training sample count/budget and augmentation; not augmentation-only ablation",
                  torch=str(torch.__version__), gpu=torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
                  source_hashes={name:digest(Path(name)) for name in
                                 ("run_occlusion_v2.py", "feasibility_augmentation_v2.py", "feasibility_data.py", "hrm.py")})
    dump(output/"config.json", config)
    dump(output/"split.json", {"train":train_records,"val":val_records})
    best, started = -1., time.perf_counter()
    print(json.dumps({"start":"v2","train":len(data),"validation_examples":len(val)}), flush=True)
    with (output/"history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch","train_loss","val_dice","val_iou","seconds"])
        writer.writeheader()
        for epoch in range(1,args.epochs+1):
            tick = time.perf_counter()
            model.train()
            loss_sum = 0.
            for image, target in loader:
                image, target = image.to(device,non_blocking=True), target.to(device,non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=device.type == "cuda"):
                    loss = hrm.loss_fn(model(image), target)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item()*len(image)
            scheduler.step()
            score = hrm.score_model(model,val_loader,device)
            row = dict(epoch=epoch,train_loss=loss_sum/len(data),val_dice=score["dice"],val_iou=score["iou"],seconds=time.perf_counter()-tick)
            writer.writerow(row); stream.flush()
            print(json.dumps(row), flush=True)
            if score["dice"] > best:
                best = score["dice"]
                torch.save(dict(model=model.state_dict(),encoder="resnet18",size=args.size,epoch=epoch,
                                threshold=.5,config=config,training_images=[r["rgb"] for r in train_records],
                                validation=score),output/"best.pt")
    dump(output/"completed.json",dict(best_validation_dice=best,seconds=time.perf_counter()-started,
                                      checkpoint_sha256=digest(output/"best.pt")))


def evaluate(args):
    from feasibility_backends import OpenCVBaseline
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    records = spaced(load_records()["test"],args.limit)
    methods = {"opencv":OpenCVBaseline(),"unet_v1":UNetBackend(args.v1,args.device),"unet_v2":UNetBackend(args.v2,args.device)}
    for name, backend in methods.items():
        if isinstance(backend,UNetBackend):
            seen={Path(p).name for p in backend.checkpoint["training_images"]}
            if seen & {Path(r["rgb"]).name for r in records}:
                raise ValueError("Training / test source overlap")
    warm,_=read_pair(load_records()["val"][0])
    for backend in methods.values():
        backend.predict_rgb(warm)
    results, cases = [], []
    examples = set(np.linspace(0,len(records)-1,5,dtype=int))
    with (output/"per_frame.csv").open("w",newline="") as stream:
        writer = None
        for i,record in enumerate(records):
            original,mask=read_pair(record)
            for condition in CONDITIONS:
                rgb,target,meta,occlusion=make_case_v2(original,mask,source_seed(record,args.seed),condition,background_paths("test"))
                cases.append(dict(source=record,metadata=meta,occluder_sha256=hashlib.sha256(occlusion.tobytes()).hexdigest()))
                tiles=[labeled_tile(rgb,np.zeros_like(target),condition),labeled_tile(rgb,target,"Visible pseudo-label")]
                for name,backend in methods.items():
                    status="ok"
                    if args.device.startswith("cuda"): torch.cuda.synchronize()
                    tick=time.perf_counter()
                    try:
                        pred=np.asarray(backend.predict_rgb(rgb))
                        if pred.shape != target.shape: raise ValueError("Prediction size mismatch")
                        if not np.isin(pred,[0,1]).all(): raise ValueError("Prediction must be finite binary 0/1")
                    except Exception as exc:
                        status=f"{type(exc).__name__}: {exc}"
                        pred=np.zeros_like(target)
                    if args.device.startswith("cuda"): torch.cuda.synchronize()
                    elapsed_ms=(time.perf_counter()-tick)*1000
                    m=metrics(pred,target)
                    row=dict(method=name,condition=condition,source=Path(record["rgb"]).name,status=status,
                             latency_ms=elapsed_ms,**m)
                    if writer is None:
                        writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                    writer.writerow(row);results.append(row)
                    if i in examples:
                        tiles.append(labeled_tile(rgb,pred,f"{name} D={m['dice']:.3f}",target))
                if i in examples:
                    hrm.save_image(output/f"{condition}_example_{i:03d}.jpg", cv2.cvtColor(np.hstack(tiles),cv2.COLOR_RGB2BGR))
            if (i+1)%16==0:
                stream.flush(); print(f"Segmentation {i+1}/{len(records)}",flush=True)
    summary=[]
    for name in methods:
        for condition in CONDITIONS:
            group=[r for r in results if r["method"]==name and r["condition"]==condition]
            row=dict(method=name,condition=condition,frames=len(group),errors=sum(r["status"]!="ok" for r in group))
            for metric in ["dice","iou","precision","recall","boundary_f1_2px","minimum_component_recall"]:
                values=[r[metric] for r in group]
                row[metric+"_mean"]=float(np.mean(values));row[metric+"_p10"]=float(np.percentile(values,10))
            row["components_recall80_rate"]=float(np.mean([r["all_visible_components_recall_ge_80"] for r in group]))
            row["latency_median_ms"]=float(np.median([r["latency_ms"] for r in group]))
            row["latency_p95_ms"]=float(np.percentile([r["latency_ms"] for r in group],95))
            summary.append(row)
    with (output/"summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    dump(output/"summary.json",dict(records=len(records),results=summary,
        label_provenance=LABEL_PROVENANCE,test_role="development stress test previously examined in pilot1; no blind accuracy claim",
        model_hashes={"v1":digest(args.v1),"v2":digest(args.v2)},
        caveats=["RGB homography augmentation only, not a true new 3D camera view or wide-angle lens model",
                 "Different training budgets; this is an operational comparison, not an augmentation-only causal ablation"]))
    dump(output/"cases.json",cases)
    print(json.dumps(dict(output=str(output),rows=len(results))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("train")
    p.add_argument("--output",required=True);p.add_argument("--epochs",type=int,default=16)
    p.add_argument("--train-limit",type=int,default=1600);p.add_argument("--val-limit",type=int,default=96)
    p.add_argument("--batch-size",type=int,default=8);p.add_argument("--size",type=int,default=384)
    p.set_defaults(func=train)
    p=sub.add_parser("evaluate")
    p.add_argument("--output",required=True);p.add_argument("--v1",required=True);p.add_argument("--v2",required=True)
    p.add_argument("--limit",type=int,default=128);p.set_defaults(func=evaluate)
    for p in sub.choices.values():
        p.add_argument("--device",default="cuda");p.add_argument("--seed",type=int,default=173)
    args=parser.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(4);cv2.setNumThreads(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed);torch.backends.cudnn.benchmark=True
    args.func(args)


if __name__=="__main__":
    main()
