# AUTHOR: Daeyun - Matched-condition HRM segmentation training and evaluation.
"""Reproducible, offline visible-HRM segmentation feasibility experiment.

Pilot labels are OpenCV pseudo-labels. Test is disjoint from the current train
manifest, not a claim of historically untouched or clinical ground truth.
No ROS, motor interfaces, or source-image writes are used here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
import zlib

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / ".ultralytics"))
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import hrm
from feasibility_data import (ROOT, CONDITIONS, LABEL_PROVENANCE, augment_pair,
                              background_paths, load_records, make_test_case, read_pair)


def dump(path, value):
    hrm.write_json(Path(path), value)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spaced(records, count):
    if not count or count >= len(records):
        return records
    return [records[i] for i in np.linspace(0, len(records)-1, count, dtype=int)]


def source_seed(record, seed):
    return seed + zlib.crc32(Path(record["rgb"]).name.encode())


class TrainDataset(Dataset):
    def __init__(self, records, size, mode, backgrounds):
        self.records, self.size, self.mode, self.backgrounds = records, size, mode, backgrounds

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image, mask = read_pair(self.records[index])
        rng = np.random.default_rng(np.random.randint(0, 2**32-1))
        image, mask, _ = augment_pair(image, mask, rng, self.mode, self.backgrounds)
        image, target, _ = hrm.letterbox(image, self.size, mask)
        return hrm.to_tensor(image), torch.from_numpy(target[None].astype(np.float32))


class FixedDataset(Dataset):
    def __init__(self, records, size, conditions, backgrounds, seed):
        self.items = []
        for record in records:
            image, mask = read_pair(record)
            for condition in conditions:
                x, y, _ = make_test_case(image, mask, source_seed(record, seed), condition, backgrounds)
                x, y, _ = hrm.letterbox(x, size, y)
                self.items.append((hrm.to_tensor(x), torch.from_numpy(y[None].astype(np.float32))))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def train(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    all_records = load_records()
    records = spaced(all_records["train"], args.train_limit)
    val = spaced(all_records["val"], args.val_limit)
    mode = "basic" if args.mode == "basic" else "train"
    dataset = TrainDataset(records, args.size, mode, background_paths("train"))
    batches = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.workers, pin_memory=True,
                         worker_init_fn=hrm.worker_seed, persistent_workers=args.workers > 0,
                         generator=torch.Generator().manual_seed(args.seed))
    validation = FixedDataset(val, args.size, ("clean", "mid_occlusion_bg_20"),
                              background_paths("val"), args.seed)
    val_batches = DataLoader(validation, batch_size=args.batch_size, num_workers=0)
    device = torch.device(args.device)
    model = hrm.build_model("resnet18", pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    config = {k: v for k, v in vars(args).items() if k != "func"}
    config.update(label_provenance=LABEL_PROVENANCE,
                  selection="micro Dice on fixed equal-count clean + mid_occlusion_bg_20 validation; never test",
                  train_images=len(records), val_images=len(val),
                  device_name=torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
                  torch=str(torch.__version__), numpy=np.__version__, opencv=cv2.__version__,
                  source_hashes={name:digest(ROOT/name) for name in
                                 ("run_feasibility.py", "feasibility_data.py", "augment.py", "hrm.py")})
    dump(out/"config.json", config)
    dump(out/"split.json", {"train":records, "val":val,
                            "test_note":"45deg_1 remains excluded from training and checkpoint selection"})
    print(json.dumps({"start":args.mode,"train":len(records),"val":len(val),"device":config["device_name"]}), flush=True)
    best = -1.
    start = time.perf_counter()
    with (out/"history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_dice","val_iou","seconds"])
        writer.writeheader()
        for epoch in range(1, args.epochs+1):
            epoch_start = time.perf_counter()
            model.train()
            total_loss = 0.
            for x, y in batches:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=device.type == "cuda"):
                    loss = hrm.loss_fn(model(x), y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()*len(x)
            scheduler.step()
            metrics = hrm.score_model(model, val_batches, device)
            row = dict(epoch=epoch, train_loss=total_loss/len(dataset), val_dice=metrics["dice"],
                       val_iou=metrics["iou"], seconds=time.perf_counter()-epoch_start)
            writer.writerow(row); f.flush()
            print(json.dumps(row), flush=True)
            if metrics["dice"] > best:
                best = metrics["dice"]
                torch.save(dict(model=model.state_dict(), encoder="resnet18",size=args.size,
                                epoch=epoch,threshold=.5,config=config,
                                training_images=[r["rgb"] for r in records],validation=metrics), out/"best.pt")
    dump(out/"completed.json", {"epochs":args.epochs,"best_validation_dice":best,
                               "elapsed_seconds":time.perf_counter()-start,
                               "checkpoint_sha256":digest(out/"best.pt")})


def metrics(prediction, target):
    p, t = prediction.astype(bool), target.astype(bool)
    tp, fp, fn = int((p&t).sum()), int((p&~t).sum()), int((~p&t).sum())
    safe = lambda a,b: float(a/b) if b else 1.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(t.astype(np.uint8), 8)
    rec = []
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA] >= 32:
            region = labels==i
            rec.append(float(p[region].mean()))
    pb = p.astype(np.uint8)-cv2.erode(p.astype(np.uint8), np.ones((3,3),np.uint8),
                                    borderType=cv2.BORDER_CONSTANT,borderValue=0)
    tb = t.astype(np.uint8)-cv2.erode(t.astype(np.uint8), np.ones((3,3),np.uint8),
                                    borderType=cv2.BORDER_CONSTANT,borderValue=0)
    bprecision = safe(int((pb & cv2.dilate(tb,np.ones((5,5),np.uint8))).sum()),int(pb.sum()))
    brecall = safe(int((tb & cv2.dilate(pb,np.ones((5,5),np.uint8))).sum()),int(tb.sum()))
    return dict(dice=safe(2*tp,2*tp+fp+fn),iou=safe(tp,tp+fp+fn),
                precision=safe(tp,tp+fp),recall=safe(tp,tp+fn),
                boundary_f1_2px=(2*bprecision*brecall/(bprecision+brecall)
                                 if bprecision+brecall else 0.),
                tp=tp,fp=fp,fn=fn,predicted_pixels=int(p.sum()),target_pixels=int(t.sum()),
                visible_components=len(rec),minimum_component_recall=min(rec) if rec else 1.,
                all_visible_components_recall_ge_80=bool(rec and min(rec)>=.8))


class UNetBackend:
    def __init__(self,path,device):
        self.device=torch.device(device)
        self.model,self.checkpoint=hrm.load_checkpoint(path,self.device)
        self.path=str(path)
    def predict_rgb(self,rgb):
        return hrm.infer_frame(self.model,rgb,self.checkpoint["size"],self.device,.5)


def labeled_tile(rgb,mask,label,truth=None):
    view=rgb.copy()
    if truth is None:
        view=hrm.overlay(view,mask)
    else:
        p,t=mask.astype(bool),truth.astype(bool)
        # TP green, FP red, FN blue. Real input is retained underneath.
        for region,color in ((p&t,[0,220,60]),(p&~t,[255,50,30]),(~p&t,[30,100,255])):
            view[region]=(.45*view[region]+.55*np.array(color)).astype(np.uint8)
    view=cv2.copyMakeBorder(view,32,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
    cv2.putText(view,label,(6,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
    return view


def evaluate(args):
    from feasibility_backends import OpenCVBaseline, UltralyticsSAM, SAMUNetRefinement
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    records=spaced(load_records()["test"],args.limit)
    backends={"opencv":OpenCVBaseline(),"unet_basic":UNetBackend(args.basic,args.device),
              "unet_robust":UNetBackend(args.robust,args.device)}
    for backend in backends.values():
        if isinstance(backend,UNetBackend):
            seen={Path(p).name for p in backend.checkpoint["training_images"]}
            if seen & {Path(r["rgb"]).name for r in records}:
                raise RuntimeError("Train/test overlap")
    if args.sam:
        backends["mobile_sam_roi"]=UltralyticsSAM(args.sam,device="0" if "cuda" in args.device else "cpu")
        backends["unet_robust_mobile_sam"]=SAMUNetRefinement(args.sam,device="0" if "cuda" in args.device else "cpu")
    paths=background_paths("test")
    rows=[]; generated=[]; errors=[]
    example_rows=defaultdict(list)
    example_ids=set(np.linspace(0,len(records)-1,min(5,len(records)),dtype=int).tolist())
    conditions=args.conditions or list(CONDITIONS)
    # Warm up without ground truth and outside inference timing. No test-based tuning.
    warm_rgb,_=read_pair(load_records()["val"][0])
    for name,backend in backends.items():
        if name=="unet_robust_mobile_sam":
            backend.predict_rgb(warm_rgb,prompt_mask=backends["unet_robust"].predict_rgb(warm_rgb))
        else:
            backend.predict_rgb(warm_rgb)
    for index,record in enumerate(records):
        rgb,full_target=read_pair(record)
        for condition in conditions:
            image,target,meta=make_test_case(rgb,full_target,source_seed(record,args.seed),condition,paths)
            generated.append(dict(source=record,**meta))
            tiles=[labeled_tile(image,np.zeros_like(target),condition),
                   labeled_tile(image,target,"Visible pseudo-label")]
            predictions={}; latencies={}
            for name,backend in backends.items():
                status="ok"
                if "cuda" in args.device: torch.cuda.synchronize()
                start=time.perf_counter()
                try:
                    if name=="unet_robust_mobile_sam":
                        pred=backend.predict_rgb(image,prompt_mask=predictions["unet_robust"])
                    else:
                        pred=backend.predict_rgb(image)
                    if pred.shape!=target.shape: raise ValueError("Prediction shape mismatch")
                except Exception as e:
                    status=f"error:{type(e).__name__}:{e}"
                    errors.append(dict(method=name,frame=Path(record["rgb"]).name,condition=condition,error=status))
                    pred=np.zeros_like(target)
                if "cuda" in args.device: torch.cuda.synchronize()
                ms=(time.perf_counter()-start)*1000
                if name=="unet_robust_mobile_sam": ms+=latencies["unet_robust"]
                predictions[name]=pred;latencies[name]=ms
                m=metrics(pred,target)
                rows.append(dict(method=name,condition=condition,source=Path(record["rgb"]).name,
                                 timestamp_ns=record["frame_timestamp_ns"],status=status,latency_ms=ms,
                                 actual_centerline_occlusion_fraction=meta["actual_centerline_occlusion_fraction"],
                                 foreground_occlusion_fraction=meta["foreground_occlusion_fraction"],**m))
                if index in example_ids:
                    tiles.append(labeled_tile(image,pred,f"{name} D={m['dice']:.3f}",target))
            if index in example_ids:
                example_rows[condition].append(np.hstack(tiles))
        if (index+1)%8==0: print(f"Evaluated {index+1}/{len(records)} frames x {len(conditions)} conditions",flush=True)
    with (out/"per_frame.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    aggregate=[]
    for name in backends:
        for condition in conditions:
            group=[r for r in rows if r["method"]==name and r["condition"]==condition]
            a=dict(method=name,condition=condition,frames=len(group),errors=sum(r["status"]!="ok" for r in group))
            for field in ("dice","iou","precision","recall","boundary_f1_2px","minimum_component_recall"):
                values=np.array([r[field] for r in group]);a[field+"_mean"]=float(values.mean());a[field+"_p10"]=float(np.percentile(values,10))
            a["all_visible_components_recall_ge_80_rate"]=float(np.mean([r["all_visible_components_recall_ge_80"] for r in group]))
            a["dice_below_80_rate"]=float(np.mean([r["dice"]<.8 for r in group]))
            a["latency_median_ms"]=float(np.median([r["latency_ms"] for r in group]))
            a["latency_p95_ms"]=float(np.percentile([r["latency_ms"] for r in group],95))
            a["actual_occlusion_mean"]=float(np.mean([r["actual_centerline_occlusion_fraction"] for r in group]))
            aggregate.append(a)
    with (out/"summary.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(aggregate[0]));writer.writeheader();writer.writerows(aggregate)
    dump(out/"summary.json",dict(label_provenance=LABEL_PROVENANCE,records=len(records),conditions=conditions,
                                results=aggregate,errors=errors,
                                caveats=["one held-out-candidate recording; historical development exposure unknown",
                                         "pseudo-label agreement, not independent segmentation accuracy",
                                         "RGB synthetic occlusion only, not 3D reconstruction or physical occlusion",
                                         "latency excludes camera/disk; hybrid includes UNet + SAM",
                                         "visible-component recall is not centerline reconstruction success"]))
    dump(out/"cases.json",generated)
    for condition,images in example_rows.items():
        # Keep full-resolution per-frame sheet plus a compact condition overview.
        for i,sheet in enumerate(images):
            hrm.save_image(out/f"{condition}_example_{i+1}.jpg",cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR))
        overview=np.vstack([cv2.resize(im,None,fx=.55,fy=.55) for im in images])
        hrm.save_image(out/f"{condition}_overview.jpg",cv2.cvtColor(overview,cv2.COLOR_RGB2BGR))
    print(json.dumps({"output":str(out),"rows":len(rows),"errors":len(errors)}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    p=commands.add_parser("train")
    p.add_argument("--mode",choices=["basic","robust"],required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--epochs",type=int,default=12)
    p.add_argument("--train-limit",type=int,default=1200)
    p.add_argument("--val-limit",type=int,default=64)
    p.add_argument("--size",type=int,default=384)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--lr",type=float,default=3e-4)
    p.set_defaults(func=train)
    p=commands.add_parser("evaluate")
    p.add_argument("--basic",required=True);p.add_argument("--robust",required=True)
    p.add_argument("--sam");p.add_argument("--output",required=True)
    p.add_argument("--limit",type=int,default=96)
    p.add_argument("--conditions",nargs="+",choices=CONDITIONS)
    p.set_defaults(func=evaluate)
    for p in commands.choices.values():
        p.add_argument("--device",default="cuda");p.add_argument("--seed",type=int,default=73)
    args=parser.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(4);cv2.setNumThreads(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed);torch.backends.cudnn.benchmark=True
    args.func(args)


if __name__=="__main__":
    main()
