#!/usr/bin/env python
"""Object-insertion ("FG fusion") metrics: object identity preservation and image quality.

Self-insertion protocol: each model put the object of <img_dir>/<id> (where <mask_dir>/<id> > 127) back into the clean
background, so the real photo is the ground truth. For each prediction <pred_dir>/<id><pred_suffix>:

Object identity (reference object vs inserted object, both on a white background):
    reference = the photo inside the mask, white elsewhere, cropped to the mask's bounding box;
    generated = the prediction inside the same mask (--gen_fg mask, default), white elsewhere, same crop; or the
                plain bounding-box crop of the prediction (--gen_fg bbox, SHINE's evaluation.py protocol).
    Both crops are padded to a square with white so the encoders' center crops don't cut the object.
    - CLIP-I   cosine similarity of CLIP ViT-B/32 image embeddings (SHINE tools/clip_score.py; HF weights)
    - DINOv2   SHINE tools/dinov2_score.py: facebook/dinov2-base, mean of last_hidden_state, cosine, (sim + 1) / 2
    - CUTE     foreground feature averaging (s-tian/CUTE foreground_feature_averaging.py, variant Crop-Feat): carvekit
               Tracer-B7 background removal, DINOv2 ViT-B/14 patch tokens at 336 px averaged over the foreground, cosine
    - DreamSim distance of the DreamSim ensemble (dreamsim(pretrained=True), as SHINE's evaluation.py); lower is better
Image quality (full prediction vs the real photo, at the photo's native resolution, PyIQA):
    - PSNR (RGB), SSIM (Y channel, PyIQA default), LPIPS (AlexNet), FID (set of predictions vs set of photos)

Several --pred_dir (models of the same dataset) can be given in one call; the reference embeddings are computed once.
Writes <pred_dir>/<out_name>.json (settings + summary) and .csv (per image) for each.

Weights: CLIP and DINOv2 come from the Hugging Face cache (HF_HOME); pyiqa and torch.hub use TORCH_HOME; DreamSim and
carvekit checkpoints go under --cache_dir.
"""

import argparse
import csv
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.environ.get("LAYER_FUSION_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "layer_fusion"))

import numpy as np
import pyiqa
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
WHITE = (255, 255, 255)
CLIP_I_MODEL = "openai/clip-vit-base-patch32"
DINOV2_MODEL = "facebook/dinov2-base"
IDENTITY = ("clip_i", "dinov2", "cute", "dreamsim")
QUALITY = ("psnr", "ssim", "lpips")


def find(d, s):
    for ext in IMG_EXTS:
        p = os.path.join(d, s + ext)
        if os.path.exists(p):
            return p
    return None


def fg_on_white(image, mask, bbox):
    """image inside mask (bool HxW), white elsewhere, cropped to bbox and padded to a white square."""
    a = np.where(mask[..., None], np.asarray(image), np.uint8(255)).astype(np.uint8)
    return pad_square(Image.fromarray(a).crop(bbox))


def pad_square(img):
    side = max(img.size)
    out = Image.new("RGB", (side, side), WHITE)
    out.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return out


class Identity:
    """Embeddings for the four identity metrics; similarity()/distance() compare two embedding dicts."""

    def __init__(self, device, cache_dir, cute=True):
        from dreamsim import dreamsim
        from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor, CLIPModel
        self.device = device
        self.clip = CLIPModel.from_pretrained(CLIP_I_MODEL).to(device).eval()
        self.clip_proc = CLIPImageProcessor.from_pretrained(CLIP_I_MODEL)
        self.dino = AutoModel.from_pretrained(DINOV2_MODEL).to(device).eval()
        self.dino_proc = AutoImageProcessor.from_pretrained(DINOV2_MODEL)
        self.dreamsim, self.dreamsim_pre = dreamsim(pretrained=True, device=device,
                                                    cache_dir=os.path.join(cache_dir, "dreamsim"))
        self.ffa = None
        if cute:
            import pathlib
            import carvekit.ml.files
            # carvekit hardcodes ~/.cache/carvekit/checkpoints; redirect it before its loaders import the path
            carvekit.ml.files.checkpoints_dir = pathlib.Path(cache_dir, "carvekit")
            sys.path.insert(0, HERE)
            from cute_ffa import ForegroundFeatureAveraging
            self.ffa = ForegroundFeatureAveraging(device)

    @torch.no_grad()
    def embed(self, img):
        x = self.clip_proc(images=img, return_tensors="pt")["pixel_values"].to(self.device)
        e = self.clip.get_image_features(pixel_values=x)
        out = {"clip_i": getattr(e, "pooler_output", e)[0]}
        x = self.dino_proc(images=img, return_tensors="pt")["pixel_values"].to(self.device)
        out["dinov2"] = self.dino(pixel_values=x).last_hidden_state.mean(dim=1)[0]
        out["dreamsim"] = self.dreamsim.embed(self.dreamsim_pre(img).to(self.device))[0]
        if self.ffa is not None:
            # carvekit rejects images under 32 px (tiny objects); FFA resizes to 336 px anyway
            small = min(img.size) < 64
            x = img.resize((max(64, img.width), max(64, img.height)), Image.Resampling.BICUBIC) if small else img
            out["cute"] = self.ffa.forward_single([x], "Crop-Feat")[0]
        return out

    @staticmethod
    def compare(a, b):
        cos = {k: float(F.cosine_similarity(a[k], b[k], dim=0)) for k in a}
        row = {"clip_i": cos["clip_i"], "dinov2": (cos["dinov2"] + 1) / 2, "dreamsim": 1 - cos["dreamsim"]}
        if "cute" in cos:
            row["cute"] = cos["cute"]
        return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, nargs="+", help="one or more model output folders (same dataset)")
    ap.add_argument("--img_dir", required=True, help="real photos (ground truth)")
    ap.add_argument("--mask_dir", required=True, help="object masks (white = object)")
    ap.add_argument("--pred_suffix", default="_comp.png")
    ap.add_argument("--gen_fg", choices=("mask", "bbox"), default="mask",
                    help="generated object: prediction inside the GT mask on white, or plain bbox crop (SHINE)")
    ap.add_argument("--out_name", default="metrics_insertion")
    ap.add_argument("--save_fg", action="store_true", help="also write the FG crops to <pred_dir>/fg_eval/")
    ap.add_argument("--no_cute", action="store_true")
    ap.add_argument("--no_fid", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache_dir", default=DEFAULT_CACHE, help="DreamSim / carvekit checkpoints (env LAYER_FUSION_CACHE)")
    args = ap.parse_args()
    device = torch.device(args.device)

    photos = {os.path.splitext(f)[0]: os.path.join(args.img_dir, f)
              for f in sorted(os.listdir(args.img_dir)) if f.lower().endswith(IMG_EXTS)}
    if args.limit:
        photos = dict(list(photos.items())[:args.limit])
    ident = Identity(device, args.cache_dir, cute=not args.no_cute)
    quality = {n: pyiqa.create_metric(n, device=device) for n in QUALITY}
    fid = None if args.no_fid else pyiqa.create_metric("fid", device=device)
    fields = [n for n in IDENTITY if n != "cute" or not args.no_cute] + list(QUALITY)

    # Reference objects (shared by all models).
    refs = {}
    for s, p in tqdm(photos.items(), desc="reference objects", mininterval=10):
        photo = Image.open(p).convert("RGB")
        m = np.asarray(Image.open(find(args.mask_dir, s)).convert("L").resize(photo.size, Image.Resampling.NEAREST)) > 127
        if not m.any():
            print(f"[warn] {s}: empty mask, skipped", flush=True)
            continue
        ys, xs = np.nonzero(m)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        refs[s] = {"mask": m, "bbox": bbox, "emb": ident.embed(fg_on_white(photo, m, bbox))}

    for pred_dir in args.pred_dir:
        preds = {f[:-len(args.pred_suffix)]: os.path.join(pred_dir, f)
                 for f in os.listdir(pred_dir) if f.endswith(args.pred_suffix)}
        ids = [s for s in refs if s in preds]
        missing = [s for s in refs if s not in preds]
        print(f"== {pred_dir}: {len(ids)} predictions" + (f", {len(missing)} missing e.g. {missing[:3]}" if missing else ""),
              flush=True)
        if not ids:
            continue
        if args.save_fg:
            os.makedirs(os.path.join(pred_dir, "fg_eval"), exist_ok=True)
        rows, resized = [], 0
        with tempfile.TemporaryDirectory(prefix="insert_eval_") as tmp:
            fid_pred, fid_gt = os.path.join(tmp, "pred"), os.path.join(tmp, "gt")
            os.makedirs(fid_pred)
            os.makedirs(fid_gt)
            for s in tqdm(ids, desc=os.path.basename(os.path.normpath(pred_dir)), mininterval=10):
                photo = Image.open(photos[s]).convert("RGB")
                pred = Image.open(preds[s]).convert("RGB")
                if pred.size != photo.size:
                    pred, resized = pred.resize(photo.size, Image.Resampling.LANCZOS), resized + 1
                r = refs[s]
                gen = fg_on_white(pred, r["mask"], r["bbox"]) if args.gen_fg == "mask" else pad_square(pred.crop(r["bbox"]))
                row = {"id": s, **Identity.compare(r["emb"], ident.embed(gen))}
                with torch.no_grad():
                    p = TF.to_tensor(pred).unsqueeze(0).to(device)
                    g = TF.to_tensor(photo).unsqueeze(0).to(device)
                    row.update({n: float(m(p, g).item()) for n, m in quality.items()})
                rows.append(row)
                if args.save_fg:
                    gen.save(os.path.join(pred_dir, "fg_eval", f"{s}_gen.png"))
                    fg_on_white(photo, r["mask"], r["bbox"]).save(os.path.join(pred_dir, "fg_eval", f"{s}_ref.png"))
                os.symlink(os.path.abspath(preds[s]), os.path.join(fid_pred, f"{s}.png"))
                os.symlink(os.path.abspath(photos[s]), os.path.join(fid_gt, os.path.basename(photos[s])))
            summary = {n: {"mean": float(np.nanmean([r[n] for r in rows])), "std": float(np.nanstd([r[n] for r in rows]))}
                       for n in fields}
            if fid is not None:
                summary["fid"] = float(fid(fid_pred, fid_gt).item())

        result = {
            "pred_dir": os.path.abspath(pred_dir), "img_dir": os.path.abspath(args.img_dir),
            "mask_dir": os.path.abspath(args.mask_dir), "num_images": len(rows), "num_missing_predictions": len(missing),
            "settings": {
                "gen_fg": args.gen_fg, "predictions_resized_to_photo": resized,
                "identity_input": "object on white, cropped to the GT mask bbox, padded to a white square",
                "metric_notes": {
                    "clip_i": f"cosine of {CLIP_I_MODEL} image embeddings (SHINE recipe)",
                    "dinov2": f"{DINOV2_MODEL} mean last_hidden_state, (cosine + 1) / 2 (SHINE recipe)",
                    "cute": "CUTE foreground feature averaging, Crop-Feat (carvekit Tracer-B7 + DINOv2 ViT-B/14 @336)",
                    "dreamsim": "DreamSim ensemble distance (lower is better)",
                    "psnr/ssim/lpips/fid": f"pyiqa {pyiqa.__version__}, full prediction vs real photo, native resolution",
                },
            },
            "summary": summary,
        }
        out = os.path.join(pred_dir, args.out_name)
        with open(f"{out}.json", "w") as f:
            json.dump(result, f, indent=2)
        with open(f"{out}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", *fields])
            w.writeheader()
            w.writerows(rows)
        print(json.dumps({"num_images": len(rows), **{k: (v["mean"] if isinstance(v, dict) else v)
                                                       for k, v in summary.items()}}, indent=2), flush=True)
        print(f"saved {out}.json and {out}.csv", flush=True)


if __name__ == "__main__":
    main()
