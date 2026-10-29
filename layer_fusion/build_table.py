#!/usr/bin/env python
"""Insertion comparison table from eval_insertion.py results.

Reads <runs_root>/<dataset folder>/<model folder>/<metrics_name>.json and writes <out>.{tex,md,csv}: one block of rows
per dataset; columns are the object-identity metrics (CLIP-I, DINOv2, CUTE, DreamSim) and the image-quality metrics
(PSNR, SSIM, LPIPS, FID). A model/dataset pair that is missing, or was scored on fewer images than --dataset's photo
folder holds, is shown as "–". Best value per dataset and metric in bold.

Example:
    python build_table.py --runs_root runs --out results/insertion_table \\
        --dataset "OmniPaint-Bench" OmniPaint-Bench-Part1 data/OmniPaint-Bench-Part1/img \\
        --model "Insert Anything" InsertAnything --model "MagicQuill V2" MagicQuillV2 --model Ours Ours
"""

import argparse
import csv
import json
import os

GROUPS = [  # (group title, [(key, header, higher is better, decimals)])
    ("Object Identity Preservation", [("clip_i", "CLIP-I", True, 4), ("dinov2", "DINOv2", True, 4),
                                      ("cute", "CUTE", True, 4), ("dreamsim", "DreamSim", False, 4)]),
    ("Image Quality", [("psnr", "PSNR", True, 2), ("ssim", "SSIM", True, 4), ("lpips", "LPIPS", False, 4),
                       ("fid", "FID", False, 2)]),
]
METRICS = [m for _, ms in GROUPS for m in ms]
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def load(path, n):
    if not os.path.exists(path):
        return None
    r = json.load(open(path))
    if r.get("num_images") != n:
        print(f"[warn] {path}: {r.get('num_images')} of {n} images scored, shown as missing", flush=True)
        return None
    s = r["summary"]
    return {k: (s[k]["mean"] if isinstance(s[k], dict) else s[k]) for k, *_ in METRICS if k in s}


def fmt(v, dec):
    return "–" if v is None else f"{v:.{dec}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True, help="holds <dataset folder>/<model folder>/<metrics_name>.json")
    ap.add_argument("--dataset", nargs=3, action="append", required=True, metavar=("TITLE", "FOLDER", "PHOTO_DIR"),
                    help="row-block title, its folder under runs_root, and its photo folder (for the image count)")
    ap.add_argument("--model", nargs=2, action="append", required=True, metavar=("TITLE", "FOLDER"))
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--metrics_name", default="metrics_insertion", help="eval_insertion.py --out_name")
    ap.add_argument("--label", default="tab:insertion")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    datasets, models = args.dataset, args.model

    table, best = {}, {}
    for d, folder, photo_dir in datasets:
        n = sum(f.lower().endswith(IMG_EXTS) for f in os.listdir(photo_dir))
        for m, mf in models:
            table[m, d] = load(os.path.join(args.runs_root, folder, mf, f"{args.metrics_name}.json"), n)
        for k, _, higher, _ in METRICS:
            vals = [table[m, d][k] for m, _ in models if table[m, d] and k in table[m, d]]
            best[d, k] = (max if higher else min)(vals) if vals else None

    def cell(m, d, k, dec, bold):
        v = table[m, d].get(k) if table[m, d] else None
        s = fmt(v, dec)
        return bold(s) if v is not None and v == best[d, k] else s

    arrow = lambda higher: "↑" if higher else "↓"
    with open(f"{args.out}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", *[h for _, h, _, _ in METRICS]])
        for d, *_ in datasets:
            for m, _ in models:
                w.writerow([d, m, *[fmt(table[m, d].get(k) if table[m, d] else None, dec) for k, _, _, dec in METRICS]])

    md = ["| Dataset | Model | " + " | ".join(f"{h} {arrow(hi)}" for _, h, hi, _ in METRICS) + " |",
          "|---|---|" + "---|" * len(METRICS)]
    for d, *_ in datasets:
        for i, (m, _) in enumerate(models):
            md.append(f"| {d if i == 0 else ''} | {m} | " + " | ".join(
                cell(m, d, k, dec, lambda s: f"**{s}**") for k, _, _, dec in METRICS) + " |")
    open(f"{args.out}.md", "w").write("\n".join(md) + "\n")

    # LaTeX (booktabs + multirow): one row block per dataset, column groups for identity and quality.
    nid, nq = (len(ms) for _, ms in GROUPS)
    tex = [r"\begin{table*}[t]", r"\centering", r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{ll|" + "c" * nid + "|" + "c" * nq + "}", r"\toprule",
           rf" & & \multicolumn{{{nid}}}{{c|}}{{\textbf{{{GROUPS[0][0]}}}}} & \multicolumn{{{nq}}}{{c}}{{\textbf{{{GROUPS[1][0]}}}}} \\",
           r"\textbf{Dataset} & \textbf{Model} & " + " & ".join(
               rf"\textbf{{{h}}} $\{'uparrow' if hi else 'downarrow'}$" for _, h, hi, _ in METRICS) + r" \\"]
    for d, *_ in datasets:
        tex.append(r"\midrule")
        for i, (m, _) in enumerate(models):
            first = rf"\multirow{{{len(models)}}}{{*}}{{{d}}}" if i == 0 else ""
            cells = [cell(m, d, k, dec, lambda s: rf"\textbf{{{s}}}").replace("–", "--") for k, _, _, dec in METRICS]
            tex.append(f"{first} & {m}\n  & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}}",
            r"\caption{Object insertion (self-insertion: the object is inserted back into the clean background, so the "
            r"real photo is the ground truth). Object identity compares the inserted object with the reference object, "
            r"both inside the object mask on a white background and cropped to its bounding box: CLIP-I (CLIP ViT-B/32 "
            r"cosine), DINOv2 (DINOv2-B mean-token cosine, rescaled to $[0,1]$), CUTE (foreground feature averaging) and "
            r"DreamSim distance. Image quality compares the full result with the real photo. Best per dataset in bold.}",
            rf"\label{{{args.label}}}", r"\end{table*}"]
    open(f"{args.out}.tex", "w").write("\n".join(tex) + "\n")
    print("\n".join(md))
    print(f"\nsaved {args.out}.{{tex,md,csv}}", flush=True)


if __name__ == "__main__":
    main()
