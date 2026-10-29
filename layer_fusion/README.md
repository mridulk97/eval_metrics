# Layer fusion (object insertion) evaluation

Self-contained evaluation of object insertion ("layer fusion"): how well a model keeps the **identity** of an inserted
object and how close the **composited image** is to the real photo.

## Protocol: self-insertion

Each benchmark sample has a real photo with an object, the object's mask and the clean background without the object.
The model inserts the object back into the clean background at the mask's location, so **the real photo is the
ground truth**.

```
<data_dir>/
    img/<id>.jpg     real photo with the object        (ground truth)
    mask/<id>.jpg    object mask, white = object
    gt/<id>.jpg      clean background (the model's insertion target; not used by the evaluation)
<pred_dir>/
    <id>_comp.png    the model's result (any size; resized to the photo if it differs)
```

## Metrics

**Object identity**: the reference object and the inserted object are compared as cut-outs. Each is the image inside
the GT mask on a white background, cropped to the mask's bounding box and padded to a white square, so the encoders'
center crops never cut the object. `--gen_fg bbox` instead uses the plain bounding-box crop of the result, as in
SHINE's evaluation.

| Metric | | Recipe |
|---|---|---|
| CLIP-I | ↑ | cosine similarity of CLIP ViT-B/32 image embeddings (`openai/clip-vit-base-patch32`), as SHINE's `tools/clip_score.py` |
| DINOv2 | ↑ | `facebook/dinov2-base`, mean of `last_hidden_state`, cosine rescaled to [0, 1] as `(cos + 1) / 2`, as SHINE's `tools/dinov2_score.py` |
| CUTE | ↑ | foreground feature averaging ([CUTE](https://github.com/s-tian/CUTE), variant Crop-Feat): carvekit Tracer-B7 background removal, DINOv2 ViT-B/14 patch tokens at 336 px averaged over the foreground, cosine |
| DreamSim | ↓ | DreamSim ensemble distance (`dreamsim(pretrained=True)`), as SHINE's evaluation |

**Image quality**: the full result against the real photo, at the photo's native resolution, with
[PyIQA](https://github.com/chaofengc/IQA-PyTorch).

| Metric | | Recipe |
|---|---|---|
| PSNR | ↑ | RGB |
| SSIM | ↑ | Y channel (PyIQA default) |
| LPIPS | ↓ | AlexNet |
| FID | ↓ | set of results vs set of real photos (PyIQA default Inception); noisy for small sets |

## Install

```bash
conda create -n layer_fusion_eval python=3.10 -y && conda activate layer_fusion_eval
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install --no-deps carvekit==4.1.2 && pip install loguru requests typer click
```

Weights download on first use: CLIP and DINOv2 to the Hugging Face cache (`HF_HOME`), PyIQA and torch.hub models to
`TORCH_HOME`, DreamSim and carvekit checkpoints to `--cache_dir` (default `$LAYER_FUSION_CACHE` or
`~/.cache/layer_fusion`).

## Usage

Score one or more models on one dataset (the reference cut-outs are computed once and shared):

```bash
python eval_insertion.py \
    --pred_dir runs/OmniPaint-Bench-Part1/ModelA runs/OmniPaint-Bench-Part1/ModelB \
    --img_dir data/OmniPaint-Bench-Part1/img --mask_dir data/OmniPaint-Bench-Part1/mask
```

Each `<pred_dir>` gets `metrics_insertion.json` (settings + mean/std per metric + FID) and `metrics_insertion.csv`
(per-image scores). Useful options: `--pred_suffix` (default `_comp.png`), `--gen_fg mask|bbox`, `--save_fg` (writes
the cut-outs to `<pred_dir>/fg_eval/` for inspection), `--no_cute`, `--no_fid`, `--limit N`.

Build the comparison table (LaTeX, Markdown, CSV) from those results:

```bash
python build_table.py --runs_root runs --out results/insertion_table \
    --dataset "OmniPaint-Bench" OmniPaint-Bench-Part1 data/OmniPaint-Bench-Part1/img \
    --model "Model A" ModelA --model "Model B" ModelB
```

`--dataset` and `--model` repeat; a model whose result covers fewer images than the dataset is shown as "–".

## Notes

- Cut-outs smaller than 64 px are upscaled before CUTE, because carvekit rejects images under 32 px; CUTE resizes
  to 336 px anyway. The other metrics are unaffected.
- The identity metrics use the GT mask for both cut-outs, so a model that moves, resizes or reshapes the object is
  penalised.
- `cute_ffa.py` is CUTE's `foreground_feature_averaging.py` (MIT License, notice in the file), with one change: if
  GitHub cannot be reached, DINOv2 is loaded from the torch.hub cache.
