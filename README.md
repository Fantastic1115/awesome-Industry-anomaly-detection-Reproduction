# awesome-Industry-anomaly-detection-Reproduction

# UniADet (Paper Reproduction Version)
This repository reproduces **UniADet** from the paper **“One Language-Free Foundation Model Is Enough for Universal Vision Anomaly Detection”**: it learns only a small number of **decoupled weights** while keeping the vision foundation model frozen, simultaneously performing **image-level anomaly classification** and **pixel-level anomaly segmentation**, and supports few-shot inference with only a few normal samples.

## Core Ideas (Corresponding to Paper Sec.3)

- **Language-Free**: In anomaly detection, the language encoder of a VLM essentially generates binary classification (normal/anomaly) **decision weights** `W`; UniADet directly learns `W` without requiring prompts / a text encoder.
- **Task Decoupling (DCS)**: Classification uses global tokens, segmentation uses local patch tokens, and their feature distributions differ; therefore the shared weights are split into `W_cls` and `W_seg`.
- **Hierarchical Decoupling (DHF)**: Features from different transformer blocks also lie on different manifolds; independent `W_cls^l` and `W_seg^l` are learned for each selected block.
- **Few‑shot Memory Bank**: For `K ∈ {1,2,4}` normal images of the target class, their multi‑layer patch features are stored in a memory bank. For each patch of a query image, a nearest‑neighbor cosine distance yields a few‑shot anomaly map, which is then fused with the zero‑shot map.

## Installation

It is recommended to use Python 3.10+ and install PyTorch (choose the version matching your CUDA).

```bash
pip install -r requirements.txt
```

## Data Format (meta.json)

To avoid hard‑coding the directory structure of specific datasets, this implementation uses `meta.json` to describe the data (common practice in AnomalyCLIP / some anomaly detection code).

`meta.json` structure:

```json
{
  "train": {
    "class_a": [
      {"img_path": "class_a/train/good/000.png", "mask_path": "", "cls_name": "class_a", "specie_name": "good", "anomaly": 0}
    ]
  },
  "test": {
    "class_a": [
      {"img_path": "class_a/test/bad/001.png", "mask_path": "class_a/ground_truth/bad/001.png", "cls_name": "class_a", "specie_name": "bad", "anomaly": 1}
    ]
  }
}
```

You can use `tools/make_meta_mvtec_style.py` to generate `meta.json` for MVTec‑style datasets (e.g., MVTec, BTAD, etc.).

## Training (Only Learn Decoupled Weights)

Taking the CLIP vision encoder as an example (default paper settings: blocks = 12,15,18,21,24; input size 518; lr=1e-3; epochs=15):

```bash
python train.py ^
  --data-root /path/to/dataset ^
  --split test ^
  --backbone clip-vit-l14-336 ^
  --epochs 15 ^
  --batch-size 8 ^
  --lr 0.001 ^
  --out checkpoints/uniadet_clip.pth
```

If you wish to reproduce the paper’s protocol **“train on VisA test, generalize to other datasets”**, point `--split` to the corresponding split you have written into `meta.json` (e.g., `test`).

## Testing (zero-shot / few-shot)

Zero-shot:

```bash
python test.py ^
  --data-root /path/to/dataset ^
  --split test ^
  --backbone clip-vit-l14-336 ^
  --ckpt checkpoints/uniadet_clip.pth
```

Few-shot (sample K normal images per class from `--fewshot-split` of the target dataset to build the memory bank):

```bash
python test.py ^
  --data-root /path/to/dataset ^
  --split test ^
  --fewshot --shots 4 --fewshot-split train ^
  --backbone clip-vit-l14-336 ^
  --ckpt checkpoints/uniadet_clip.pth
```

## Supported Backbones

- `clip-vit-l14-336`: HuggingFace `openai/clip-vit-large-patch14-336`
- `dinov2-vit-l14-reg`: `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')`
- `dinov3-vit-l16`: `torch.hub.load('facebookresearch/dinov3', 'dinov3_vitl16')`

> Note: For DINOv2/DINOv3, CLS and patch tokens from the specified blocks are obtained directly via `get_intermediate_layers()`; for CLIP we use the `hidden_states` output from HuggingFace.

## Result
Zero-shot(train on mvtec test on visa)
| object     | image_auroc | image_aupr | pixel_auroc | pixel_aupr |
|------------|-------------|------------|-------------|------------|
| candle     | 91.38       | 93.88      | 99.44       | 53.35      |
| capsules   | 95.67       | 97.86      | 99.47       | 70.96      |
| cashew     | 79.94       | 90.27      | 90.93       | 17.28      |
| chewinggum | 99.34       | 99.69      | 99.69       | 86.37      |
| fryum      | 92.70       | 96.65      | 94.13       | 28.27      |
| macaroni1  | 94.24       | 94.42      | 99.62       | 34.67      |
| macaroni2  | 78.95       | 81.59      | 99.10       | 9.85       |
| pcb1       | 95.38       | 95.76      | 91.47       | 9.75       |
| pcb2       | 86.30       | 88.19      | 94.53       | 27.78      |
| pcb3       | 70.73       | 71.24      | 91.40       | 4.87       |
| pcb4       | 96.12       | 96.07      | 94.95       | 24.39      |
| pipe_fryum | 99.00       | 99.53      | 94.49       | 22.37      |
| mean       | 89.98       | 92.10      | 95.77       | 32.49      |
