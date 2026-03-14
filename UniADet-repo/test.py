import argparse
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from uniadet.backbones import build_backbone
from uniadet.data.fewshot import FewShotMemoryBank
from uniadet.data.registry import build_dataset, default_metrics_for_dataset
from uniadet.data.transforms import build_image_transform, build_mask_transform
from uniadet.models.uniadet import UniADetHead
from uniadet.utils import set_seed


@dataclass
class TestConfig:
    data_root: str
    split: str
    dataset: str
    backbone: str
    layers: List[int]
    image_size: int
    ckpt: str
    num_workers: int
    seed: int
    lambda_p: float
    metrics: str
    num_bins: int
    fewshot: bool
    shots: int
    fewshot_split: str
    lambda_f: float
    gaussian_sigma: float
    report_format: str


def parse_args() -> TestConfig:
    parser = argparse.ArgumentParser("UniADet testing (zero-shot / few-shot)")
    parser.add_argument("--data-root", type=str, required=True, help="dataset root containing meta.json")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--dataset",
        type=str,
        default="auto",
        help="dataset name to auto-adapt: auto|meta|folder|mvtec|visa|btad|dtd|ksdd|real-iad|...",
    )
    parser.add_argument("--backbone", type=str, default="clip-vit-l14-336")
    parser.add_argument("--layers", type=int, nargs="+", default=[12, 15, 18, 21, 24])
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--lambda-p", type=float, default=0.5)
    parser.add_argument(
        "--metrics",
        type=str,
        default="auto",
        choices=["auto", "image", "pixel", "image-pixel"],
        help="auto uses a dataset default; otherwise override metric reporting",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=4096,
        help="number of histogram bins for streaming pixel AUROC/AUPR (avoids storing all pixel maps)",
    )

    parser.add_argument("--fewshot", action="store_true")
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--fewshot-split", type=str, default="train", help="split used to sample normal images")
    parser.add_argument("--lambda-f", type=float, default=0.5)
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=4.0,
        help="apply gaussian blur (sigma) to anomaly maps; set 0 to disable",
    )
    parser.add_argument(
        "--report-format",
        type=str,
        default="table",
        choices=["table", "csv", "both"],
        help="how to print final metrics",
    )

    args = parser.parse_args()
    return TestConfig(
        data_root=args.data_root,
        split=args.split,
        dataset=args.dataset,
        backbone=args.backbone,
        layers=args.layers,
        image_size=args.image_size,
        ckpt=args.ckpt,
        num_workers=args.num_workers,
        seed=args.seed,
        lambda_p=args.lambda_p,
        metrics=args.metrics,
        num_bins=args.num_bins,
        fewshot=args.fewshot,
        shots=args.shots,
        fewshot_split=args.fewshot_split,
        lambda_f=args.lambda_f,
        gaussian_sigma=args.gaussian_sigma,
        report_format=args.report_format,
    )


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))


def gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    x: [1,1,H,W] float tensor
    """
    if sigma <= 0:
        return x
    radius = int(round(3.0 * sigma))
    if radius <= 0:
        return x
    device = x.device
    k = torch.arange(-radius, radius + 1, device=device, dtype=x.dtype)
    kernel1d = torch.exp(-(k**2) / (2.0 * sigma * sigma))
    kernel1d = kernel1d / kernel1d.sum()
    # separable conv: horizontal then vertical
    kernel_x = kernel1d.view(1, 1, 1, -1)
    kernel_y = kernel1d.view(1, 1, -1, 1)
    x = F.conv2d(x, kernel_x, padding=(0, radius))
    x = F.conv2d(x, kernel_y, padding=(radius, 0))
    return x


@dataclass
class BinaryHist:
    num_bins: int
    pos: torch.Tensor  # [B] int64
    neg: torch.Tensor  # [B] int64

    @classmethod
    def create(cls, num_bins: int, *, device: Optional[torch.device] = None) -> "BinaryHist":
        device = device if device is not None else torch.device("cpu")
        return cls(
            num_bins=int(num_bins),
            pos=torch.zeros(int(num_bins), dtype=torch.int64, device=device),
            neg=torch.zeros(int(num_bins), dtype=torch.int64, device=device),
        )

    def update(self, y_true_mask: torch.Tensor, y_score_map: torch.Tensor) -> None:
        """
        y_true_mask: [H,W] or [1,1,H,W] float/bool tensor (1=anomaly)
        y_score_map: [H,W] float tensor in [0,1] (or will be clamped)
        """
        if y_true_mask.ndim == 4:
            y_true_mask = y_true_mask[0, 0]
        if y_score_map.ndim == 4:
            y_score_map = y_score_map[0, 0]
        if y_score_map.ndim == 3:
            y_score_map = y_score_map[0]

        y_true = (y_true_mask.reshape(-1) > 0.5)
        y_score = y_score_map.reshape(-1).detach()
        y_score = y_score.clamp(0.0, 1.0)

        bins = torch.floor(y_score * float(self.num_bins - 1)).to(torch.int64)
        pos_bins = bins[y_true]
        neg_bins = bins[~y_true]

        pos_counts = torch.bincount(pos_bins, minlength=self.num_bins)
        neg_counts = torch.bincount(neg_bins, minlength=self.num_bins)

        self.pos.add_(pos_counts.to(device=self.pos.device, dtype=self.pos.dtype))
        self.neg.add_(neg_counts.to(device=self.neg.device, dtype=self.neg.dtype))

    def compute_auroc_aupr(self) -> Tuple[float, float]:
        pos = self.pos.detach().cpu().to(dtype=torch.float64)
        neg = self.neg.detach().cpu().to(dtype=torch.float64)
        total_pos = float(pos.sum().item())
        total_neg = float(neg.sum().item())
        if total_pos <= 0.0 or total_neg <= 0.0:
            return float("nan"), float("nan")

        tp = torch.flip(pos, dims=[0]).cumsum(0)
        fp = torch.flip(neg, dims=[0]).cumsum(0)

        tpr = tp / total_pos
        fpr = fp / total_neg
        tpr = torch.cat([torch.zeros(1, dtype=torch.float64), tpr], dim=0)
        fpr = torch.cat([torch.zeros(1, dtype=torch.float64), fpr], dim=0)
        auroc = float(torch.trapz(tpr, fpr).item())

        precision = tp / (tp + fp + 1e-12)
        recall = tp / total_pos
        precision = torch.cat([torch.ones(1, dtype=torch.float64), precision], dim=0)
        recall = torch.cat([torch.zeros(1, dtype=torch.float64), recall], dim=0)
        aupr = float(torch.trapz(precision, recall).item())
        return auroc, aupr


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    metrics_mode = cfg.metrics
    if metrics_mode == "auto":
        metrics_mode = default_metrics_for_dataset(cfg.dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = build_backbone(cfg.backbone, device=device, image_size=cfg.image_size)
    backbone.eval()
    if cfg.image_size % int(backbone.patch_size) != 0:
        raise ValueError(f"--image-size {cfg.image_size} must be divisible by patch_size={backbone.patch_size} for {cfg.backbone}")

    ckpt = torch.load(cfg.ckpt, map_location="cpu")
    tau = float(ckpt.get("tau", 0.07))
    head = UniADetHead(layers=cfg.layers, embed_dim=backbone.embed_dim, tau=tau).to(device)
    head.load_state_dict(ckpt["head"], strict=True)
    head.eval()

    image_transform = build_image_transform(cfg.backbone, cfg.image_size)
    mask_transform = build_mask_transform(cfg.image_size)

    test_dataset = build_dataset(
        dataset=cfg.dataset,
        root=cfg.data_root,
        split=cfg.split,
        image_transform=image_transform,
        mask_transform=mask_transform,
        caa=None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    memory_bank: Optional[FewShotMemoryBank] = None
    if cfg.fewshot:
        fewshot_dataset = build_dataset(
            dataset=cfg.dataset,
            root=cfg.data_root,
            split=cfg.fewshot_split,
            image_transform=image_transform,
            mask_transform=mask_transform,
            caa=None,
        )
        memory_bank = FewShotMemoryBank.build(
            backbone=backbone,
            dataset=fewshot_dataset,
            layers=cfg.layers,
            shots=cfg.shots,
            device=device,
            seed=cfg.seed,
        )

    per_class: Dict[str, Dict[str, Any]] = {}
    for cls_name in test_dataset.class_names:
        entry: Dict[str, Any] = {"y_true_img": [], "y_score_img": []}
        if metrics_mode in {"pixel", "image-pixel"}:
            entry["px_hist"] = BinaryHist.create(cfg.num_bins, device=device)
        per_class[cls_name] = entry

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="testing", dynamic_ncols=True):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            labels = batch["label"].to(device)
            cls_name = batch["cls_name"][0]

            layer_tokens = backbone.extract_tokens(images, cfg.layers)

            layer_scores_z: List[float] = []
            layer_map_max: List[float] = []
            patch_sum: Optional[torch.Tensor] = None
            for layer in cfg.layers:
                layer_key = str(layer)
                cls_tok, patch_tok = layer_tokens[layer]  # cls: [1,D], patch: [1,N,D]
                cls_tok = F.normalize(cls_tok, dim=-1)
                patch_tok = F.normalize(patch_tok, dim=-1)

                # Eq.(8) in the paper: per-layer predictions.
                # y_z^l: anomaly probability from global token; Y_z^l: anomaly map from patch tokens.
                w_cls = F.normalize(head.w_cls[layer_key], dim=-1)  # [2,D]
                logits_cls = (cls_tok @ w_cls.t()) / float(head.tau)  # [1,2]
                y_z_l = float(F.softmax(logits_cls, dim=-1)[0, 1].item())

                w_seg = F.normalize(head.w_seg[layer_key], dim=-1)  # [2,D]
                logits_seg = (patch_tok @ w_seg.t()) / float(head.tau)  # [1,N,2]
                probs_seg = logits_seg.softmax(dim=-1)[0, :, 1]  # [N]

                layer_scores_z.append(y_z_l)
                layer_map_max.append(float(probs_seg.max().item()))
                patch_sum = probs_seg if patch_sum is None else (patch_sum + probs_seg)

            # Zero-shot inference (paper Sec.3.5):
            # fuse per-layer y_z^l and max(Y_z^l), then average over layers.
            y_img_z = float(np.mean(layer_scores_z))
            if patch_sum is None:
                raise RuntimeError("Empty layers for inference.")
            patch_avg = patch_sum / float(len(cfg.layers))  # [N]
            y_img_p = float(np.mean(layer_map_max))

            # Build pixel map from aggregated patch grid.
            num_patches = int(patch_avg.numel())
            side = int(math.isqrt(num_patches))
            if side * side != num_patches:
                raise ValueError(f"Cannot reshape patches: N={num_patches} is not a square")
            final_map_t = patch_avg.view(1, 1, side, side)
            final_map_t = torch.nn.functional.interpolate(
                final_map_t, size=(cfg.image_size, cfg.image_size), mode="bilinear", align_corners=False
            )[0, 0]
            if cfg.gaussian_sigma > 0:
                final_map_t = gaussian_blur_2d(
                    final_map_t.view(1, 1, cfg.image_size, cfg.image_size), cfg.gaussian_sigma
                )[0, 0]
            final_score = (1.0 - cfg.lambda_p) * y_img_z + cfg.lambda_p * y_img_p

            if memory_bank is not None:
                fewshot_map = memory_bank.score_map_for_class(
                    cls_name, layer_tokens, output_size=(cfg.image_size, cfg.image_size)
                )
                fewshot_map_t = torch.from_numpy(fewshot_map).to(device=device, dtype=final_map_t.dtype)
                fused_map_t = (1.0 - cfg.lambda_f) * final_map_t + cfg.lambda_f * fewshot_map_t
                if cfg.gaussian_sigma > 0:
                    fused_map_t = gaussian_blur_2d(
                        fused_map_t.view(1, 1, cfg.image_size, cfg.image_size), cfg.gaussian_sigma
                    )[0, 0]
                final_map_t = fused_map_t
                final_score = (1.0 - cfg.lambda_p) * y_img_z + cfg.lambda_p * float(final_map_t.max().item())

            per_class[cls_name]["y_true_img"].append(int(labels.item()))
            per_class[cls_name]["y_score_img"].append(final_score)
            if metrics_mode in {"pixel", "image-pixel"}:
                per_class[cls_name]["px_hist"].update(masks, final_map_t)

    rows: List[Tuple[str, float, float, float, float]] = []
    img_auroc_list: List[float] = []
    img_aupr_list: List[float] = []
    px_auroc_list: List[float] = []
    px_aupr_list: List[float] = []

    for cls_name, values in per_class.items():
        y_true_img = np.asarray(values["y_true_img"], dtype=np.int64)
        y_score_img = np.asarray(values["y_score_img"], dtype=np.float32)

        img_auroc = _safe_roc_auc(y_true_img, y_score_img) if metrics_mode in {"image", "image-pixel"} else float("nan")
        img_aupr = _safe_average_precision(y_true_img, y_score_img) if metrics_mode in {"image", "image-pixel"} else float("nan")
        if metrics_mode in {"pixel", "image-pixel"}:
            px_auroc, px_aupr = values["px_hist"].compute_auroc_aupr()
        else:
            px_auroc, px_aupr = float("nan"), float("nan")

        rows.append((cls_name, img_auroc, img_aupr, px_auroc, px_aupr))
        img_auroc_list.append(img_auroc)
        img_aupr_list.append(img_aupr)
        px_auroc_list.append(px_auroc)
        px_aupr_list.append(px_aupr)

    def _nanmean(xs: List[float]) -> float:
        arr = np.asarray(xs, dtype=np.float32)
        return float(np.nanmean(arr))

    rows.append(("mean", _nanmean(img_auroc_list), _nanmean(img_aupr_list), _nanmean(px_auroc_list), _nanmean(px_aupr_list)))

    headers = ["object", "image_auroc", "image_aupr", "pixel_auroc", "pixel_aupr"]

    def _fmt_pct(x: float) -> str:
        if isinstance(x, float) and math.isnan(x):
            return "nan"
        return f"{x*100:.2f}"

    def _print_markdown_table() -> None:
        table_rows = [[r[0], _fmt_pct(r[1]), _fmt_pct(r[2]), _fmt_pct(r[3]), _fmt_pct(r[4])] for r in rows]
        widths = [len(h) for h in headers]
        for tr in table_rows:
            for i, v in enumerate(tr):
                widths[i] = max(widths[i], len(str(v)))

        def _row(vals: List[str]) -> str:
            return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(vals)) + " |"

        print(_row(headers))
        print("|-" + "-|-".join("-" * w for w in widths) + "-|")
        for tr in table_rows:
            print(_row([str(x) for x in tr]))

    def _print_csv() -> None:
        print(",".join(headers))
        for r in rows:
            print(f"{r[0]},{_fmt_pct(r[1])},{_fmt_pct(r[2])},{_fmt_pct(r[3])},{_fmt_pct(r[4])}")

    if cfg.report_format == "table":
        _print_markdown_table()
    elif cfg.report_format == "csv":
        _print_csv()
    else:
        _print_markdown_table()
        print()
        _print_csv()


if __name__ == "__main__":
    main()
