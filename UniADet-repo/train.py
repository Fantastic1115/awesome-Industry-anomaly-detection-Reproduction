import argparse
import os
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from uniadet.backbones import build_backbone
from uniadet.data.caa import ClassAwareAugmentor
from uniadet.data.transforms import build_image_transform, build_mask_transform
from uniadet.losses import BinaryDiceLoss, FocalLoss
from uniadet.models.uniadet import UniADetHead
from uniadet.utils import set_seed
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_descriptor')
@dataclass
class TrainConfig:
    data_root: str
    split: str
    dataset: str
    backbone: str
    layers: List[int]
    image_size: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seg_loss_weight: float
    tau: float
    seed: int
    num_workers: int
    out: str
    caa_prob: float


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser("UniADet training (learn decoupled weights only)")
    parser.add_argument("--data-root", type=str, required=True, help="dataset root containing meta.json")
    parser.add_argument("--split", type=str, default="train", help="split name in meta.json (e.g., train/test)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="auto",
        help="dataset name to auto-adapt: auto|meta|folder|mvtec|visa|btad|dtd|ksdd|real-iad|...",
    )
    parser.add_argument("--backbone", type=str, default="clip-vit-l16-336")
    parser.add_argument("--layers", type=int, nargs="+", default=[12, 15, 18, 21, 24])
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seg-loss-weight", type=float, default=4.0)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="checkpoint path. If a directory (or no .pth suffix), saves into it as epoch_XX.pth",
    )
    parser.add_argument("--caa-prob", type=float, default=0.5)
    args = parser.parse_args()
    return TrainConfig(
        data_root=args.data_root,
        split=args.split,
        dataset=args.dataset,
        backbone=args.backbone,
        layers=args.layers,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seg_loss_weight=args.seg_loss_weight,
        tau=args.tau,
        seed=args.seed,
        num_workers=args.num_workers,
        out=args.out,
        caa_prob=args.caa_prob,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = build_backbone(cfg.backbone, device=device, image_size=cfg.image_size)
    backbone.eval()
    if cfg.image_size % int(backbone.patch_size) != 0:
        raise ValueError(f"--image-size {cfg.image_size} must be divisible by patch_size={backbone.patch_size} for {cfg.backbone}")

    image_transform = build_image_transform(cfg.backbone, cfg.image_size)
    mask_transform = build_mask_transform(cfg.image_size)

    caa = ClassAwareAugmentor(prob=cfg.caa_prob, grid_sizes=(2, 3))
    from uniadet.data import build_dataset

    dataset = build_dataset(
        dataset=cfg.dataset,
        root=cfg.data_root,
        split=cfg.split,
        image_transform=image_transform,
        mask_transform=mask_transform,
        caa=caa,
    )
    num_anom = sum(int(it.anomaly) for it in dataset.items)
    num_total = len(dataset.items)
    num_norm = num_total - num_anom
    print(f"train split '{cfg.split}': total={num_total} normal={num_norm} anomaly={num_anom}")
    if num_anom == 0 or num_norm == 0:
        raise ValueError(
            "Training split must contain both normal and anomaly images (with masks for anomalies). "
            "For the paper's cross-dataset protocol, use MVTec test split when evaluating on VisA."
        )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    head = UniADetHead(layers=cfg.layers, embed_dim=backbone.embed_dim, tau=cfg.tau).to(device)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.5, 0.999),
    )

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    def resolve_ckpt_path(out: str, epoch: int) -> str:
        out = os.fspath(out)
        # If user passes a directory (or an "out" without .pth), write epoch_XX.pth inside it.
        # This avoids torch.save failing on directory paths like "./save".
        if out.endswith(os.sep) or os.path.isdir(out):
            os.makedirs(out, exist_ok=True)
            return os.path.join(out, f"epoch_{epoch:02d}.pth")
        if out.lower().endswith(".pth"):
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            return out
        if os.path.exists(out) and os.path.isfile(out):
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            return out
        # Path doesn't exist and has no .pth suffix: treat as output directory.
        os.makedirs(out, exist_ok=True)
        return os.path.join(out, f"epoch_{epoch:02d}.pth")

    for epoch in range(1, cfg.epochs + 1):
        head.train()
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_seg_loss = 0.0

        pbar = tqdm(loader, desc=f"epoch {epoch}/{cfg.epochs}", dynamic_ncols=True)
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.no_grad():
                layer_tokens = backbone.extract_tokens(images, cfg.layers)

            outputs = head(layer_tokens, output_size=(cfg.image_size, cfg.image_size))

            cls_loss = 0.0
            seg_loss = 0.0
            for layer in cfg.layers:
                layer_key = str(layer)
                cls_logits = outputs["cls_logits"][layer_key]
                seg_probs = outputs["seg_probs"][layer_key]  # [B,2,H,W]

                cls_loss = cls_loss + F.cross_entropy(cls_logits, labels.long())

                target = masks.long()
                seg_loss = seg_loss + loss_focal(seg_probs, target)
                seg_loss = seg_loss + loss_dice(seg_probs[:, 1, :, :], masks)
                seg_loss = seg_loss + loss_dice(seg_probs[:, 0, :, :], 1.0 - masks)

            cls_loss = cls_loss / float(len(cfg.layers))
            seg_loss = seg_loss / float(len(cfg.layers))
            loss = cls_loss + cfg.seg_loss_weight * seg_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_cls_loss += float(cls_loss.item())
            epoch_seg_loss += float(seg_loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}", cls=f"{cls_loss.item():.4f}", seg=f"{seg_loss.item():.4f}")

        num_steps = max(1, len(loader))
        print(
            f"[epoch {epoch}] loss={epoch_loss/num_steps:.4f} "
            f"cls={epoch_cls_loss/num_steps:.4f} seg={epoch_seg_loss/num_steps:.4f}"
        )

        ckpt_path = resolve_ckpt_path(cfg.out, epoch)
        torch.save(
            {
                "backbone": cfg.backbone,
                "layers": cfg.layers,
                "image_size": cfg.image_size,
                "tau": cfg.tau,
                "head": head.state_dict(),
            },
            ckpt_path,
        )
        if epoch == cfg.epochs:
            print(f"saved: {ckpt_path}")


if __name__ == "__main__":
    main()
