import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class Item:
    img_path: str
    mask_path: str
    cls_name: str
    specie_name: str
    anomaly: int

    def to_dict(self) -> Dict:
        return {
            "img_path": self.img_path,
            "mask_path": self.mask_path,
            "cls_name": self.cls_name,
            "specie_name": self.specie_name,
            "anomaly": self.anomaly,
        }


def _list_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if not folder.exists():
        return []
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def build_meta(root: Path, good_name: str = "good", gt_dir_name: str = "ground_truth") -> Dict:
    meta: Dict[str, Dict[str, List[Dict]]] = {"train": {}, "test": {}}

    class_dirs = [p for p in root.iterdir() if p.is_dir()]
    for class_dir in sorted(class_dirs):
        cls_name = class_dir.name

        train_good_dir = class_dir / "train" / good_name
        test_dir = class_dir / "test"
        gt_dir = class_dir / gt_dir_name

        train_items: List[Item] = []
        for img_path in _list_images(train_good_dir):
            rel_img = img_path.relative_to(root).as_posix()
            train_items.append(Item(rel_img, "", cls_name, good_name, 0))

        test_items: List[Item] = []
        if test_dir.exists():
            for specie_dir in sorted([p for p in test_dir.iterdir() if p.is_dir()]):
                specie_name = specie_dir.name
                for img_path in _list_images(specie_dir):
                    rel_img = img_path.relative_to(root).as_posix()
                    if specie_name == good_name:
                        test_items.append(Item(rel_img, "", cls_name, specie_name, 0))
                    else:
                        candidate1 = gt_dir / specie_name / img_path.name
                        candidate2 = gt_dir / specie_name / (img_path.stem + "_mask" + img_path.suffix)
                        if candidate1.exists():
                            mask_path = candidate1.relative_to(root).as_posix()
                        elif candidate2.exists():
                            mask_path = candidate2.relative_to(root).as_posix()
                        else:
                            mask_path = ""
                        test_items.append(Item(rel_img, mask_path, cls_name, specie_name, 1))

        meta["train"][cls_name] = [it.to_dict() for it in train_items]
        meta["test"][cls_name] = [it.to_dict() for it in test_items]

    return meta


def main() -> None:
    parser = argparse.ArgumentParser("Make meta.json for MVTec-style datasets")
    parser.add_argument("--root", type=str, required=True, help="dataset root containing class folders")
    parser.add_argument("--good-name", type=str, default="good")
    parser.add_argument("--gt-dir-name", type=str, default="ground_truth")
    parser.add_argument("--out", type=str, default="meta.json")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    meta = build_meta(root, good_name=args.good_name, gt_dir_name=args.gt_dir_name)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()

