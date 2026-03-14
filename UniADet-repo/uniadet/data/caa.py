from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image


@dataclass
class ClassAwareAugmentor:
    """
    Class-Aware Augmentation (CAA) from the UniADet supplementary material:
      - Grid Mosaic
      - Grid Cropping

    This augmentor must be bound to a MetaJsonDataset via bind().
    """

    prob: float = 0.5
    grid_sizes: Sequence[int] = (2, 3)
    mosaic_prob: float = 0.5

    _dataset: Optional[object] = None

    def bind(self, dataset: object) -> None:
        self._dataset = dataset

    def __call__(self, img: Image.Image, mask: Image.Image, item) -> Tuple[Image.Image, Image.Image]:
        if self._dataset is None:
            return img, mask
        if random.random() > self.prob:
            return img, mask

        grid_n = int(random.choice(list(self.grid_sizes)))
        if random.random() < self.mosaic_prob:
            return self._grid_mosaic(img, mask, item, grid_n)
        return self._grid_crop(img, mask, item, grid_n)

    def _grid_mosaic(self, img: Image.Image, mask: Image.Image, item, n: int) -> Tuple[Image.Image, Image.Image]:
        dataset = self._dataset
        assert dataset is not None

        w, h = img.size
        cell_w = max(1, w // n)
        cell_h = max(1, h // n)

        canvas = Image.new("RGB", (cell_w * n, cell_h * n))
        canvas_mask = Image.new("L", (cell_w * n, cell_h * n))

        total = n * n
        indices = [None] * total
        for k in range(1, total):
            if int(item.anomaly) == 0:
                idx = dataset.sample_index(item.cls_name, anomaly=0)
            else:
                idx = dataset.sample_index(item.cls_name, anomaly=None)
            indices[k] = idx

        for k in range(total):
            r = k // n
            c = k % n
            x0, y0 = c * cell_w, r * cell_h

            if k == 0:
                src_img, src_mask = img, mask
            else:
                src_img, src_mask, _ = dataset.load_raw(indices[k])

            src_img = src_img.resize((cell_w, cell_h), resample=Image.BICUBIC)
            src_mask = src_mask.resize((cell_w, cell_h), resample=Image.NEAREST)

            canvas.paste(src_img, (x0, y0))
            canvas_mask.paste(src_mask, (x0, y0))

        return canvas, canvas_mask

    def _grid_crop(self, img: Image.Image, mask: Image.Image, item, n: int) -> Tuple[Image.Image, Image.Image]:
        w, h = img.size
        cell_w = max(1, w // n)
        cell_h = max(1, h // n)

        candidates = []
        if int(item.anomaly) == 1:
            mask_arr = np.array(mask, dtype=np.uint8) > 0
            for r in range(n):
                for c in range(n):
                    x0, y0 = c * cell_w, r * cell_h
                    x1, y1 = x0 + cell_w, y0 + cell_h
                    if mask_arr[y0:y1, x0:x1].any():
                        candidates.append((x0, y0, x1, y1))

        if not candidates:
            r = random.randrange(n)
            c = random.randrange(n)
            x0, y0 = c * cell_w, r * cell_h
            candidates = [(x0, y0, x0 + cell_w, y0 + cell_h)]

        crop_box = random.choice(candidates)
        img_c = img.crop(crop_box)
        mask_c = mask.crop(crop_box)

        img_c = img_c.resize((w, h), resample=Image.BICUBIC)
        mask_c = mask_c.resize((w, h), resample=Image.NEAREST)
        return img_c, mask_c

