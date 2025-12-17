import math
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from datasets.cfg import class_colors 

def _id_to_name(cid: int) -> str:
    if cid == 23:
        return "X"
    if cid == 24:
        return "Y"
    return str(int(cid))

def _mask_center(mask: np.ndarray):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0, 0
    return int(xs.mean()), int(ys.mean())

def render_overlay_pil(pil_gray: Image.Image, boxes, labels, masks):
    base = pil_gray.convert("RGB")
    W, H = base.size
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for b, lab, m in zip(boxes, labels, masks):
        color = class_colors.get(int(lab), (0.0, 1.0, 0.0))
        rgba = (int(color[0]*255), int(color[1]*255), int(color[2]*255), 90)

        ys, xs = np.where(m)
        for x, y in zip(xs.tolist(), ys.tolist()):
            overlay.putpixel((x, y), rgba)

        cx, cy = _mask_center(m)
        txt = _id_to_name(int(lab))
        od.rectangle([cx-10, cy-10, cx+10, cy+10], outline=(0,0,0,180))
        od.text((cx+6, cy-10), txt, fill=(255, 255, 255, 255), font=font)

    out = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    return out

def _crop_with_pad(pil_gray: Image.Image, mask: np.ndarray, box, pad: int = 5):
    W, H = pil_gray.size
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(x1 - pad, 0); y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, W); y2 = min(y2 + pad, H)
    if x2 <= x1 or y2 <= y1:
        return None, None
    crop_img = pil_gray.crop((x1, y1, x2, y2)).convert("RGB")
    crop_m = mask[y1:y2, x1:x2]
    return crop_img, crop_m

def render_karyogram_pil(pil_gray: Image.Image, boxes, labels, masks, pad: int = 5):
    by_lab = defaultdict(list)
    for b, lab, m in zip(boxes, labels, masks):
        ci, cm = _crop_with_pad(pil_gray, m, b, pad=pad)
        if ci is None:
            continue
        by_lab[int(lab)].append((ci, cm))

    for lab in by_lab:
        by_lab[lab] = by_lab[lab][:2]

    labs = sorted(by_lab.keys())
    if not labs:
        return Image.new("RGB", (600, 200), (30, 30, 30))

    cols = 2
    cell_w, cell_h = 220, 140
    rows = len(labs)
    out = Image.new("RGB", (cols * cell_w, rows * cell_h), (20, 20, 20))
    draw = ImageDraw.Draw(out)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for r, lab in enumerate(labs):
        items = by_lab[lab]
        for c in range(cols):
            x0, y0 = c * cell_w, r * cell_h
            draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], outline=(60, 60, 60))

            if c >= len(items):
                continue

            img, m = items[c]
            img = img.copy()
            img.thumbnail((cell_w - 10, cell_h - 30), Image.BILINEAR)

            # mask overlay (class color)
            color = class_colors.get(int(lab), (0.0, 1.0, 0.0))
            rgba = (int(color[0]*255), int(color[1]*255), int(color[2]*255), 100)
            ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ovd = ImageDraw.Draw(ov)

            # resize mask to match thumbnail
            m_img = Image.fromarray((m.astype(np.uint8) * 255), mode="L").resize(img.size, Image.NEAREST)
            m_np = np.array(m_img) > 0
            ys, xs = np.where(m_np)
            for x, y in zip(xs.tolist(), ys.tolist()):
                ov.putpixel((x, y), rgba)

            img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
            out.paste(img, (x0 + 5, y0 + 20))

            draw.text((x0 + 5, y0 + 2), f"chr { _id_to_name(lab) }", fill=(240, 240, 240), font=font)

    return out
