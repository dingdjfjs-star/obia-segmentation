# -*- coding: utf-8 -*-
"""出图：多尺度分割对比图 / 像素级 vs 对象级对比图 / 参数敏感性曲线。"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from .metrics import boundary_f

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CLASS_COLORS = {                    # 与 synthetic.CLASS_CODES 对应
    1: (0.20, 0.45, 0.85),          # water  蓝
    2: (0.10, 0.50, 0.20),          # forest 深绿
    3: (0.75, 0.80, 0.25),          # farmland 黄绿
    4: (0.78, 0.65, 0.45),          # soil   土棕
    5: (0.75, 0.35, 0.30),          # builtup 砖红
}
CLASS_NAMES_CN = {1: "水体", 2: "林地", 3: "耕地", 4: "裸土", 5: "建设用地"}


def rgb_composite(image):
    """自然色合成（波段序 [B,G,R,NIR] → 取 R,G,B）+ 2% 线性拉伸。"""
    rgb = image[:, :, [2, 1, 0]].astype(np.float32)
    lo, hi = np.percentile(rgb, [2, 98])
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)


def class_map_png(cls_map):
    """类别码图 → 彩色 RGB。"""
    out = np.zeros(cls_map.shape + (3,), np.float32)
    for code, col in CLASS_COLORS.items():
        out[cls_map == code] = col
    return out


def overlay_boundaries(rgb, labels, color=(1.0, 0.85, 0.0), dilate=1):
    """把图斑边界叠加到 RGB 上（亮黄线）。"""
    b = np.zeros(labels.shape, bool)
    b[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    b[:-1, :] |= labels[:-1, :] != labels[1:, :]
    if dilate:
        b = ndimage.binary_dilation(b, iterations=dilate)
    out = rgb.copy()
    out[b] = color
    return out


def _panel(ax, img, title, xlabel_note=""):
    ax.imshow(img, interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    if xlabel_note:
        ax.set_xlabel(xlabel_note, fontsize=9)


def fig_multiscale(image, truth_parcels, scale_runs, path):
    """多尺度分割对比：RGB / 真值图斑 / 各 scale 的分割边界叠加。

    scale_runs: [(scale, labels, info), ...]
    """
    n = len(scale_runs) + 2
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 4.0 * nrow))
    axes = np.atleast_2d(axes)
    rgb = rgb_composite(image)
    flat = axes.ravel()
    _panel(flat[0], rgb, "影像 RGB 合成")
    truth_rgb = class_map_png(np.zeros_like(truth_parcels))
    rng = np.random.default_rng(7)
    pmap = np.zeros(truth_parcels.shape + (3,), np.float32)
    cmap = plt.get_cmap("tab20")
    for i, uid in enumerate(np.unique(truth_parcels)):
        pmap[truth_parcels == uid] = cmap((i % 20) / 20)[:3]
    _panel(flat[1], pmap, f"真值图斑（{len(np.unique(truth_parcels))} 块）")
    for i, (s, lab, info) in enumerate(scale_runs):
        _panel(flat[2 + i], overlay_boundaries(rgb, lab),
               f"scale={s:g}  ·  {info['n_objects']} 图斑",
               f"shape={info['shape']}  compact={info['compactness']}  "
               f"{info['seconds']:.0f}s")
    for j in range(n, len(flat)):
        flat[j].axis("off")
    fig.suptitle("手写多尺度区域生长分割：scale 参数从小到大（黄色线为图斑边界）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_pixel_vs_object(image, truth_cls, pix_res, obj_res, obj_map, path):
    """像素级 vs 面向对象分类对比 + 误差图。"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    rgb = rgb_composite(image)
    _panel(axes[0, 0], rgb, "影像 RGB 合成")
    _panel(axes[0, 1], class_map_png(truth_cls),
           "真值地类")
    _panel(axes[0, 2], class_map_png(pix_res["pred"]),
           f"像素级 RF   OA={pix_res['oa']:.3f}   椒盐率={pix_res['speckle']:.3f}")
    _panel(axes[1, 0], class_map_png(obj_map),
           f"面向对象 RF   OA={obj_res['oa']:.3f}   椒盐率={obj_res['speckle']:.3f}")
    err_p = pix_res["pred"] != truth_cls
    err_o = obj_map != truth_cls
    e1 = rgb.copy(); e1[err_p] = (1, 0, 0)
    e2 = rgb.copy(); e2[err_o] = (1, 0, 0)
    _panel(axes[1, 1], e1, f"像素级错分像素（红） {err_p.mean() * 100:.1f}%")
    _panel(axes[1, 2], e2, f"面向对象错分像素（红） {err_o.mean() * 100:.1f}%")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=CLASS_COLORS[c]) for c in sorted(CLASS_COLORS)]
    axes[0, 1].legend(handles, [CLASS_NAMES_CN[c] for c in sorted(CLASS_COLORS)],
                      loc="lower left", fontsize=8, framealpha=0.9)
    fig.suptitle("像素级分类 vs 面向对象分类（同一 RandomForest，同 seed）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_sensitivity(scan_rows, main_scale, path):
    """参数敏感性曲线：scale 扫描（对象数/平均面积/边界F/OA）+ shape 扫描。"""
    import pandas as pd
    df = pd.DataFrame(scan_rows)
    sc = df[df.param == "scale"].sort_values("value")
    sh = df[df.param == "shape"].sort_values("value")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(sc.value, sc.n_objects, "o-", color="#c0392b")
    ax.set_xlabel("scale 参数"); ax.set_ylabel("图斑数量", color="#c0392b")
    ax2 = ax.twinx()
    ax2.plot(sc.value, sc.mean_area, "s--", color="#2471a3")
    ax2.set_ylabel("平均图斑面积（px）", color="#2471a3")
    ax.set_xscale("log")
    ax.set_title(f"scale 扫描（主尺度 {main_scale:g} 竖线）")
    ax.axvline(main_scale, color="gray", ls=":")

    ax = axes[0, 1]
    ax.plot(sc.value, sc.boundary_prec, "o-", label="边界精确率")
    ax.plot(sc.value, sc.boundary_rec, "s-", label="边界召回率")
    ax.plot(sc.value, sc.boundary_f, "^-", label="边界 F1")
    ax.set_xscale("log"); ax.set_xlabel("scale 参数"); ax.set_ylabel("对真值图斑边界的贴合度")
    ax.legend(fontsize=9); ax.set_title("scale 扫描：边界贴合度（容差 2px）")

    ax = axes[1, 0]
    ax.plot(sh.value, sh.n_objects, "o-", color="#7d3c98")
    ax.set_xlabel("shape 权重"); ax.set_ylabel("图斑数量", color="#7d3c98")
    ax2 = ax.twinx()
    ax2.plot(sh.value, sh.boundary_f, "^--", color="#1e8449")
    ax2.set_ylabel("边界 F1", color="#1e8449")
    ax.set_title("shape 权重扫描（其余参数固定）")

    ax = axes[1, 1]
    ax.plot(sc.value, sc.obj_oa, "o-", color="#c0392b",
            label="面向对象分类 OA（测试图斑口径）")
    ax.set_xscale("log"); ax.set_xlabel("scale 参数")
    ax.set_ylabel("总体精度 OA")
    ax.set_ylim(0.5, 1.02); ax.legend(fontsize=9, loc="lower left")
    ax.set_title("scale 扫描：图斑级分类精度（主尺度全图口径 OA 见 README 表）")

    fig.suptitle("参数敏感性分析（512×512 合成场景，seed=20260829）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)
