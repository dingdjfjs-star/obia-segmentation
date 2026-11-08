# -*- coding: utf-8 -*-
"""图斑（对象）特征提取：光谱均值/方差 + GLCM 纹理 + 形状/比值指数。

特征体系（面向对象分类的核心：把"像素的谱"升级为"图斑的谱+纹+形"）
----------------------------------------------------------------------
* 光谱：各波段均值 mean_b*、标准差 std_b*（对象内方差）、全谱亮度 brightness
* 比值：ndvi = (NIR-Red)/(NIR+Red)（对象级）、水指数 ndwi = (G-NIR)/(G+NIR)
* 纹理：对象内掩膜 GLCM（16 级灰度，{右,下} 两方向对称化）
        contrast / homogeneity / entropy / asm
* 形状：shape_index = 周长/√面积（eCognition 同名指数）、
        bbox_fill = 面积/外包矩形面积（矩形度）、elongation = 长宽比

GLCM 采用手写"掩膜共现统计"：只统计对象内部相邻像素对（右向 + 下向），
避免 bbox 内混入邻斑像素——这也是面向对象纹理与方块窗口纹理的本质区别。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

__all__ = ["extract_object_features", "object_majority_truth"]


def _object_perimeter(labels):
    """每个图斑的周长（像素边数，含影像边界边）。向量化 4 邻域比较。"""
    lab = labels
    per = np.zeros(int(lab.max()) + 1, np.float64)
    per[lab[:, :-1].ravel()] += (lab[:, :-1] != lab[:, 1:]).ravel()
    per[lab[:, 1:].ravel()] += (lab[:, :-1] != lab[:, 1:]).ravel()
    per[lab[:-1, :].ravel()] += (lab[:-1, :] != lab[1:, :]).ravel()
    per[lab[1:, :].ravel()] += (lab[:-1, :] != lab[1:, :]).ravel()
    # 影像边界边
    per[lab[0, :]] += 1.0
    per[lab[-1, :]] += 1.0
    per[lab[:, 0]] += 1.0
    per[lab[:, -1]] += 1.0
    return per


def _masked_glcm(gray_q, labels, n_levels=16):
    """对象内掩膜 GLCM：右向+下向共现，对称化后算 4 个常用纹理量。

    Returns
    -------
    contrast, homogeneity, entropy, asm : 长度 K 的数组（按图斑编号 1..K）
    """
    k = int(labels.max())
    right = np.stack([gray_q[:, :-1].ravel(), gray_q[:, 1:].ravel(),
                      labels[:, :-1].ravel(), labels[:, 1:].ravel()], 1)
    down = np.stack([gray_q[:-1, :].ravel(), gray_q[1:, :].ravel(),
                     labels[:-1, :].ravel(), labels[1:, :].ravel()], 1)
    pairs = np.concatenate([right, down], 0)
    same = pairs[:, 2] == pairs[:, 3]
    pairs = pairs[same]
    lab_ids = pairs[:, 2].astype(np.int64)
    gi = pairs[:, 0].astype(np.int64)
    gj = pairs[:, 1].astype(np.int64)
    code = (lab_ids * n_levels + gi) * n_levels + gj
    flat = np.bincount(code, minlength=(k + 1) * n_levels * n_levels)
    P = flat.reshape(k + 1, n_levels, n_levels)[1:]   # 丢弃 0 号占位（label 从 1 起）
    P = P + np.transpose(P, (0, 2, 1))          # 对称化
    tot = P.sum((1, 2))
    tot[tot == 0] = 1.0
    Pn = P / tot[:, None, None]
    ii = np.arange(n_levels)
    diff = ii[:, None] - ii[None, :]
    w_con = (diff ** 2)[None]
    contrast = (Pn * w_con).sum((1, 2))
    homogeneity = (Pn / (1.0 + w_con)).sum((1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(Pn * np.log(Pn + 1e-12)).sum((1, 2))
    asm = (Pn ** 2).sum((1, 2))
    return contrast, homogeneity, ent, asm


def extract_object_features(image, labels, n_glcm_levels=16, verbose=False):
    """提取全部图斑特征，返回 DataFrame（index = 图斑编号 1..K）。"""
    img = np.asarray(image, np.float64)
    if img.ndim == 2:
        img = img[:, :, None]
    h, w, nb = img.shape
    k = int(labels.max())
    lab_flat = labels.ravel()
    area = np.bincount(lab_flat, minlength=k + 1)[1:].astype(np.float64)

    feats = {"area": area}
    # --- 光谱均值 / 标准差 ---
    means, stds = [], []
    for b in range(nb):
        s = np.bincount(lab_flat, weights=img[:, :, b].ravel(), minlength=k + 1)[1:]
        s2 = np.bincount(lab_flat, weights=(img[:, :, b] ** 2).ravel(), minlength=k + 1)[1:]
        m = s / area
        v = np.maximum(s2 / area - m ** 2, 0.0)
        means.append(m); stds.append(np.sqrt(v))
        feats[f"mean_b{b}"] = m
        feats[f"std_b{b}"] = np.sqrt(v)
    feats["brightness"] = np.mean(means, axis=0)
    # --- 比值指数（4 波段约定 [B, G, R, NIR]）---
    if nb >= 4:
        nir, red, green = means[3], means[2], means[1]
        eps = 1e-6
        feats["ndvi"] = (nir - red) / np.maximum(nir + red, eps)
        feats["ndwi"] = (green - nir) / np.maximum(green + nir, eps)
    # --- 形状 ---
    per = _object_perimeter(labels)[1:k + 1]
    feats["shape_index"] = per / np.sqrt(np.maximum(area, 1.0))
    slices = ndimage.find_objects(labels)
    bh = np.empty(k); bw_ = np.empty(k)
    for i, sl in enumerate(slices):
        bh[i] = 1 if sl is None else sl[0].stop - sl[0].start
        bw_[i] = 1 if sl is None else sl[1].stop - sl[1].start
    feats["bbox_fill"] = area / np.maximum(bh * bw_, 1.0)
    feats["elongation"] = np.maximum(bh, bw_) / np.maximum(np.minimum(bh, bw_), 1.0)
    # --- GLCM 纹理（全图灰度分位数量化 → 对象内掩膜共现）---
    gray = img.mean(axis=2)
    lo, hi = np.percentile(gray, [1, 99])
    q = np.clip((gray - lo) / max(hi - lo, 1e-9), 0, 1)
    gray_q = (q * (n_glcm_levels - 1)).astype(np.int32)
    con, hom, ent, asm = _masked_glcm(gray_q, labels, n_glcm_levels)
    # _masked_glcm 内部已按 label 从 1 起切掉 0 号占位行，这里直接使用
    feats["glcm_contrast"] = con
    feats["glcm_homogeneity"] = hom
    feats["glcm_entropy"] = ent
    feats["glcm_asm"] = asm

    df = pd.DataFrame(feats, index=pd.Index(np.arange(1, k + 1), name="seg_id"))
    if verbose:
        print(f"    [features] {df.shape[0]} objects × {df.shape[1]} features")
    return df


def object_majority_truth(labels, truth, n_classes=5):
    """每个图斑的真值 = 图斑内占多数的像素真值类别（1..n_classes）。"""
    k = int(labels.max())
    lab_flat = labels.ravel()
    counts = np.stack(
        [np.bincount(lab_flat, weights=(truth == c).ravel().astype(np.float64), minlength=k + 1)[1:]
         for c in range(1, n_classes + 1)], axis=1)          # (K, n_classes)
    return counts.argmax(axis=1) + 1
