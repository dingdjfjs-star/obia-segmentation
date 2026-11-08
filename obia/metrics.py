# -*- coding: utf-8 -*-
"""精度指标：总体精度 / 混淆矩阵 / 椒盐率 / 边界贴合度 F。

* 椒盐率（salt-pepper rate）：分类图中某像素类别 ≠ 其 3×3 邻域多数类 的像素占比，
  度量分类结果的"碎斑"程度——像素级分类的典型病，面向对象分类的卖点。
* 边界贴合度 F：预测对象边界与真值图斑边界在容差 d 像素内的双向匹配 F1，
  度量分割/分类结果对真实地物边界的还原程度。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["confusion_oa", "salt_pepper_rate", "boundary_f", "mode_filter"]


def mode_filter(cls_map, size=3):
    """3×3（或 size×size）多数类滤波，用于椒盐率统计。"""
    codes = np.unique(cls_map)
    codes = codes[codes > 0]
    votes = np.stack([(cls_map == c) for c in codes], 0).astype(np.float32)
    cnt = np.stack([ndimage.uniform_filter(v, size=size) for v in votes], 0)
    return codes[np.argmax(cnt, axis=0)]


def confusion_oa(pred, truth, n_classes):
    """混淆矩阵（行=真值，列=预测）与总体精度 OA。"""
    t = truth.ravel().astype(np.int64)
    p = pred.ravel().astype(np.int64)
    valid = (t >= 1) & (t <= n_classes) & (p >= 1) & (p <= n_classes)
    cm = np.bincount((t[valid] - 1) * n_classes + (p[valid] - 1),
                     minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    oa = float(np.trace(cm)) / max(float(cm.sum()), 1.0)
    return cm, oa


def salt_pepper_rate(cls_map, size=3):
    """分类图中"与其 3×3 邻域多数类不同"的像素占比（0~1，越小越干净）。"""
    maj = mode_filter(cls_map, size=size)
    return float((cls_map != maj).mean())


def _boundary_mask(labels_or_map):
    """边界像素 = 与右/下邻居类别不同的像素（含影像边缘不视作边界）。"""
    m = np.zeros(labels_or_map.shape, bool)
    m[:, :-1] |= labels_or_map[:, :-1] != labels_or_map[:, 1:]
    m[:-1, :] |= labels_or_map[:-1, :] != labels_or_map[1:, :]
    return m


def boundary_f(pred_map, truth_map, tol=2):
    """边界贴合度 F1：在容差 tol 像素内互相命中的边界 F1。"""
    bp = _boundary_mask(pred_map)
    bt = _boundary_mask(truth_map)
    dp = ndimage.binary_dilation(bp, iterations=tol)
    dt = ndimage.binary_dilation(bt, iterations=tol)
    prec = float((bp & dt).sum()) / max(float(bp.sum()), 1.0)
    rec = float((bt & dp).sum()) / max(float(bt.sum()), 1.0)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1}
