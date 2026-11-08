# -*- coding: utf-8 -*-
"""对照基线：skimage SLIC 超像素（现成算法，仅作对照，不作主流程）。

任务约束：核心分割逻辑必须手写（obia/segment.py 的多分辨率区域生长），
skimage 的现成分割算法只允许作为对照基线入图——本模块即该对照：
同样走"图斑特征 → RandomForest → 全图回填"的下游，与前文主流程同协议对比，
用来回答"手写 MRS 与经典超像素方法相比处在什么水平"。
"""
from __future__ import annotations

import time

import numpy as np
from skimage.segmentation import slic

from .features import extract_object_features, object_majority_truth
from .classify import classify_objects
from .vectorize import rasterize_object_pred
from .metrics import confusion_oa, boundary_f, salt_pepper_rate

__all__ = ["slic_baseline"]


def slic_baseline(image, truth, parcels, n_segments=440, seed=20260829,
                  tol_px=2, verbose=True):
    """skimage SLIC 超像素对照：分割 → 特征 → RF → 全图回填 → 指标。

    Returns dict(n_objects, obj_oa, map_oa, speckle, boundary_f, seconds)
    """
    t0 = time.time()
    img = np.asarray(image, np.float32)
    labels = slic(img, n_segments=n_segments, compactness=10.0,
                  channel_axis=-1 if img.ndim == 3 else None,
                  start_label=1, enforce_connectivity=True)
    labels = labels.astype(np.int32)
    # SLIC 可能产生少量不连续 label，重映射为连续 1..K
    uniq = np.unique(labels)
    remap = np.zeros(int(uniq.max()) + 1, np.int32)
    remap[uniq] = np.arange(1, len(uniq) + 1)
    labels = remap[labels]

    feats = extract_object_features(img, labels)
    obj_truth = object_majority_truth(labels, truth)
    res = classify_objects(feats, obj_truth, seed=seed)
    obj_map = rasterize_object_pred(labels, res["pred"])
    cm, map_oa = confusion_oa(obj_map, truth, int(truth.max()))
    out = {
        "n_objects": int(labels.max()),
        "obj_oa": float(res["oa"]),
        "map_oa": float(map_oa),
        "speckle": salt_pepper_rate(obj_map),
        "boundary_f": boundary_f(obj_map, truth, tol=tol_px)["f1"],
        "seconds": time.time() - t0,
    }
    if verbose:
        print(f"    [SLIC 对照] 对象 {out['n_objects']}, 全图 OA={map_oa:.3f}, "
              f"椒盐率={out['speckle']:.4f}, 边界F={out['boundary_f']:.3f}, "
              f"{out['seconds']:.0f}s", flush=True)
    return out
