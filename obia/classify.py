# -*- coding: utf-8 -*-
"""分类：图斑级 RandomForest（主流程） + 像素级 RandomForest（对照基线）。

* 对象级：图斑特征表（光谱+纹理+形状约 19 维融合）→ RF → 每图斑一个类别，
  分类结果按图斑整体回填栅格，天然无椒盐。
* 像素级：逐像素 [4 波段光谱 + 5×5 局部标准差] → RF，作"传统像素分类"对照，
  用来量化面向对象方法在椒盐噪声与边界还原上的收益。
两类分类器同架构（sklearn RF 300 树，同 seed），保证对比公平。
"""
from __future__ import annotations

import time

import numpy as np
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .metrics import confusion_oa

__all__ = ["classify_objects", "classify_pixels"]

RF_PARAMS = dict(n_estimators=300, random_state=20260829, n_jobs=-1)


def classify_objects(features, obj_truth, seed=20260829, train_frac=0.6, verbose=False):
    """图斑级 RF 分类。

    Parameters
    ----------
    features : DataFrame（index = seg_id 1..K）
    obj_truth : 长度 K 的多数票真值（1..C）

    Returns
    -------
    dict(pred=每图斑预测, oa, cm, model, train_idx, test_idx, seconds)
    """
    t0 = time.time()
    X = features.to_numpy(np.float64)
    y = np.asarray(obj_truth)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    idx = np.arange(len(y))
    counts = np.bincount(y)
    if counts[counts > 0].min() >= 2:
        tr, te = train_test_split(idx, train_size=train_frac, random_state=seed,
                                  stratify=y)
        mode = "stratified"
    else:
        # 极端参数下某类图斑数 < 2，无法分层 → 退化为普通随机划分
        tr, te = train_test_split(idx, train_size=train_frac, random_state=seed)
        mode = "random-fallback"
    clf = RandomForestClassifier(**{**RF_PARAMS, "random_state": seed})
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X)
    cm, oa = confusion_oa(pred, y, int(y.max()))
    if verbose:
        print(f"    [classify] objects: OA={oa:.3f} "
              f"({len(tr)} train / {len(te)} test objs, {time.time() - t0:.1f}s)")
    return {"pred": pred, "oa": oa, "cm": cm, "model": clf,
            "train_idx": tr, "test_idx": te, "seconds": time.time() - t0}


def classify_pixels(image, pixel_truth, seed=20260829, n_train=30000, verbose=False):
    """像素级 RF 基线：[各波段 + 5×5 局部标准差] → RF → 全图逐像素类别。"""
    t0 = time.time()
    img = np.asarray(image, np.float32)
    if img.ndim == 2:
        img = img[:, :, None]
    h, w, nb = img.shape
    cols = [img[:, :, b] for b in range(nb)]
    cols += [ndimage.uniform_filter(img[:, :, b]) .astype(np.float32) for b in range(nb)]
    cols += [np.sqrt(np.maximum(ndimage.uniform_filter(img[:, :, b] ** 2)
                                - ndimage.uniform_filter(img[:, :, b]) ** 2, 0)).astype(np.float32)
             for b in range(nb)]
    X = np.stack(cols, axis=2).reshape(-1, len(cols))
    y = pixel_truth.ravel().astype(np.int64)
    rng = np.random.default_rng(seed)
    known = np.where(y > 0)[0]
    # 分层抽样训练像素
    tr_idx = []
    for c in np.unique(y[known]):
        cands = known[y[known] == c]
        take = min(len(cands), max(500, n_train // len(np.unique(y[known]))))
        tr_idx.append(rng.choice(cands, take, replace=False))
    tr = np.concatenate(tr_idx)
    clf = RandomForestClassifier(**{**RF_PARAMS, "random_state": seed})
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X).reshape(h, w).astype(np.int64)
    cm, oa = confusion_oa(pred, pixel_truth, int(pixel_truth.max()))
    if verbose:
        print(f"    [classify] pixels: OA={oa:.3f} ({len(tr)} train px, {time.time() - t0:.1f}s)")
    return {"pred": pred, "oa": oa, "cm": cm, "seconds": time.time() - t0}
