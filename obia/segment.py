# -*- coding: utf-8 -*-
"""手写多尺度区域生长分割（Multiresolution Segmentation，复刻 eCognition 思路）。

原理（Baatz & Schäpe 2000 / eCognition 多分辨率分割）
-----------------------------------------------------
从单像素对象出发，反复挑选"合并代价最小"的相邻对象对执行合并，直到最小合并
代价超过尺度阈值 scale。合并代价 f 由光谱异质性与形状异质性加权构成：

    f = w_color * Δh_color + w_shape * Δh_shape ,  w_color = 1 - w_shape
    Δh_color = Σ_b w_b * [ n_m σ_m,b - n_a σ_a,b - n_b σ_b,b ]        (标准差口径)
    Δh_shape = w_smooth * Δh_smooth + w_cpt * Δh_cpt ,  w_smooth = 1 - w_cpt
    Δh_smooth = n_m (b_m/l_m) - n_a (b_a/l_a) - n_b (b_b/l_b)         (光滑度: 周长/外包矩形周长)
    Δh_cpt    = n_m (b_m/√n_m) - n_a (b_a/√n_a) - n_b (b_b/√n_b)      (紧致度: 周长/√面积)

其中 n 为对象像素数，σ_b 为波段内标准差，b 为对象周长（像素边数），l 为外包
矩形周长，m/a/b 分别代表合并后/两源对象。

实现要点（本文件为手写核心，不依赖任何面向对象解译 SDK）
--------------------------------------------------------
* 对象存储：以初始像素 id 为下标的一维数组（计数/各波段和/平方和/周长/外包框），
  合并即数值累加，异质性增量 O(波段数) 内算完；
* 邻接图：dict-of-dict，值 = 两对象共享边界长度；合并时邻接表按"边权加和"融合；
* 优先队列：相邻对象对代价入堆，合并后惰性失效（弹堆时校验双方是否仍是活跃根
  节点），每次合并向其全部邻居重推新代价；
* 停止准则：堆顶（全局最小代价）> scale 即停——合并序列与 scale 无关，scale 只
  截断序列，因此不同 scale 的分割结果天然嵌套（这正是"多尺度层级"的来源）。

复杂度：N 像素 → 约 N 次合并 × 平均邻居数次代价计算，纯 Python 下 512×512 约
1~2 分钟量级（见 README 耗时表）。
"""
from __future__ import annotations

import heapq
import time
from math import sqrt

import numpy as np

__all__ = ["multiresolution_segment"]


def _initial_perimeter_bbox(h, w):
    """单像素对象的初始周长（像素边数，含影像边界边）与外包框下标。"""
    per = np.full(h * w, 4.0)
    col = np.arange(h * w) % w
    row = np.arange(h * w) // w
    per[col == w - 1] -= 1.0   # 无右邻
    per[col == 0] -= 1.0       # 无左邻
    per[row == h - 1] -= 1.0   # 无下邻
    per[row == 0] -= 1.0       # 无上邻
    r0 = row.astype(np.int64)
    c0 = col.astype(np.int64)
    return per, r0, c0, r0.copy(), c0.copy()


def multiresolution_segment(image, scale=40.0, shape=0.2, compactness=0.5,
                            band_weights=None, min_objects=1, verbose=True):
    """多尺度区域生长分割（主入口）。

    Parameters
    ----------
    image : (H, W) 或 (H, W, B) ndarray, float
    scale : float
        合并代价阈值（eCognition 的 Scale Parameter）：越大对象越大、数量越少。
        由于合并序列与 scale 无关，不同 scale 的结果严格嵌套。
    shape : float, 0~1
        形状异质性权重 w_shape（eCognition 的 Shape），光谱权重 = 1 - shape。
    compactness : float, 0~1
        紧致度权重 w_cpt（eCognition 的 Compactness），光滑度权重 = 1 - compactness。
    band_weights : 可选，长度 B 的波段权重，默认等权。
    min_objects : int
        保底对象数（代价再小也不少于此数），防御性参数。

    Returns
    -------
    labels : (H, W) int32，图斑编号 1..K
    info   : dict，含 n_objects / n_merges / seconds 等运行信息
    """
    t0 = time.time()
    img = np.asarray(image, np.float64)
    if img.ndim == 2:
        img = img[:, :, None]
    h, w, nb = img.shape
    n = h * w
    if band_weights is None:
        bw = [1.0 / nb] * nb
    else:
        s = float(sum(band_weights))
        bw = [float(x) / s for x in band_weights]
    w_shape = float(shape)
    w_color = 1.0 - w_shape
    w_smooth = 1.0 - float(compactness)
    w_cpt = float(compactness)

    flat = img.reshape(n, nb)
    # 一维统计数组，下标 = 对象id*nb + 波段k
    cnt = np.ones(n, np.float64)
    sm = flat.ravel().copy()
    sm2 = (flat * flat).ravel().copy()
    per, r0, c0, r1, c1 = _initial_perimeter_bbox(h, w)

    # ---- 邻接图：{obj: {nbr: shared_border}}，右侧/下侧建边避免重复 ----
    adj = {}
    idx = np.arange(n).reshape(h, w)
    for a, b in np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1):
        adj.setdefault(int(a), {})[int(b)] = 1.0
        adj.setdefault(int(b), {})[int(a)] = 1.0
    for a, b in np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], 1):
        adj.setdefault(int(a), {})[int(b)] = 1.0
        adj.setdefault(int(b), {})[int(a)] = 1.0

    root = list(range(n))  # union-find：root[b]=a 表示 b 已并入 a

    # ---- 合并代价（纯 Python 浮点，避免小数组的 numpy 调用开销）----
    def cost(a, b):
        na = cnt[a]; nb_ = cnt[b]; nm = na + nb_
        f = 0.0
        for k in range(nb):
            j = a * nb + k
            sa = sm[j]; qa = sm2[j]
            j = b * nb + k
            sb = sm[j]; qb = sm2[j]
            ma = sa / na; mb = sb / nb_; mm = (sa + sb) / nm
            va = qa / na - ma * ma; vb = qb / nb_ - mb * mb
            vm = (qa + qb) / nm - mm * mm
            f += bw[k] * (nm * sqrt(vm if vm > 0 else 0.0)
                          - na * sqrt(va if va > 0 else 0.0)
                          - nb_ * sqrt(vb if vb > 0 else 0.0))
        shared = adj[a].get(b, 0.0)
        bm = per[a] + per[b] - 2.0 * shared
        la = 2.0 * ((r1[a] - r0[a] + 1) + (c1[a] - c0[a] + 1))
        lb = 2.0 * ((r1[b] - r0[b] + 1) + (c1[b] - c0[b] + 1))
        lm = 2.0 * ((max(r1[a], r1[b]) - min(r0[a], r0[b]) + 1)
                    + (max(c1[a], c1[b]) - min(c0[a], c0[b]) + 1))
        h_smooth = nm * (bm / lm) - na * (per[a] / la) - nb_ * (per[b] / lb)
        h_cpt = nm * (bm / sqrt(nm)) - na * (per[a] / sqrt(na)) - nb_ * (per[b] / sqrt(nb_))
        h_shape = w_smooth * h_smooth + w_cpt * h_cpt
        return w_color * f + w_shape * h_shape

    # ---- 初始堆：全部相邻像素对 ----
    heap = []
    for a, nbrs in adj.items():
        for b in nbrs:
            if a < b:
                heap.append((cost(a, b), a, b))
    heapq.heapify(heap)

    n_merges = 0
    alive = n
    while heap and alive > min_objects:
        f, a, b = heapq.heappop(heap)
        if root[a] != a or root[b] != b:
            continue                      # 惰性失效：有一方已被合并
        if f > scale:
            break                         # 全局最小代价已超阈值 → 停止
        if b < a:
            a, b = b, a
        # --- 统计融合（b 并入 a）---
        cnt[a] = cnt[a] + cnt[b]
        sm[a * nb:(a + 1) * nb] += sm[b * nb:(b + 1) * nb]
        sm2[a * nb:(a + 1) * nb] += sm2[b * nb:(b + 1) * nb]
        per[a] = per[a] + per[b] - 2.0 * adj[a].get(b, 0.0)
        if r0[b] < r0[a]: r0[a] = r0[b]
        if c0[b] < c0[a]: c0[a] = c0[b]
        if r1[b] > r1[a]: r1[a] = r1[b]
        if c1[b] > c1[a]: c1[a] = c1[b]
        # --- 邻接表按边权加和融合 ---
        adj_a = adj.pop(a); adj_b = adj.pop(b)
        adj_a.pop(b, None); adj_b.pop(a, None)
        merged = adj_a
        for k, v in adj_b.items():
            merged[k] = merged.get(k, 0.0) + v
        for k, v in merged.items():
            dk = adj[k]
            dk.pop(b, None)
            dk[a] = v
        adj[a] = merged
        root[b] = a
        # --- 与新邻居重算代价入堆 ---
        for k in merged:
            heapq.heappush(heap, (cost(a, k), a, k))
        n_merges += 1
        alive -= 1
        if verbose and n_merges % 50000 == 0:
            print(f"    [MRS] merges={n_merges} objects={alive} "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)

    # ---- union-find 路径压缩（指针倍增，向量化）----
    lab = np.asarray(root, np.int64)
    while True:
        nxt = lab[lab]
        if np.array_equal(nxt, lab):
            break
        lab = nxt
    lab = lab.reshape(h, w)
    uniq = np.unique(lab)
    remap = np.zeros(int(uniq.max()) + 1, np.int32)
    remap[uniq] = np.arange(1, len(uniq) + 1, dtype=np.int32)
    labels = remap[lab].astype(np.int32)

    info = {
        "n_objects": int(len(uniq)),
        "n_merges": int(n_merges),
        "n_pixels": int(n),
        "scale": float(scale),
        "shape": float(shape),
        "compactness": float(compactness),
        "seconds": time.time() - t0,
    }
    if verbose:
        print(f"    [MRS] done: {info['n_objects']} objects, "
              f"{info['n_merges']} merges, {info['seconds']:.1f}s", flush=True)
    return labels, info


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from obia.synthetic import generate_scene

    img, tru, par = generate_scene(128, 128)
    for s in (0.1, 0.5, 2.0):
        lab, info = multiresolution_segment(img, scale=s, verbose=False)
        print(f"scale={s}: {info['n_objects']} objects, {info['seconds']:.1f}s")
