# -*- coding: utf-8 -*-
"""合成多光谱遥感影像生成器（带像素级真值与图斑真值，seed 可复现）。

设计动机
--------
复刻 eCognition 工作流需要一幅"地类图斑边界清晰、类内同质性受控、类间光谱可分、
自带地物纹理差异"的影像。真实影像没有像素级真值，本模块用固定随机种子的
Voronoi 图斑场 + 分类别纹理合成，同时输出：

* image  : (H, W, 4) float32，波段顺序 [Blue, Green, Red, NIR]，取值约 [0, 1]
* truth  : (H, W) uint8，地类编码 1..5（像素级真值，用于分类精度评估）
* parcels: (H, W) int32，图斑编号（Voronoi 田块，用于边界贴合度评估）

五类地物的光谱/纹理设定（近似 Sentinel-2 波段量级）：
  water    水    —— 全波段暗、NIR 极低、表面光滑（低频微小起伏）
  forest   林地  —— NIR 最高、高频颗粒纹理（冠层颗粒感）
  farmland 耕地  —— NIR 较高、条垄纹理（随机方向的周期条纹）
  soil     裸土  —— 可见光较亮、四波段接近、中频斑块纹理
  builtup  建设  —— 全波段亮且平、块状高对比纹理（建筑物块）
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

# 地类定义：名称 -> (编码, [B, G, R, NIR] 基线光谱)
CLASSES = {
    "water":    (1, [0.20, 0.18, 0.15, 0.08]),
    "forest":   (2, [0.08, 0.14, 0.09, 0.55]),
    "farmland": (3, [0.12, 0.20, 0.13, 0.42]),
    "soil":     (4, [0.34, 0.37, 0.40, 0.46]),
    "builtup":  (5, [0.42, 0.44, 0.46, 0.52]),
}
CLASS_NAMES = [name for name, _ in sorted(CLASSES.items(), key=lambda kv: kv[1][0])]
CLASS_CODES = [CLASSES[n][0] for n in CLASS_NAMES]
BAND_NAMES = ["Blue", "Green", "Red", "NIR"]

DEFAULT_SEED = 20260829


def _voronoi_fields(h, w, seeds, values):
    """按最近种子给每个像素赋值（欧氏距离 Voronoi）。values 与 seeds 等长。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.empty((len(seeds), h, w), np.float32)
    for i, (sy, sx) in enumerate(seeds):
        d[i] = (yy - sy) ** 2 + (xx - sx) ** 2
    idx = np.argmin(d, axis=0)
    out = np.asarray(values, np.int32)[idx]
    return out, idx  # 赋值结果 + 每像素最近种子编号


def _poisson_disc_seeds(rng, h, w, n, min_dist):
    """简单蓝噪声采样：随机撒点 + 最小间距拒绝，保证图斑大小均匀。"""
    seeds = []
    tries = 0
    while len(seeds) < n and tries < n * 400:
        tries += 1
        y, x = rng.uniform(0, h), rng.uniform(0, w)
        if all((y - sy) ** 2 + (x - sx) ** 2 >= min_dist ** 2 for sy, sx in seeds):
            seeds.append((y, x))
    return seeds


def generate_scene(h=512, w=512, seed=DEFAULT_SEED, n_regions=12, n_parcels=70):
    """生成合成场景。

    Returns
    -------
    image   : (h, w, 4) float32
    truth   : (h, w) uint8   地类编码 1..5
    parcels : (h, w) int32   图斑编号 1..n（种子顺序，供边界贴合度评估）
    """
    rng = np.random.default_rng(seed)

    # --- 1) 类别分区：n_regions 个"大类区域"的 Voronoi，保证同类地域连片 ---
    region_seeds = _poisson_disc_seeds(rng, h, w, n_regions, min_dist=0.35 * (h * w / n_regions) ** 0.5)
    # 轮盘保证 5 类都出现：前 5 个区域固定 5 类，其余随机
    region_cls = [1, 2, 3, 4, 5] + [int(rng.integers(1, 6)) for _ in range(n_regions - 5)]
    rng.shuffle(region_cls)
    region_map, _ = _voronoi_fields(h, w, region_seeds, region_cls)

    # --- 2) 图斑细分：n_parcels 个田块种子落在哪个大类区域就属于哪个类 ---
    parcel_seeds = _poisson_disc_seeds(rng, h, w, n_parcels, min_dist=0.55 * (h * w / n_parcels) ** 0.5)
    parcel_cls = [int(region_map[int(y), int(x)]) for y, x in parcel_seeds]
    parcel_labels, nearest = _voronoi_fields(h, w, parcel_seeds, np.arange(1, len(parcel_seeds) + 1))
    parcels = parcel_labels
    truth = np.zeros((h, w), np.uint8)
    for i, c in enumerate(parcel_cls):
        truth[nearest == i] = c

    # --- 3) 基线光谱 + 类别纹理 ---
    bands = np.empty((4, h, w), np.float32)
    for b in range(4):
        base = np.zeros((h, w), np.float32)
        for name, (code, spec) in CLASSES.items():
            base[truth == code] = spec[b]
        bands[b] = base

    nz = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    tex = np.zeros((h, w), np.float32)
    tex += 0.045 * ndimage.gaussian_filter(nz, 25.0)          # 全图低频光照起伏
    tex[truth == 1] += 0.015 * ndimage.gaussian_filter(nz, 6.0)[truth == 1]          # 水：极平滑
    tex[truth == 2] += (0.09 * ndimage.gaussian_filter(nz, 1.2) + 0.03 * nz)[truth == 2]  # 林：高频颗粒
    tex[truth == 4] += 0.05 * ndimage.gaussian_filter(nz, 4.0)[truth == 4]           # 土：中频斑块
    blk = ndimage.zoom(rng.normal(0, 1, (h // 8, w // 8)), 8, order=0)[:h, :w]        # 建：8px 块状
    tex[truth == 5] += 0.085 * blk[truth == 5]

    # 耕地条垄：逐图斑随机方向的周期条纹
    for i in range(len(parcel_seeds)):
        if parcel_cls[i] != 3:
            continue
        m = parcels == i + 1
        if not m.any():
            continue
        th = rng.uniform(0, np.pi)
        cy, cx = parcel_seeds[i]
        u = np.cos(th) * (xx - cx) + np.sin(th) * (yy - cy)
        tex[m] += 0.045 * np.sin(2 * np.pi * u / rng.uniform(9, 16))[m]

    # 图斑亮度/光谱抖动（同类不同田块有差异，逼真且防"逐像素查表"取巧）
    jitter = np.zeros((h, w), np.float32)
    for i in range(len(parcel_seeds)):
        m = parcels == i + 1
        jitter[m] = rng.normal(0.0, 0.018)

    # --- 4) 逐波段合成 + 噪声 + 边界混合像元压暗 ---
    for b in range(4):
        bands[b] += jitter * (0.6 + 0.4 * b / 3.0) + rng.normal(0, 0.028, (h, w))
    bands += tex  # 纹理对四波段共享（结构项）
    # 边界混合像元：图斑交界处 1px 压暗
    pb = (parcels != np.roll(parcels, 1, 0)) | (parcels != np.roll(parcels, 1, 1))
    bands[:, pb] *= 0.90

    image = np.clip(np.transpose(bands, (1, 2, 0)), 0.0, 1.0).astype(np.float32)
    return image, truth, parcels.astype(np.int32)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    img, tru, par = generate_scene(256, 256)
    print("image", img.shape, img.dtype, "range", float(img.min()), float(img.max()))
    print("truth codes", np.unique(tru), "| parcels", par.max())
