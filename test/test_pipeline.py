# -*- coding: utf-8 -*-
"""自动化质检：面向对象分割分类流水线（不依赖 run_demo 产物，自足小场景复现全链路）。

运行：E:\\Ana\\python.exe test\\test_pipeline.py
输出：逐条 PASS/FAIL + 汇总，任何 FAIL 都会使退出码为 1。

覆盖面（A 合成数据 / B 分割核心 / C 特征 / D 分类 / E 输出与对比结论）
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from shapely.geometry import shape

from obia.synthetic import generate_scene, CLASS_NAMES
from obia.segment import multiresolution_segment
from obia.features import extract_object_features, object_majority_truth
from obia.classify import classify_objects, classify_pixels
from obia.vectorize import (write_raster, labels_to_geojson,
                            rasterize_object_pred)
from obia.metrics import confusion_oa, salt_pepper_rate, boundary_f

SEED = 20260829
TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_out")

FAILS = []
N_CHECK = [0]


def check(name, cond, detail=""):
    N_CHECK[0] += 1
    ok = bool(cond)
    print(f"[{'PASS' if ok else 'FAIL'}] #{N_CHECK[0]:02d} {name}"
          + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def main():
    if os.path.isdir(TMP):
        shutil.rmtree(TMP)
    os.makedirs(TMP, exist_ok=True)

    # ============ A. 合成数据 ============
    t0 = time.time()
    img, truth, parcels = generate_scene(256, 256, seed=SEED)
    img2, truth2, parcels2 = generate_scene(256, 256, seed=SEED)
    check("A1 合成影像可复现（两次生成完全一致）",
          np.array_equal(img, img2) and np.array_equal(truth, truth2)
          and np.array_equal(parcels, parcels2))
    check("A2 影像形状/取值合理（float32, [0,1], 4 波段）",
          img.shape == (256, 256, 4) and img.dtype == np.float32
          and 0.0 <= float(img.min()) and float(img.max()) <= 1.0)
    codes = np.unique(truth).tolist()
    check("A3 真值编码 1..5 且五类齐全", codes == [1, 2, 3, 4, 5], str(codes))
    nir_fo = img[:, :, 3][truth == 2].mean()
    nir_wa = img[:, :, 3][truth == 1].mean()
    check("A4 光谱设定合理（林地 NIR 显著高于水体）", nir_fo > nir_wa + 0.2,
          f"forest={nir_fo:.2f} water={nir_wa:.2f}")
    check("A5 图斑数量合理（>= 40 块）", int(parcels.max()) >= 40,
          f"parcels={int(parcels.max())}")

    # ============ B. 分割核心 ============
    t_seg = time.time()
    labels, info = multiresolution_segment(img, scale=8.0, verbose=False)
    seg_seconds = time.time() - t_seg
    k = int(labels.max())
    check("B1 分割全图覆盖（无 0/空洞）", int(labels.min()) >= 1)
    check("B2 图斑编号连续 1..K", np.array_equal(np.unique(labels), np.arange(1, k + 1)))
    check("B3 图斑面积总和 == 像素总数",
          int(np.bincount(labels.ravel(), minlength=k + 1)[1:].sum()) == 256 * 256)
    # 连通性：取最大的 30 个图斑逐一验证 4-连通单分量
    areas = np.bincount(labels.ravel(), minlength=k + 1)[1:]
    top30 = np.argsort(areas)[-30:] + 1
    ncomp = 0
    for sid in top30:
        _, nc = ndimage.label(labels == sid, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        ncomp += (nc == 1)
    check("B4 最大 30 个图斑均 4-连通单分量", ncomp == 30, f"{ncomp}/30")
    labels_b, _ = multiresolution_segment(img, scale=8.0, verbose=False)
    check("B5 分割可复现（同参数两次运行 label 完全一致）",
          np.array_equal(labels, labels_b))
    check("B6 256² 分割耗时 < 120s", seg_seconds < 120, f"{seg_seconds:.1f}s")

    # 尺度单调性 + 嵌套性（128² 快速验证；尺度取值避开"全图并一"的坍缩区）
    img_s, tru_s, par_s = generate_scene(128, 128, seed=SEED)
    lab1, i1 = multiresolution_segment(img_s, scale=0.2, verbose=False)
    lab4, i4 = multiresolution_segment(img_s, scale=1.0, verbose=False)
    lab16, i16 = multiresolution_segment(img_s, scale=4.0, verbose=False)
    check("B7 尺度单调性：scale 增大对象数单调不增",
          i1["n_objects"] >= i4["n_objects"] >= i16["n_objects"] > 1,
          f"{i1['n_objects']} -> {i4['n_objects']} -> {i16['n_objects']}")
    check("B8 小尺度产生过分割（scale=0.2 对象数 > 1000）", i1["n_objects"] > 1000)
    # 嵌套性：每个细尺度（scale=1）对象必须恰好整体落入一个粗尺度（scale=4）对象
    #（合并序列与 scale 无关、阈值只截断 → 粗分割必为细分割的再合并）
    pair = np.stack([lab4.ravel(), lab16.ravel()], 1)   # (细, 粗)
    uniq = np.unique(pair, axis=0)
    owners = {}
    for a, b in uniq:
        owners.setdefault(a, set()).add(b)
    nested = all(len(v) == 1 for v in owners.values())
    check("B9 多尺度层级嵌套（粗尺度对象不被细尺度对象跨越）", nested)

    # ============ C. 特征提取 ============
    feats = extract_object_features(img, labels)
    check("C1 特征表行数 == 图斑数", len(feats) == k, f"{len(feats)} vs {k}")
    check("C2 特征表无 NaN/Inf", np.isfinite(feats.to_numpy()).all())
    check("C3 area 列求和 == 像素总数", int(feats["area"].sum()) == 256 * 256)
    check("C4 指数取值域合理（ndvi∈[-1,1], glcm_contrast≥0, shape_index>0, bbox_fill∈(0,1]）",
          feats["ndvi"].between(-1, 1).all() and (feats["glcm_contrast"] >= 0).all()
          and (feats["shape_index"] > 0).all()
          and feats["bbox_fill"].between(1e-9, 1.0).all())
    check("C5 光谱均值在 [0,1]、std 非负",
          feats[[c for c in feats.columns if c.startswith("mean_")]].to_numpy().min() >= 0
          and feats[[c for c in feats.columns if c.startswith("mean_")]].to_numpy().max() <= 1
          and (feats[[c for c in feats.columns if c.startswith("std_")]].to_numpy() >= 0).all())

    # ============ D. 分类 ============
    obj_truth = object_majority_truth(labels, truth)
    r1 = classify_objects(feats, obj_truth, seed=SEED)
    r2 = classify_objects(feats, obj_truth, seed=SEED)
    check("D1 图斑级分类可复现（同 seed 预测一致）", np.array_equal(r1["pred"], r2["pred"]))
    check("D2 图斑级 OA ≥ 0.85（门槛按实测留余量）", r1["oa"] >= 0.85, f"OA={r1['oa']:.3f}")
    pix = classify_pixels(img, truth, seed=SEED)
    check("D3 像素级基线 OA ≥ 0.80（健康性检查）", pix["oa"] >= 0.80, f"OA={pix['oa']:.3f}")

    # ============ E. 输出与对比结论 ============
    obj_map = rasterize_object_pred(labels, r1["pred"])
    write_raster(os.path.join(TMP, "labels.tif"), labels)
    write_raster(os.path.join(TMP, "classified.tif"), obj_map)
    fc = labels_to_geojson(labels, path=os.path.join(TMP, "segments.geojson"))

    check("E1 GeoJSON 为 FeatureCollection 且要素数 == 图斑数",
          fc["type"] == "FeatureCollection" and len(fc["features"]) == k)
    check("E2 GeoJSON 含 crs 声明", "crs" in fc and "32650" in fc["crs"]["properties"]["name"])
    sids = [f["properties"]["seg_id"] for f in fc["features"]]
    check("E3 GeoJSON seg_id 与图斑一一对应", sorted(sids) == list(range(1, k + 1)))
    # 几何面积守恒：多边形面积之和（m²）== 像素数 × 100（10m 像元）
    poly_area = sum(shape(f["geometry"]).area for f in fc["features"])
    check("E4 矢量面积守恒（Σ多边形面积 == 全影像面积）",
          abs(poly_area - 256 * 256 * 100.0) / (256 * 256 * 100.0) < 1e-6,
          f"{poly_area:.1f} m² vs {256 * 256 * 100} m²")
    with rasterio.open(os.path.join(TMP, "labels.tif")) as src:
        lab_tif = src.read(1)
        check("E5 标签 GeoTIFF 可读且与内存一致",
              lab_tif.shape == labels.shape and np.array_equal(lab_tif.astype(np.int32), labels))
    with rasterio.open(os.path.join(TMP, "classified.tif")) as src:
        cls_tif = src.read(1)
    check("E6 分类栅格 uint8 且类别码 ⊆ {1..5}",
          cls_tif.dtype == np.uint8 and set(np.unique(cls_tif)).issubset(set(range(1, 6))))

    sp_obj = salt_pepper_rate(obj_map)
    sp_pix = salt_pepper_rate(pix["pred"])
    check("E7 椒盐率：面向对象 < 像素级（OBIA 核心卖点成立）", sp_obj < sp_pix,
          f"obj={sp_obj:.4f} vs pix={sp_pix:.4f}")
    bf_obj = boundary_f(obj_map, truth, tol=2)     # 分类边界 vs 真值地类边界（同台口径）
    bf_pix = boundary_f(pix["pred"], truth, tol=2)
    check("E8 边界贴合度记录完整（precision/recall/f1 均在 [0,1]）",
          all(0 <= bf_obj[k2] <= 1 for k2 in ("precision", "recall", "f1"))
          and all(0 <= bf_pix[k2] <= 1 for k2 in ("precision", "recall", "f1")),
          f"obj F1={bf_obj['f1']:.3f} pix F1={bf_pix['f1']:.3f}")
    feats_path = os.path.join(TMP, "object_features.csv")
    feats.to_csv(feats_path, encoding="utf-8-sig")
    feats_rt = pd.read_csv(feats_path).drop(columns=["seg_id"])  # 首列是索引名
    check("E9 特征表 CSV 往返一致（行列数一致）",
          feats_rt.shape == feats.shape and list(feats_rt.columns) == list(feats.columns))

    # ============ 汇总 ============
    total = time.time() - t0
    print("\n" + "=" * 60)
    if FAILS:
        print(f"质检未通过：{len(FAILS)}/{N_CHECK[0]} 项 FAIL -> {FAILS}")
        sys.exit(1)
    print(f"质检全部通过：{N_CHECK[0]}/{N_CHECK[0]} 项 PASS，耗时 {total:.0f}s")
    print(f"关键数字：图斑 {k} 个 | 对象 OA={r1['oa']:.3f} 像素 OA={pix['oa']:.3f} "
          f"| 椒盐率 obj={sp_obj:.4f} pix={sp_pix:.4f} | 边界F obj={bf_obj['f1']:.3f}")
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
