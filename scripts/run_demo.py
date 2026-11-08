# -*- coding: utf-8 -*-
"""一键端到端：合成场景 → 手写多尺度分割 → 图斑特征 → RF 分类 → 矢量/栅格/图件。

用法（解释器固定 E:\\Ana\\python.exe）
    E:\\Ana\\python.exe scripts\\run_demo.py                     # 512² 全流程 + 敏感性扫描
    E:\\Ana\\python.exe scripts\\run_demo.py --size 256 --quick  # 快速冒烟（跳过扫描）
    E:\\Ana\\python.exe scripts\\run_demo.py --scale 40 --shape 0.2 --compactness 0.5
    E:\\Ana\\python.exe scripts\\run_demo.py --config config.sample.json   # 读取真实 GeoTIFF
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from obia.synthetic import generate_scene, CLASS_NAMES          # noqa: E402
from obia.segment import multiresolution_segment                # noqa: E402
from obia.features import extract_object_features, object_majority_truth  # noqa: E402
from obia.classify import classify_objects, classify_pixels     # noqa: E402
from obia.vectorize import (write_raster, labels_to_geojson,    # noqa: E402
                            rasterize_object_pred, SCENE_TRANSFORM, SCENE_CRS)
from obia.metrics import confusion_oa, salt_pepper_rate, boundary_f  # noqa: E402
from obia import viz                                            # noqa: E402

SEED = 20260829


def load_real_or_synthetic(args):
    """--config 指向真实 GeoTIFF 时读真实影像（真值可选）；否则用内置合成场景。"""
    if not args.config:
        return generate_scene(args.size, args.size, seed=SEED)
    import rasterio
    with rasterio.open(args.config) as src:
        img = src.read().astype(np.float32)
        img = np.transpose(img, (1, 2, 0))
        truth = None
        if args.truth:
            with rasterio.open(args.truth) as tsrc:
                truth = tsrc.read(1).astype(np.uint8)
    return img, truth, None


def per_class_table(cm, class_names):
    """混淆矩阵 → 逐类 PA(生产者)/UA(用户)/F1 表。"""
    rows = []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        pa = tp / max(cm[i, :].sum(), 1)
        ua = tp / max(cm[:, i].sum(), 1)
        f1 = 2 * pa * ua / max(pa + ua, 1e-9)
        rows.append({"class": name, "pa": pa, "ua": ua, "f1": f1,
                     "support": int(cm[i, :].sum())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="手写面向对象分割分类端到端演示")
    ap.add_argument("--size", type=int, default=512, help="合成影像边长（默认 512）")
    ap.add_argument("--scale", type=float, default=10.0, help="主尺度参数（512² 标定：10 最优）")
    ap.add_argument("--shape", type=float, default=0.2, help="形状权重")
    ap.add_argument("--compactness", type=float, default=0.5, help="紧致度权重")
    ap.add_argument("--config", default="", help="真实影像 GeoTIFF 路径（可选）")
    ap.add_argument("--truth", default="", help="真实影像真值栅格路径（可选）")
    ap.add_argument("--quick", action="store_true", help="快速模式：跳过敏感性扫描")
    ap.add_argument("--skip-sensitivity", dest="skip_sens", action="store_true")
    args = ap.parse_args()

    t_all = time.time()
    data_dir = os.path.join(PROJ, "data")
    res_dir = os.path.join(PROJ, "results")
    fig_dir = os.path.join(res_dir, "figs")
    for d in (data_dir, res_dir, fig_dir):
        os.makedirs(d, exist_ok=True)

    print(f"[1/9] 生成/读取影像 ...", flush=True)
    image, truth, parcels = load_real_or_synthetic(args)
    if truth is None:
        raise SystemExit("没有真值栅格（合成模式自动生成；真实影像请用 --truth 提供），无法做分类评估。")
    h, w = image.shape[:2]
    print(f"    影像 {h}×{w}×{image.shape[2]}，seed={SEED}", flush=True)
    if truth is not None:
        write_raster(os.path.join(data_dir, "truth.tif"), truth)
    if parcels is not None:
        write_raster(os.path.join(data_dir, "parcels.tif"), parcels)
    write_bands(image, os.path.join(data_dir, "synthetic.tif"))

    print("[2/9] 手写多尺度区域生长分割（核心，Baatz & Schäpe 准则）...", flush=True)
    labels, info = multiresolution_segment(image, scale=args.scale,
                                           shape=args.shape,
                                           compactness=args.compactness)
    k = info["n_objects"]
    write_raster(os.path.join(data_dir, "labels_main.tif"), labels)

    print("[3/9] 提取图斑特征（光谱均值/方差 + GLCM 纹理 + 形状/比值）...", flush=True)
    feats = extract_object_features(image, labels, verbose=True)
    feats.to_csv(os.path.join(res_dir, "object_features.csv"), encoding="utf-8-sig")

    print("[4/9] 图斑级 RandomForest 分类 ...", flush=True)
    obj_truth = object_majority_truth(labels, truth) if truth is not None else None
    obj_res = classify_objects(feats, obj_truth, seed=SEED, verbose=True)
    feats_out = feats.copy()
    feats_out["truth"] = obj_truth
    feats_out["pred"] = obj_res["pred"]
    feats_out["pred_name"] = [CLASS_NAMES[c - 1] for c in obj_res["pred"]]
    feats_out.to_csv(os.path.join(res_dir, "object_features.csv"),
                     encoding="utf-8-sig")  # 覆写为含预测的版本
    obj_map = rasterize_object_pred(labels, obj_res["pred"])
    write_raster(os.path.join(res_dir, "classified_objects.tif"), obj_map)

    print("[5/9] 像素级 RandomForest 基线 + skimage SLIC 超像素对照 ...", flush=True)
    pix_res = classify_pixels(image, truth, seed=SEED, verbose=True)
    write_raster(os.path.join(res_dir, "classified_pixels.tif"),
                 pix_res["pred"].astype(np.uint8))
    slic_res = None
    try:
        from obia.baselines import slic_baseline
        slic_res = slic_baseline(image, truth, parcels, n_segments=440, seed=SEED)
    except Exception as e:                       # 对照失败不拖垮主流程
        print(f"    [SLIC 对照] 跳过：{type(e).__name__}: {e}", flush=True)

    print("[6/9] 精度指标与对比 ...", flush=True)
    cm_obj, oa_obj = confusion_oa(obj_map, truth, len(CLASS_NAMES))
    cm_pix, oa_pix = confusion_oa(pix_res["pred"], truth, len(CLASS_NAMES))
    sp_obj = salt_pepper_rate(obj_map)
    sp_pix = salt_pepper_rate(pix_res["pred"])
    # 边界还原口径：两套分类结果图都对"真值地类边界"比（公平同台）；
    # 分割本身的贴合度在敏感性扫描里对"真值图斑边界"单独评估。
    bf_obj = boundary_f(obj_map, truth, tol=2)
    bf_pix = boundary_f(pix_res["pred"], truth, tol=2)
    # fig2 面板标题统一用全图口径 OA（obj_res["oa"] 原为测试图斑口径，混口径会误导）
    obj_res["oa"] = oa_obj
    print(f"    对象级 OA={oa_obj:.3f} 椒盐率={sp_obj:.4f} 边界F={bf_obj.get('f1', float('nan')):.3f}")
    print(f"    像素级 OA={oa_pix:.3f} 椒盐率={sp_pix:.4f} 边界F={bf_pix.get('f1', float('nan')):.3f}")
    per_class_table(cm_obj, CLASS_NAMES).to_csv(
        os.path.join(res_dir, "accuracy_per_class_object.csv"), index=False,
        encoding="utf-8-sig")
    pd.DataFrame(cm_obj, index=[f"真值_{n}" for n in CLASS_NAMES],
                 columns=[f"预测_{n}" for n in CLASS_NAMES]).to_csv(
        os.path.join(res_dir, "confusion_object.csv"), encoding="utf-8-sig")
    pd.DataFrame(cm_pix, index=[f"真值_{n}" for n in CLASS_NAMES],
                 columns=[f"预测_{n}" for n in CLASS_NAMES]).to_csv(
        os.path.join(res_dir, "confusion_pixel.csv"), encoding="utf-8-sig")

    print("[7/9] 输出分割矢量 GeoJSON ...", flush=True)
    prop_cols = ["area", "mean_b0", "mean_b1", "mean_b2", "mean_b3",
                 "ndvi", "glcm_contrast", "shape_index", "truth", "pred"]
    props = feats_out[prop_cols] if all(c in feats_out for c in prop_cols) else feats_out
    fc = labels_to_geojson(labels, properties=props, path=os.path.join(res_dir, "segments.geojson"))
    print(f"    segments.geojson: {len(fc['features'])} 个图斑要素", flush=True)

    print("[8/9] 参数敏感性扫描（scale/shape/compactness）...", flush=True)
    rows = []
    scale_labels = {}          # 复用给 fig1，避免重复分割
    if not (args.quick or args.skip_sens):
        scan = [(("scale", s), dict(scale=s)) for s in (5.0, 20.0, 40.0, 80.0, 160.0)]
        scan += [(("shape", v), dict(shape=v)) for v in (0.0, 0.1, 0.3, 0.5)]
        scan += [(("compactness", v), dict(compactness=v)) for v in (0.0, 0.3, 0.8)]
        for (pname, pval), kw in scan:
            kw_full = dict(scale=args.scale, shape=args.shape,
                           compactness=args.compactness)
            kw_full.update(kw)
            lab_s, info_s = multiresolution_segment(image, verbose=False, **kw_full)
            if pname == "scale":
                scale_labels[pval] = (lab_s, info_s)
            f_s = extract_object_features(image, lab_s)
            ot_s = object_majority_truth(lab_s, truth)
            r_s = classify_objects(f_s, ot_s, seed=SEED)
            # 分割质量口径：分割边界（label 图）对真值图斑边界的贴合度
            bfx = boundary_f(lab_s, parcels, tol=2) if parcels is not None else {
                "precision": np.nan, "recall": np.nan, "f1": np.nan}
            rows.append({
                "param": pname, "value": pval,
                "n_objects": info_s["n_objects"],
                "mean_area": float(info_s["n_pixels"]) / info_s["n_objects"],
                "seconds": info_s["seconds"],
                "boundary_prec": bfx["precision"], "boundary_rec": bfx["recall"],
                "boundary_f": bfx["f1"],
                "obj_oa": r_s["oa"], "pix_oa": oa_pix,
            })
            print(f"    扫描 {pname}={pval}: 对象 {info_s['n_objects']}, "
                  f"OA={r_s['oa']:.3f}, 边界F={bfx['f1']:.3f}", flush=True)
            # 逐行落盘：中途异常不丢已完成扫描点
            pd.DataFrame(rows).to_csv(os.path.join(res_dir, "sensitivity.csv"),
                                      index=False, encoding="utf-8-sig")

    print("[9/9] 出图：多尺度对比 / 像素级 vs 对象级 / 敏感性曲线 ...", flush=True)
    # 多尺度对比图：主尺度直接复用第 2 步结果，其余尺度复用扫描结果
    scale_labels[args.scale] = (labels, info)
    scale_runs = []
    for s in sorted({args.scale, 20.0, 40.0, 80.0}):
        if s in scale_labels:
            lab_s, info_s = scale_labels[s]
        else:
            lab_s, info_s = multiresolution_segment(
                image, scale=s, shape=args.shape, compactness=args.compactness,
                verbose=False)
            scale_labels[s] = (lab_s, info_s)
        scale_runs.append((s, lab_s, info_s))
    viz.fig_multiscale(image, parcels, scale_runs,
                       os.path.join(fig_dir, "fig1_multiscale_compare.png"))
    obj_res["speckle"] = sp_obj
    pix_res["speckle"] = sp_pix
    viz.fig_pixel_vs_object(image, truth, pix_res, obj_res, obj_map,
                            os.path.join(fig_dir, "fig2_pixel_vs_object.png"))
    if rows:
        viz.fig_sensitivity(rows, args.scale,
                            os.path.join(fig_dir, "fig3_sensitivity.png"))
    else:
        print("    （--quick/--skip-sensitivity：跳过敏感性曲线）", flush=True)

    summary = {
        "seed": SEED,
        "image": {"h": int(h), "w": int(w), "bands": int(image.shape[2])},
        "params": {"scale": args.scale, "shape": args.shape,
                   "compactness": args.compactness},
        "segmentation": info,
        "object_rf": {"oa": oa_obj, "speckle": sp_obj,
                      "boundary_f": bf_obj.get("f1"), "seconds": obj_res["seconds"],
                      "confusion": cm_obj.tolist()},
        "pixel_rf": {"oa": oa_pix, "speckle": sp_pix,
                     "boundary_f": bf_pix.get("f1"), "seconds": pix_res["seconds"],
                     "confusion": cm_pix.tolist()},
        "slic_baseline": slic_res,
        "n_geojson_features": len(fc["features"]),
        "total_seconds": time.time() - t_all,
    }
    with open(os.path.join(res_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n==== 端到端完成，总耗时 {summary['total_seconds']:.0f}s ====")
    print(json.dumps({k: summary[k] for k in ("segmentation", "object_rf", "pixel_rf")},
                     ensure_ascii=False, indent=1))


def write_bands(image, path):
    """多波段影像 GeoTIFF 写出。"""
    import rasterio
    from rasterio.transform import Affine as _A
    arr = np.transpose(image, (2, 0, 1)).astype(np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1],
                       width=arr.shape[2], count=arr.shape[0], dtype="float32",
                       transform=SCENE_TRANSFORM, crs=SCENE_CRS,
                       compress="deflate") as dst:
        dst.write(arr)


if __name__ == "__main__":
    main()
