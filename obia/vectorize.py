# -*- coding: utf-8 -*-
"""成果输出：分割矢量 GeoJSON、图斑级/像素级分类栅格 GeoTIFF。

坐标系约定：合成影像配准到 UTM 50N（EPSG:32650），10 m 像元，起点
(500000, 4000000)——让 GeoJSON/GeoTIFF 具备真实 GIS 语义，可直接拖进 QGIS。
"""
from __future__ import annotations

import json

import numpy as np
import rasterio
from affine import Affine
from rasterio.features import shapes

__all__ = ["SCENE_TRANSFORM", "SCENE_CRS", "write_raster",
           "labels_to_geojson", "rasterize_object_pred"]

SCENE_TRANSFORM = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0)
SCENE_CRS = "EPSG:32650"


def write_raster(path, arr, transform=SCENE_TRANSFORM, crs=SCENE_CRS, nodata=None):
    """单波段 GeoTIFF写出（dtype 跟随输入数组）。"""
    arr = np.asarray(arr)
    dtype = arr.dtype.name if arr.dtype != np.int64 else "int32"
    data = arr.astype(dtype)
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
        count=1, dtype=dtype, transform=transform, crs=crs,
        nodata=nodata, compress="deflate",
    ) as dst:
        dst.write(data, 1)


def labels_to_geojson(labels, transform=SCENE_TRANSFORM, crs=SCENE_CRS,
                      properties=None, class_names=None, path=None):
    """图斑栅格 → GeoJSON FeatureCollection。

    Parameters
    ----------
    labels : (H, W) 图斑编号 1..K
    properties : 可选 DataFrame，index = seg_id，逐图斑附加属性
    class_names : 可选 dict {类别码: 名称}，若 properties 含 pred 列则翻译
    path : 给定则写文件

    Returns
    -------
    geojson dict（含 crs 字段）
    """
    lab = np.asarray(labels)
    geoms = shapes(lab.astype(np.int32), connectivity=4, transform=transform)
    feats = []
    for geom, val in geoms:
        sid = int(val)
        props = {"seg_id": sid}
        if properties is not None:
            row = properties.loc[sid]
            for col in properties.columns:
                v = row[col]
                props[col] = v.item() if hasattr(v, "item") else v
        feats.append({
            "type": "Feature",
            "geometry": geom,
            "properties": props,
        })
    fc = {
        "type": "FeatureCollection",
        "name": "obia_segments",
        "crs": {"type": "name", "properties": {"name": crs}},
        "features": feats,
    }
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fc, f, ensure_ascii=False)
    return fc


def rasterize_object_pred(labels, obj_pred):
    """图斑预测类别回填成栅格（面向对象分类结果图）。"""
    out = np.zeros(labels.shape, np.uint8)
    max_lab = int(labels.max())
    lut = np.zeros(max_lab + 1, np.uint8)
    lut[1:max_lab + 1] = np.asarray(obj_pred, np.uint8)
    return lut[labels]
