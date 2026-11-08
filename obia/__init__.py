"""obia —— 手写面向对象遥感分割分类流程（复刻 eCognition 多分辨率分割思路）。

模块一览
--------
synthetic : 合成多光谱影像 + 像素级真值（seed 可复现，不依赖外网）
segment   : 手写多尺度区域生长分割（Baatz & Schäpe 异质性准则，核心自研）
features  : 图斑光谱/纹理/形状特征提取
classify  : 图斑级 RandomForest 分类 + 像素级对照基线
vectorize : 分割矢量 GeoJSON / 分类栅格 GeoTIFF 输出
viz       : 多尺度对比图、像素级 vs 对象级对比图、敏感性曲线
metrics   : 总体精度 / 混淆矩阵 / 椒盐率 / 边界贴合度 F
"""

__version__ = "1.0.0"
