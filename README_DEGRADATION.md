# 高斯点云退化操作说明

## 已完成的工作

成功对高斯点云模型 `3a79b9aefafb0b8d.ply` 应用了三种退化操作。

## 输入模型
- **文件**: `3a79b9aefafb0b8d.ply`
- **高斯点数量**: 345,600
- **文件大小**: 23 MB

## 生成的退化模型

### 1. 遮挡退化 (output_occlusion_degraded.ply)
- **剔除比例**: 28.8%
- **剩余高斯点**: 246,052
- **文件大小**: 16 MB
- **效果**: 模拟从新视角观察时被遮挡的高斯点被剔除

### 2. 飞边退化 (output_flying_edge_degraded.ply)
- **基础**: 遮挡退化后的模型
- **高斯点数**: 246,052
- **位置变化**: 平均 10.35, 最大 43.18
- **文件大小**: 16 MB
- **效果**: 使用5x5几何滤波器模拟飞边效应，高斯点位置产生轻微偏移

### 3. 畸变退化 (output_distortion_degraded.ply)
- **基础**: 遮挡退化后的模型
- **高斯点数**: 246,052
- **位置变化**: 平均 10.35, 最大 42.52
- **文件大小**: 16 MB
- **效果**: 使用15x15大核几何滤波器模拟严重畸变，高斯点位置产生更大偏移

## 使用的脚本

### 主脚本: `run_degradation.py`
完整的退化流程脚本，实现了基于深度渲染的高斯点云退化。

**使用方法**:
```bash
python run_degradation.py
```

**退化流程**:
1. 加载 PLY 格式的高斯点云模型
2. 设置相机参数（自动根据场景大小调整）
3. 应用遮挡退化：从4个新视角渲染深度图，剔除被遮挡的高斯点
4. 应用飞边退化：使用5x5平均几何滤波器调整高斯点位置
5. 应用畸变退化：使用15x15大核滤波器产生更严重的几何畸变

### 核心模块: `gaussian_degradation.py`
包含所有退化算法的实现：
- `GaussianModel`: 高斯模型数据结构
- `GaussianRenderer`: 简化的深度渲染器
- `OcclusionSimulator`: 遮挡模拟（高斯剔除）
- `GeometricFilterSimulator`: 飞边与畸变模拟（平均几何滤波器）
- `NeoVerseDegradation`: 完整的退化流程封装

### PLY 加载器: `ply_loader.py`
无依赖的 PLY 文件读写工具，支持 3D Gaussian Splatting 的二进制 PLY 格式。

## 技术细节

### 遮挡模拟原理
1. 从多个新视角渲染深度图
2. 对每个高斯点，计算其在新视角的投影位置和深度
3. 比较高斯点深度与渲染深度图的深度
4. 如果高斯点深度远大于渲染深度（超过阈值），则判定为被遮挡并剔除

### 几何滤波原理
1. 渲染新视角的深度图
2. 对深度图应用平均滤波器（5x5或15x15）
3. 根据滤波后的深度调整高斯点的位置
4. 较大的滤波核产生更严重的畸变效果

### 参数说明
- `occlusion_threshold`: 遮挡判定阈值，当前设置为场景大小的50%
- `filter_size`: 几何滤波器大小
  - 飞边效应: 5x5
  - 严重畸变: 15x15
- `num_views`: 新视角数量（用于遮挡判定），当前为4个

## 查看结果

生成的 `.ply` 文件可以使用以下工具查看：
- [3D Gaussian Splatting Viewer](https://github.com/antimatter15/splat)
- [SuperSplat Editor](https://playcanvas.com/supersplat/editor)
- [Polycam](https://poly.cam/)
- 或其他支持 3D Gaussian Splatting 的查看器

## 依赖项

- numpy >= 1.24.4
- scipy >= 1.10.1
- torch >= 2.1.2（可选，代码中已包含但未使用）

## 注意事项

1. 所有 PLY 文件使用二进制小端序格式
2. 尺度参数存储为对数值，读取时需要取指数
3. 不透明度存储为 logit 值，需要应用 sigmoid 函数
4. 颜色使用球谐函数 (SH) DC 分量表示，需要转换为 RGB

## 作者
Claude Code - 高斯点云退化工具
