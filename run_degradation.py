"""
完整的高斯点云退化脚本
读取 .ply 文件，应用退化操作，保存结果
"""

import numpy as np
from ply_loader import read_ply_gaussian, write_ply_gaussian
from gaussian_degradation import (
    GaussianModel, Camera, NeoVerseDegradation,
    create_camera_trajectory
)
import os


def load_ply_gaussian_model(filepath: str) -> GaussianModel:
    """从 .ply 文件加载 3D Gaussian Splatting 模型

    Args:
        filepath: .ply 文件路径

    Returns:
        GaussianModel: 高斯模型
    """
    print(f"\n正在加载 PLY 文件: {filepath}")

    # 使用自定义加载器
    data = read_ply_gaussian(filepath)

    # 提取位置
    positions = np.stack([
        data['x'],
        data['y'],
        data['z']
    ], axis=1)

    # 提取尺度（需要从对数空间转换）
    scales = np.stack([
        data['scale_0'],
        data['scale_1'],
        data['scale_2']
    ], axis=1)
    scales = np.exp(scales)  # 尺度存储为对数，需要取指数

    # 提取旋转（四元数）
    rotations = np.stack([
        data['rot_0'],
        data['rot_1'],
        data['rot_2'],
        data['rot_3']
    ], axis=1)

    # 提取不透明度（需要应用sigmoid）
    opacities = data['opacity'].reshape(-1, 1)
    # 3DGS 中 opacity 通常存储为 logit，需要 sigmoid
    opacities = 1 / (1 + np.exp(-opacities))

    # 提取颜色（SH DC分量）
    # f_dc_0, f_dc_1, f_dc_2 对应 RGB
    # SH DC系数需要转换为RGB: RGB = 0.5 + SH_DC * C0
    C0 = 0.28209479177387814  # SH第0阶系数
    colors = np.stack([
        data['f_dc_0'],
        data['f_dc_1'],
        data['f_dc_2']
    ], axis=1)
    colors = 0.5 + colors * C0
    colors = np.clip(colors, 0, 1)

    print(f"\n模型统计:")
    print(f"  高斯点数量: {len(positions)}")
    print(f"  位置范围: [{positions.min():.2f}, {positions.max():.2f}]")
    print(f"  尺度范围: [{scales.min():.6f}, {scales.max():.6f}]")
    print(f"  不透明度范围: [{opacities.min():.3f}, {opacities.max():.3f}]")

    return GaussianModel(
        positions=positions,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors
    )


def save_ply_gaussian_model(gaussians: GaussianModel, filepath: str):
    """保存高斯模型为 .ply 文件

    Args:
        gaussians: 高斯模型
        filepath: 输出文件路径
    """
    print(f"\n正在保存到 PLY 文件: {filepath}")

    # 准备数据
    num_points = len(gaussians.positions)

    # 转换不透明度回 logit
    opacities_logit = -np.log(1.0 / np.clip(gaussians.opacities, 0.001, 0.999) - 1.0)

    # 转换颜色回 SH DC 系数
    C0 = 0.28209479177387814
    colors_sh = (gaussians.colors - 0.5) / C0

    # 转换尺度回对数空间
    scales_log = np.log(np.clip(gaussians.scales, 1e-8, None))

    # 构建数据字典
    data = {
        'x': gaussians.positions[:, 0],
        'y': gaussians.positions[:, 1],
        'z': gaussians.positions[:, 2],
        'nx': np.zeros(num_points),
        'ny': np.zeros(num_points),
        'nz': np.zeros(num_points),
        'f_dc_0': colors_sh[:, 0],
        'f_dc_1': colors_sh[:, 1],
        'f_dc_2': colors_sh[:, 2],
        'opacity': opacities_logit[:, 0],
        'scale_0': scales_log[:, 0],
        'scale_1': scales_log[:, 1],
        'scale_2': scales_log[:, 2],
        'rot_0': gaussians.rotations[:, 0],
        'rot_1': gaussians.rotations[:, 1],
        'rot_2': gaussians.rotations[:, 2],
        'rot_3': gaussians.rotations[:, 3]
    }

    # 使用自定义写入器
    write_ply_gaussian(filepath, data)

    print(f"保存完成: {num_points} 个高斯点")


def main():
    """主函数：完整的退化流程"""

    # 输入文件
    input_ply = "3a79b9aefafb0b8d.ply"

    if not os.path.exists(input_ply):
        print(f"错误: 找不到输入文件 {input_ply}")
        return

    # 1. 加载模型
    print("\n" + "="*60)
    print("步骤 1: 加载高斯模型")
    print("="*60)
    gaussians = load_ply_gaussian_model(input_ply)

    # 2. 设置相机参数
    print("\n" + "="*60)
    print("步骤 2: 设置相机参数")
    print("="*60)

    # 计算场景中心和范围
    scene_center = gaussians.positions.mean(axis=0)
    scene_size = np.max(gaussians.positions.max(axis=0) - gaussians.positions.min(axis=0))
    camera_radius = scene_size * 1.5

    print(f"场景中心: {scene_center}")
    print(f"场景大小: {scene_size:.2f}")
    print(f"相机半径: {camera_radius:.2f}")

    # 创建原始视角相机
    # 相机位置在场景右侧
    camera_pos = scene_center + np.array([camera_radius, 0, 0])

    # 计算相机朝向场景中心的旋转矩阵
    forward = scene_center - camera_pos
    forward = forward / np.linalg.norm(forward)
    up = np.array([0, 1, 0])
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)

    # 旋转矩阵：从世界坐标系到相机坐标系
    camera_rotation = np.stack([right, up, -forward], axis=0)

    source_camera = Camera(
        position=camera_pos,
        rotation=camera_rotation,
        focal_length=800.0,
        width=1024,
        height=768
    )

    # 创建新视角轨迹（环绕场景）
    novel_cameras = create_camera_trajectory(
        center=scene_center,
        radius=camera_radius,
        num_views=4,  # 减少视角数量以加快处理
        height=scene_size * 0.2,
        focal_length=800.0,
        image_size=(1024, 768)
    )

    print(f"创建了 {len(novel_cameras)} 个新视角相机")

    # 3. 应用遮挡退化
    print("\n" + "="*60)
    print("步骤 3: 应用遮挡退化（高斯剔除）")
    print("="*60)

    degradation = NeoVerseDegradation(
        occlusion_threshold=scene_size * 0.5,  # 根据场景大小调整阈值（增大以减少剔除）
        filter_size=5
    )

    gaussians_occluded = degradation.apply_occlusion_degradation(
        gaussians, source_camera, novel_cameras
    )

    # 保存遮挡退化结果
    output_file_1 = "output_occlusion_degraded.ply"
    save_ply_gaussian_model(gaussians_occluded, output_file_1)

    removal_ratio = (1 - len(gaussians_occluded.positions) / len(gaussians.positions)) * 100
    print(f"\n遮挡退化统计:")
    print(f"  原始点数: {len(gaussians.positions)}")
    print(f"  剩余点数: {len(gaussians_occluded.positions)}")
    print(f"  剔除比例: {removal_ratio:.1f}%")

    # 4. 应用飞边退化
    print("\n" + "="*60)
    print("步骤 4: 应用飞边退化（几何滤波）")
    print("="*60)

    gaussians_flying_edge = degradation.apply_flying_edge_degradation(
        gaussians_occluded, source_camera, novel_cameras[0]
    )

    # 保存飞边退化结果
    output_file_2 = "output_flying_edge_degraded.ply"
    save_ply_gaussian_model(gaussians_flying_edge, output_file_2)

    position_changes = np.linalg.norm(
        gaussians_flying_edge.positions - gaussians_occluded.positions,
        axis=1
    )
    print(f"\n飞边退化统计:")
    print(f"  平均位置变化: {position_changes.mean():.4f}")
    print(f"  最大位置变化: {position_changes.max():.4f}")
    print(f"  中位数位置变化: {np.median(position_changes):.4f}")

    # 5. 应用严重畸变退化
    print("\n" + "="*60)
    print("步骤 5: 应用严重畸变退化（大核滤波）")
    print("="*60)

    gaussians_distorted = degradation.apply_distortion_degradation(
        gaussians_occluded, source_camera, novel_cameras[0],
        large_filter_size=15
    )

    # 保存畸变退化结果
    output_file_3 = "output_distortion_degraded.ply"
    save_ply_gaussian_model(gaussians_distorted, output_file_3)

    position_changes_2 = np.linalg.norm(
        gaussians_distorted.positions - gaussians_occluded.positions,
        axis=1
    )
    print(f"\n畸变退化统计:")
    print(f"  平均位置变化: {position_changes_2.mean():.4f}")
    print(f"  最大位置变化: {position_changes_2.max():.4f}")
    print(f"  中位数位置变化: {np.median(position_changes_2):.4f}")

    # 完成
    print("\n" + "="*60)
    print("所有退化操作完成！")
    print("="*60)
    print(f"\n输出文件:")
    print(f"  1. 遮挡退化: {output_file_1}")
    print(f"  2. 飞边退化: {output_file_2}")
    print(f"  3. 畸变退化: {output_file_3}")
    print("\n这些文件可以在高斯点云查看器中加载查看。")


if __name__ == '__main__':
    main()
