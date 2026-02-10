"""
简化版高斯点云退化脚本
使用基于统计的方法，无需复杂的渲染过程
"""

import numpy as np
from ply_loader import read_ply_gaussian, write_ply_gaussian
from gaussian_degradation import GaussianModel
import os


def apply_random_pruning(gaussians: GaussianModel, prune_ratio: float = 0.3) -> GaussianModel:
    """
    随机剔除高斯点（模拟遮挡效果）

    Args:
        gaussians: 输入高斯模型
        prune_ratio: 剔除比例（0-1之间）

    Returns:
        剔除后的高斯模型
    """
    num_points = len(gaussians.positions)
    num_keep = int(num_points * (1 - prune_ratio))

    # 随机选择保留的点
    indices = np.random.choice(num_points, num_keep, replace=False)
    indices.sort()

    print(f"随机剔除退化:")
    print(f"  原始点数: {num_points}")
    print(f"  保留点数: {num_keep}")
    print(f"  剔除比例: {prune_ratio*100:.1f}%")

    return GaussianModel(
        positions=gaussians.positions[indices],
        scales=gaussians.scales[indices],
        rotations=gaussians.rotations[indices],
        opacities=gaussians.opacities[indices],
        colors=gaussians.colors[indices]
    )


def apply_opacity_pruning(gaussians: GaussianModel, opacity_threshold: float = 0.3) -> GaussianModel:
    """
    基于不透明度剔除高斯点

    Args:
        gaussians: 输入高斯模型
        opacity_threshold: 不透明度阈值，低于此值的点会被剔除

    Returns:
        剔除后的高斯模型
    """
    mask = gaussians.opacities[:, 0] >= opacity_threshold
    num_keep = np.sum(mask)

    print(f"不透明度剔除:")
    print(f"  原始点数: {len(gaussians.positions)}")
    print(f"  保留点数: {num_keep}")
    print(f"  剔除比例: {(1 - num_keep/len(gaussians.positions))*100:.1f}%")

    return GaussianModel(
        positions=gaussians.positions[mask],
        scales=gaussians.scales[mask],
        rotations=gaussians.rotations[mask],
        opacities=gaussians.opacities[mask],
        colors=gaussians.colors[mask]
    )


def apply_position_noise(gaussians: GaussianModel, noise_scale: float = 0.05) -> GaussianModel:
    """
    对高斯点位置添加噪声（模拟几何误差和飞边）

    Args:
        gaussians: 输入高斯模型
        noise_scale: 噪声尺度（相对于场景大小）

    Returns:
        添加噪声后的高斯模型
    """
    # 计算场景尺度
    scene_size = np.max(gaussians.positions.max(axis=0) - gaussians.positions.min(axis=0))
    noise_std = scene_size * noise_scale

    # 生成高斯噪声
    noise = np.random.randn(*gaussians.positions.shape) * noise_std

    # 应用噪声
    noisy_positions = gaussians.positions + noise

    position_changes = np.linalg.norm(noise, axis=1)
    print(f"位置噪声退化:")
    print(f"  场景尺度: {scene_size:.2f}")
    print(f"  噪声标准差: {noise_std:.4f}")
    print(f"  平均位置变化: {position_changes.mean():.4f}")
    print(f"  最大位置变化: {position_changes.max():.4f}")

    return GaussianModel(
        positions=noisy_positions,
        scales=gaussians.scales.copy(),
        rotations=gaussians.rotations.copy(),
        opacities=gaussians.opacities.copy(),
        colors=gaussians.colors.copy()
    )


def apply_scale_inflation(gaussians: GaussianModel, scale_factor: float = 1.5) -> GaussianModel:
    """
    增大高斯点的尺度（模拟模糊和畸变）

    Args:
        gaussians: 输入高斯模型
        scale_factor: 尺度缩放因子（>1 增大，<1 减小）

    Returns:
        缩放后的高斯模型
    """
    inflated_scales = gaussians.scales * scale_factor

    print(f"尺度膨胀退化:")
    print(f"  缩放因子: {scale_factor}")
    print(f"  原始平均尺度: {gaussians.scales.mean():.6f}")
    print(f"  膨胀后平均尺度: {inflated_scales.mean():.6f}")

    return GaussianModel(
        positions=gaussians.positions.copy(),
        scales=inflated_scales,
        rotations=gaussians.rotations.copy(),
        opacities=gaussians.opacities.copy(),
        colors=gaussians.colors.copy()
    )


def apply_opacity_reduction(gaussians: GaussianModel, opacity_factor: float = 0.7) -> GaussianModel:
    """
    降低高斯点的不透明度

    Args:
        gaussians: 输入高斯模型
        opacity_factor: 不透明度缩放因子（0-1之间）

    Returns:
        降低不透明度后的高斯模型
    """
    reduced_opacities = gaussians.opacities * opacity_factor

    print(f"不透明度降低:")
    print(f"  缩放因子: {opacity_factor}")
    print(f"  原始平均不透明度: {gaussians.opacities.mean():.3f}")
    print(f"  降低后平均不透明度: {reduced_opacities.mean():.3f}")

    return GaussianModel(
        positions=gaussians.positions.copy(),
        scales=gaussians.scales.copy(),
        rotations=gaussians.rotations.copy(),
        opacities=reduced_opacities,
        colors=gaussians.colors.copy()
    )


def load_ply_to_gaussian(filepath: str) -> GaussianModel:
    """从 PLY 文件加载高斯模型"""
    data = read_ply_gaussian(filepath)

    positions = np.stack([data['x'], data['y'], data['z']], axis=1)
    scales = np.stack([data['scale_0'], data['scale_1'], data['scale_2']], axis=1)
    scales = np.exp(scales)
    rotations = np.stack([data['rot_0'], data['rot_1'], data['rot_2'], data['rot_3']], axis=1)
    opacities = data['opacity'].reshape(-1, 1)
    opacities = 1 / (1 + np.exp(-opacities))

    C0 = 0.28209479177387814
    colors = np.stack([data['f_dc_0'], data['f_dc_1'], data['f_dc_2']], axis=1)
    colors = 0.5 + colors * C0
    colors = np.clip(colors, 0, 1)

    return GaussianModel(positions=positions, scales=scales, rotations=rotations,
                        opacities=opacities, colors=colors)


def save_gaussian_to_ply(gaussians: GaussianModel, filepath: str):
    """保存高斯模型为 PLY 文件"""
    num_points = len(gaussians.positions)
    opacities_logit = -np.log(1.0 / np.clip(gaussians.opacities, 0.001, 0.999) - 1.0)
    C0 = 0.28209479177387814
    colors_sh = (gaussians.colors - 0.5) / C0
    scales_log = np.log(np.clip(gaussians.scales, 1e-8, None))

    data = {
        'x': gaussians.positions[:, 0], 'y': gaussians.positions[:, 1], 'z': gaussians.positions[:, 2],
        'nx': np.zeros(num_points), 'ny': np.zeros(num_points), 'nz': np.zeros(num_points),
        'f_dc_0': colors_sh[:, 0], 'f_dc_1': colors_sh[:, 1], 'f_dc_2': colors_sh[:, 2],
        'opacity': opacities_logit[:, 0],
        'scale_0': scales_log[:, 0], 'scale_1': scales_log[:, 1], 'scale_2': scales_log[:, 2],
        'rot_0': gaussians.rotations[:, 0], 'rot_1': gaussians.rotations[:, 1],
        'rot_2': gaussians.rotations[:, 2], 'rot_3': gaussians.rotations[:, 3]
    }

    write_ply_gaussian(filepath, data)


def main():
    """主函数"""
    input_file = "3a79b9aefafb0b8d.ply"

    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        return

    print("="*60)
    print("简化版高斯点云退化脚本")
    print("="*60)

    # 加载模型
    print("\n1. 加载模型...")
    gaussians = load_ply_to_gaussian(input_file)
    print(f"加载完成: {len(gaussians.positions)} 个高斯点")

    # 退化类型1: 随机剔除（模拟遮挡）
    print("\n" + "="*60)
    print("2. 应用随机剔除退化（模拟遮挡效果）")
    print("="*60)
    gaussians_pruned = apply_random_pruning(gaussians, prune_ratio=0.3)
    save_gaussian_to_ply(gaussians_pruned, "degraded_pruned_30percent.ply")

    # 退化类型2: 位置噪声（模拟飞边）
    print("\n" + "="*60)
    print("3. 应用位置噪声退化（模拟飞边效果）")
    print("="*60)
    gaussians_noisy = apply_position_noise(gaussians, noise_scale=0.05)
    save_gaussian_to_ply(gaussians_noisy, "degraded_position_noise.ply")

    # 退化类型3: 尺度膨胀（模拟畸变）
    print("\n" + "="*60)
    print("4. 应用尺度膨胀退化（模拟畸变效果）")
    print("="*60)
    gaussians_inflated = apply_scale_inflation(gaussians, scale_factor=2.0)
    save_gaussian_to_ply(gaussians_inflated, "degraded_scale_inflated.ply")

    # 退化类型4: 组合退化（剔除 + 噪声 + 膨胀）
    print("\n" + "="*60)
    print("5. 应用组合退化（剔除 + 噪声 + 膨胀）")
    print("="*60)
    gaussians_combined = apply_random_pruning(gaussians, prune_ratio=0.2)
    gaussians_combined = apply_position_noise(gaussians_combined, noise_scale=0.03)
    gaussians_combined = apply_scale_inflation(gaussians_combined, scale_factor=1.5)
    save_gaussian_to_ply(gaussians_combined, "degraded_combined.ply")

    # 退化类型5: 不透明度剔除
    print("\n" + "="*60)
    print("6. 应用不透明度剔除")
    print("="*60)
    gaussians_opacity_pruned = apply_opacity_pruning(gaussians, opacity_threshold=0.5)
    save_gaussian_to_ply(gaussians_opacity_pruned, "degraded_opacity_pruned.ply")

    # 完成
    print("\n" + "="*60)
    print("所有退化操作完成！")
    print("="*60)
    print("\n生成的文件:")
    print("  1. degraded_pruned_30percent.ply - 随机剔除30%的点")
    print("  2. degraded_position_noise.ply - 位置添加噪声")
    print("  3. degraded_scale_inflated.ply - 尺度膨胀2倍")
    print("  4. degraded_combined.ply - 组合退化（剔除20% + 噪声 + 膨胀1.5倍）")
    print("  5. degraded_opacity_pruned.ply - 不透明度剔除")
    print("\n这些文件可以在高斯点云查看器中加载查看。")


if __name__ == '__main__':
    # 设置随机种子以便复现
    np.random.seed(42)
    main()
