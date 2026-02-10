"""
自然退化 - 基于距离和视角的选择性剔除（40-50%）
"""

import numpy as np
from ply_loader import read_ply_gaussian, write_ply_gaussian
from gaussian_degradation import GaussianModel, Camera


def load_ply_to_gaussian(filepath):
    """从PLY文件加载高斯模型"""
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


def save_gaussian_to_ply(gaussians, filepath):
    """保存高斯模型为PLY文件"""
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


def distance_based_pruning(gaussians, camera, prune_ratio=0.4):
    """
    基于相机距离的剔除
    剔除离相机较远的点（模拟只有近距离采集完整的情况）
    """
    distances = np.linalg.norm(gaussians.positions - camera.position, axis=1)

    # 计算分位数阈值
    threshold = np.percentile(distances, prune_ratio * 100)

    # 保留距离小于阈值的点
    mask = distances < threshold

    print(f"距离剔除:")
    print(f"  原始点数: {len(gaussians.positions)}")
    print(f"  距离阈值: {threshold:.2f}")
    print(f"  保留点数: {np.sum(mask)}")
    print(f"  剔除比例: {(1 - np.sum(mask)/len(gaussians.positions))*100:.1f}%")

    return GaussianModel(
        positions=gaussians.positions[mask],
        scales=gaussians.scales[mask],
        rotations=gaussians.rotations[mask],
        opacities=gaussians.opacities[mask],
        colors=gaussians.colors[mask]
    )


def directional_pruning(gaussians, camera, direction_axis='x', prune_ratio=0.4):
    """
    定向剔除 - 沿某个方向剔除
    模拟从一个方向采集时背面缺失的情况
    """
    # 计算相对于相机的方向
    relative_pos = gaussians.positions - camera.position

    # 选择轴向
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis = axis_map[direction_axis.lower()]

    # 根据该轴的值进行剔除
    axis_values = relative_pos[:, axis]
    threshold = np.percentile(axis_values, prune_ratio * 100)

    mask = axis_values > threshold

    print(f"定向剔除 ({direction_axis}轴):")
    print(f"  原始点数: {len(gaussians.positions)}")
    print(f"  阈值: {threshold:.2f}")
    print(f"  保留点数: {np.sum(mask)}")
    print(f"  剔除比例: {(1 - np.sum(mask)/len(gaussians.positions))*100:.1f}%")

    return GaussianModel(
        positions=gaussians.positions[mask],
        scales=gaussians.scales[mask],
        rotations=gaussians.rotations[mask],
        opacities=gaussians.opacities[mask],
        colors=gaussians.colors[mask]
    )


def view_dependent_pruning(gaussians, camera, angle_threshold=60, prune_ratio=0.4):
    """
    视角相关剔除
    剔除与相机视线夹角过大的点（模拟只能看到正面的情况）
    """
    # 相机到点的向量
    view_directions = gaussians.positions - camera.position
    view_directions = view_directions / np.linalg.norm(view_directions, axis=1, keepdims=True)

    # 相机朝向
    camera_forward = -camera.rotation[2, :]  # 相机看向-Z方向

    # 计算夹角
    cos_angles = np.dot(view_directions, camera_forward)
    angles = np.arccos(np.clip(cos_angles, -1, 1)) * 180 / np.pi

    # 保留夹角小于阈值的点
    mask = angles < angle_threshold

    # 如果保留的点太多，额外随机剔除一些
    num_keep = np.sum(mask)
    target_keep = int(len(gaussians.positions) * (1 - prune_ratio))

    if num_keep > target_keep:
        keep_indices = np.where(mask)[0]
        selected_indices = np.random.choice(keep_indices, target_keep, replace=False)
        mask = np.zeros(len(gaussians.positions), dtype=bool)
        mask[selected_indices] = True

    print(f"视角相关剔除 (角度阈值={angle_threshold}度):")
    print(f"  原始点数: {len(gaussians.positions)}")
    print(f"  保留点数: {np.sum(mask)}")
    print(f"  剔除比例: {(1 - np.sum(mask)/len(gaussians.positions))*100:.1f}%")

    return GaussianModel(
        positions=gaussians.positions[mask],
        scales=gaussians.scales[mask],
        rotations=gaussians.rotations[mask],
        opacities=gaussians.opacities[mask],
        colors=gaussians.colors[mask]
    )


def combined_pruning(gaussians, camera, distance_ratio=0.2, directional_ratio=0.2, random_ratio=0.1):
    """组合剔除策略"""
    print("\n组合剔除策略:")

    # 1. 距离剔除
    print("\n步骤1: 距离剔除")
    gaussians = distance_based_pruning(gaussians, camera, distance_ratio)

    # 2. 定向剔除
    print("\n步骤2: 定向剔除")
    gaussians = directional_pruning(gaussians, camera, 'x', directional_ratio)

    # 3. 随机稀疏
    print("\n步骤3: 随机稀疏")
    num_keep = int(len(gaussians.positions) * (1 - random_ratio))
    indices = np.random.choice(len(gaussians.positions), num_keep, replace=False)

    gaussians = GaussianModel(
        positions=gaussians.positions[indices],
        scales=gaussians.scales[indices],
        rotations=gaussians.rotations[indices],
        opacities=gaussians.opacities[indices],
        colors=gaussians.colors[indices]
    )

    print(f"  保留点数: {len(gaussians.positions)}")

    return gaussians


def main():
    """生成自然的退化效果"""

    input_file = "3a79b9aefafb0b8d.ply"

    print("="*60)
    print("自然退化 - 基于距离和视角的剔除（目标：40-50%）")
    print("="*60)

    # 加载模型
    print("\n加载模型...")
    gaussians = load_ply_to_gaussian(input_file)
    original_count = len(gaussians.positions)
    print(f"加载完成: {original_count} 个高斯点")

    # 计算场景参数
    scene_center = gaussians.positions.mean(axis=0)
    scene_size = np.max(gaussians.positions.max(axis=0) - gaussians.positions.min(axis=0))
    camera_radius = scene_size * 1.5

    # 设置相机
    camera_pos = scene_center + np.array([camera_radius, 0, 0])
    forward = scene_center - camera_pos
    forward = forward / np.linalg.norm(forward)
    up = np.array([0, 1, 0])
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)
    camera_rotation = np.stack([right, up, -forward], axis=0)

    camera = Camera(
        position=camera_pos,
        rotation=camera_rotation,
        focal_length=800.0,
        width=1024,
        height=768
    )

    # 生成多种退化版本
    print("\n" + "="*60)
    print("生成不同退化版本")
    print("="*60)

    # 版本1: 距离剔除 40%
    print("\n[版本1] 距离剔除 40%")
    print("-" * 40)
    g1 = distance_based_pruning(gaussians.copy(), camera, 0.4)
    save_gaussian_to_ply(g1, "natural_distance_40percent.ply")

    # 版本2: 定向剔除 45%
    print("\n[版本2] 定向剔除 45%")
    print("-" * 40)
    g2 = directional_pruning(gaussians.copy(), camera, 'x', 0.45)
    save_gaussian_to_ply(g2, "natural_directional_45percent.ply")

    # 版本3: 视角剔除 40%
    print("\n[版本3] 视角剔除 40%")
    print("-" * 40)
    g3 = view_dependent_pruning(gaussians.copy(), camera, 75, 0.4)
    save_gaussian_to_ply(g3, "natural_viewangle_40percent.ply")

    # 版本4: 组合剔除 约45-50%
    print("\n[版本4] 组合剔除 约45-50%")
    print("-" * 40)
    g4 = combined_pruning(gaussians.copy(), camera, 0.2, 0.2, 0.1)
    total_pruned = (1 - len(g4.positions) / original_count) * 100
    print(f"\n总剔除比例: {total_pruned:.1f}%")
    save_gaussian_to_ply(g4, "natural_combined_50percent.ply")

    print("\n" + "="*60)
    print("完成！")
    print("="*60)
    print("\n生成的文件:")
    print(f"  1. natural_distance_40percent.ply - 距离剔除 ({len(g1.positions)} 点)")
    print(f"  2. natural_directional_45percent.ply - 定向剔除 ({len(g2.positions)} 点)")
    print(f"  3. natural_viewangle_40percent.ply - 视角剔除 ({len(g3.positions)} 点)")
    print(f"  4. natural_combined_50percent.ply - 组合剔除 ({len(g4.positions)} 点)")

    print("\n重要提示:")
    print("  请使用专业3DGS查看器查看.ply文件以获得真实效果！")
    print("  推荐: https://playcanvas.com/supersplat/editor")
    print("       https://antimatter15.com/splat/")


if __name__ == '__main__':
    np.random.seed(42)
    main()
