"""
更激进的退化参数，产生明显的空洞和退化效果
"""

import numpy as np
from ply_loader import read_ply_gaussian, write_ply_gaussian
from gaussian_degradation import (
    GaussianModel, Camera, NeoVerseDegradation,
    create_camera_trajectory
)


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


def main():
    """更激进的退化参数"""

    input_file = "3a79b9aefafb0b8d.ply"

    print("="*60)
    print("激进退化模式 - 产生明显的空洞和退化效果")
    print("="*60)

    # 1. 加载模型
    print("\n1. 加载模型...")
    gaussians = load_ply_to_gaussian(input_file)
    print(f"加载完成: {len(gaussians.positions)} 个高斯点")

    # 计算场景参数
    scene_center = gaussians.positions.mean(axis=0)
    scene_size = np.max(gaussians.positions.max(axis=0) - gaussians.positions.min(axis=0))
    camera_radius = scene_size * 1.5

    print(f"场景中心: {scene_center}")
    print(f"场景大小: {scene_size:.2f}")

    # 2. 设置相机
    camera_pos = scene_center + np.array([camera_radius, 0, 0])
    forward = scene_center - camera_pos
    forward = forward / np.linalg.norm(forward)
    up = np.array([0, 1, 0])
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)
    camera_rotation = np.stack([right, up, -forward], axis=0)

    source_camera = Camera(
        position=camera_pos,
        rotation=camera_rotation,
        focal_length=800.0,
        width=1024,
        height=768
    )

    # 创建更多视角（增加遮挡判定的严格性）
    novel_cameras = create_camera_trajectory(
        center=scene_center,
        radius=camera_radius,
        num_views=8,  # 增加视角数
        height=scene_size * 0.2,
        focal_length=800.0,
        image_size=(1024, 768)
    )

    # 3. 激进的遮挡退化（更小的阈值 = 更严格的剔除）
    print("\n" + "="*60)
    print("2. 应用激进遮挡退化")
    print("="*60)

    degradation = NeoVerseDegradation(
        occlusion_threshold=scene_size * 0.1,  # 从0.5降到0.1，更严格
        filter_size=5
    )

    gaussians_aggressive = degradation.apply_occlusion_degradation(
        gaussians, source_camera, novel_cameras
    )

    save_gaussian_to_ply(gaussians_aggressive, "aggressive_occlusion.ply")

    removal_ratio = (1 - len(gaussians_aggressive.positions) / len(gaussians.positions)) * 100
    print(f"\n激进遮挡退化统计:")
    print(f"  原始点数: {len(gaussians.positions)}")
    print(f"  剩余点数: {len(gaussians_aggressive.positions)}")
    print(f"  剔除比例: {removal_ratio:.1f}%")

    # 4. 更大的几何畸变
    print("\n" + "="*60)
    print("3. 应用更严重的几何畸变")
    print("="*60)

    degradation_strong = NeoVerseDegradation(filter_size=25)  # 从15增加到25
    gaussians_strong_distortion = degradation_strong.apply_distortion_degradation(
        gaussians_aggressive, source_camera, novel_cameras[0],
        large_filter_size=25
    )

    save_gaussian_to_ply(gaussians_strong_distortion, "aggressive_distortion.ply")

    position_changes = np.linalg.norm(
        gaussians_strong_distortion.positions - gaussians_aggressive.positions,
        axis=1
    )
    print(f"\n严重畸变统计:")
    print(f"  平均位置变化: {position_changes.mean():.4f}")
    print(f"  最大位置变化: {position_changes.max():.4f}")

    # 5. 区域性剔除（人工制造明显空洞）
    print("\n" + "="*60)
    print("4. 创建区域性空洞")
    print("="*60)

    # 根据位置剔除某个区域的点
    # 剔除X坐标在某个范围的点
    x_min, x_max = gaussians.positions[:, 0].min(), gaussians.positions[:, 0].max()
    hole_start = x_min + (x_max - x_min) * 0.3
    hole_end = x_min + (x_max - x_min) * 0.6

    mask = ~((gaussians.positions[:, 0] > hole_start) & (gaussians.positions[:, 0] < hole_end))

    gaussians_with_hole = GaussianModel(
        positions=gaussians.positions[mask],
        scales=gaussians.scales[mask],
        rotations=gaussians.rotations[mask],
        opacities=gaussians.opacities[mask],
        colors=gaussians.colors[mask]
    )

    save_gaussian_to_ply(gaussians_with_hole, "aggressive_hole.ply")

    print(f"区域性空洞统计:")
    print(f"  剔除区域: X轴 {hole_start:.2f} 到 {hole_end:.2f}")
    print(f"  剩余点数: {len(gaussians_with_hole.positions)}")
    print(f"  剔除比例: {(1 - len(gaussians_with_hole.positions)/len(gaussians.positions))*100:.1f}%")

    # 6. 随机稀疏化（剔除50%的点）
    print("\n" + "="*60)
    print("5. 应用随机稀疏化（50%）")
    print("="*60)

    num_keep = len(gaussians.positions) // 2
    indices = np.random.choice(len(gaussians.positions), num_keep, replace=False)

    gaussians_sparse = GaussianModel(
        positions=gaussians.positions[indices],
        scales=gaussians.scales[indices],
        rotations=gaussians.rotations[indices],
        opacities=gaussians.opacities[indices],
        colors=gaussians.colors[indices]
    )

    save_gaussian_to_ply(gaussians_sparse, "aggressive_sparse_50percent.ply")

    print(f"随机稀疏化统计:")
    print(f"  原始点数: {len(gaussians.positions)}")
    print(f"  剩余点数: {len(gaussians_sparse.positions)}")
    print(f"  剔除比例: 50.0%")

    print("\n" + "="*60)
    print("激进退化完成！")
    print("="*60)
    print("\n生成的文件:")
    print("  1. aggressive_occlusion.ply - 激进遮挡退化（更多剔除）")
    print("  2. aggressive_distortion.ply - 严重几何畸变（25x25滤波）")
    print("  3. aggressive_hole.ply - 区域性空洞（剔除中间30%区域）")
    print("  4. aggressive_sparse_50percent.ply - 随机稀疏化（剔除50%）")


if __name__ == '__main__':
    np.random.seed(42)
    main()
