"""
优化的退化参数 - 产生自然的遮挡效果（40-50%剔除）
参考NeoVerse论文的退化效果
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
    """优化的退化流程 - 自然遮挡效果"""

    input_file = "3a79b9aefafb0b8d.ply"

    print("="*60)
    print("优化退化模式 - 自然遮挡效果（目标：40-50%剔除）")
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

    # 创建新视角轨迹（更多视角以产生更自然的遮挡）
    novel_cameras = create_camera_trajectory(
        center=scene_center,
        radius=camera_radius * 1.2,  # 稍远一些
        num_views=12,  # 更多视角
        height=scene_size * 0.15,
        focal_length=800.0,
        image_size=(1024, 768)
    )

    # 3. 尝试不同的遮挡阈值，找到合适的剔除率
    print("\n" + "="*60)
    print("2. 测试不同阈值的遮挡效果")
    print("="*60)

    thresholds = [
        (scene_size * 0.2, "optimal_occlusion_40percent"),
        (scene_size * 0.3, "optimal_occlusion_35percent"),
        (scene_size * 0.25, "optimal_occlusion_38percent"),
    ]

    results = []

    for threshold, name in thresholds:
        print(f"\n测试阈值: {threshold:.2f}")

        degradation = NeoVerseDegradation(
            occlusion_threshold=threshold,
            filter_size=5
        )

        gaussians_degraded = degradation.apply_occlusion_degradation(
            gaussians, source_camera, novel_cameras
        )

        removal_ratio = (1 - len(gaussians_degraded.positions) / len(gaussians.positions)) * 100

        results.append({
            'name': name,
            'threshold': threshold,
            'removal_ratio': removal_ratio,
            'gaussians': gaussians_degraded
        })

        print(f"结果: 剔除 {removal_ratio:.1f}%")

    # 4. 保存所有结果
    print("\n" + "="*60)
    print("3. 保存结果")
    print("="*60)

    for result in results:
        filename = f"{result['name']}.ply"
        save_gaussian_to_ply(result['gaussians'], filename)
        print(f"✓ {filename} - 剔除 {result['removal_ratio']:.1f}% ({len(result['gaussians'].positions)} 点)")

    # 5. 在最优结果上应用飞边效果
    print("\n" + "="*60)
    print("4. 应用飞边和畸变效果")
    print("="*60)

    # 选择剔除率最接近40-50%的结果
    best_result = min(results, key=lambda x: abs(x['removal_ratio'] - 45))
    print(f"\n选择最优结果: 剔除 {best_result['removal_ratio']:.1f}%")

    # 飞边效果
    degradation_fe = NeoVerseDegradation(filter_size=7)
    gaussians_flying_edge = degradation_fe.apply_flying_edge_degradation(
        best_result['gaussians'], source_camera, novel_cameras[0]
    )
    save_gaussian_to_ply(gaussians_flying_edge, "optimal_with_flying_edge.ply")

    # 畸变效果
    gaussians_distortion = degradation_fe.apply_distortion_degradation(
        best_result['gaussians'], source_camera, novel_cameras[0],
        large_filter_size=20
    )
    save_gaussian_to_ply(gaussians_distortion, "optimal_with_distortion.ply")

    print("\n" + "="*60)
    print("优化退化完成！")
    print("="*60)
    print("\n生成的文件（请用3DGS查看器查看）：")
    for result in results:
        print(f"  • {result['name']}.ply - 剔除 {result['removal_ratio']:.1f}%")
    print(f"  • optimal_with_flying_edge.ply - 最优 + 飞边效果")
    print(f"  • optimal_with_distortion.ply - 最优 + 畸变效果")

    print("\n推荐查看器：")
    print("  • SuperSplat: https://playcanvas.com/supersplat/editor")
    print("  • Antimatter15: https://antimatter15.com/splat/")
    print("\n⚠️ 注意：我们的简单渲染器无法展现真实效果，")
    print("   请务必用专业3DGS查看器查看.ply文件！")


if __name__ == '__main__':
    np.random.seed(42)
    main()
