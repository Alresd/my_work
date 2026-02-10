"""
渲染激进退化效果对比
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ply_loader import read_ply_gaussian
from gaussian_degradation import GaussianModel, Camera
import os


def load_gaussian_from_ply(filepath):
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


def simple_render(gaussians: GaussianModel, camera: Camera, image_size=(800, 600)):
    """简单的点云投影渲染"""
    width, height = image_size

    view_matrix = camera.get_view_matrix()
    positions_homo = np.concatenate([gaussians.positions, np.ones((len(gaussians.positions), 1))], axis=1)
    positions_cam = (view_matrix @ positions_homo.T).T[:, :3]

    depths = -positions_cam[:, 2]
    mask = depths > 0
    positions_cam = positions_cam[mask]
    depths = depths[mask]
    colors = gaussians.colors[mask]
    opacities = gaussians.opacities[mask]
    scales = gaussians.scales[mask]

    print(f"  可见点数: {len(depths)} / {len(gaussians.positions)}")

    if len(depths) == 0:
        return np.zeros((height, width, 3))

    fx = fy = camera.focal_length
    cx = width / 2
    cy = height / 2

    x_proj = (positions_cam[:, 0] / -positions_cam[:, 2]) * fx + cx
    y_proj = (positions_cam[:, 1] / -positions_cam[:, 2]) * fy + cy

    image = np.zeros((height, width, 3))
    depth_buffer = np.full((height, width), np.inf)

    sort_indices = np.argsort(-depths)

    for idx in sort_indices:
        x, y = int(x_proj[idx]), int(y_proj[idx])
        depth = depths[idx]
        color = colors[idx]
        opacity = opacities[idx, 0]
        scale = np.mean(scales[idx])

        radius = max(1, int(scale * fx / depth * 10))
        radius = min(radius, 20)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                px, py = x + dx, y + dy

                if 0 <= px < width and 0 <= py < height:
                    dist_sq = dx*dx + dy*dy
                    weight = opacity * np.exp(-dist_sq / (2 * radius * radius))

                    if depth < depth_buffer[py, px] + 0.5:
                        alpha = weight
                        image[py, px] = image[py, px] * (1 - alpha) + color * alpha
                        depth_buffer[py, px] = min(depth_buffer[py, px], depth)

    return np.clip(image, 0, 1)


def create_camera_for_scene(scene_center, scene_size, angle_deg=0):
    """为场景创建相机"""
    angle_rad = np.deg2rad(angle_deg)
    radius = scene_size * 1.5

    x = scene_center[0] + radius * np.cos(angle_rad)
    z = scene_center[2] + radius * np.sin(angle_rad)
    y = scene_center[1] + scene_size * 0.3
    camera_pos = np.array([x, y, z])

    forward = scene_center - camera_pos
    forward = forward / np.linalg.norm(forward)
    up = np.array([0, 1, 0])
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)

    camera_rotation = np.stack([right, up, -forward], axis=0)

    return Camera(
        position=camera_pos,
        rotation=camera_rotation,
        focal_length=800.0,
        width=1024,
        height=768
    )


def main():
    """渲染激进退化效果"""

    files = {
        'Original\n(345,600 points)': '3a79b9aefafb0b8d.ply',
        'Aggressive Occlusion\n(99.7% pruned)': 'aggressive_occlusion.ply',
        'Regional Hole\n(68.2% pruned)': 'aggressive_hole.ply',
        'Random Sparse\n(50% pruned)': 'aggressive_sparse_50percent.ply',
        'Strong Distortion\n(25x25 filter)': 'aggressive_distortion.ply'
    }

    # 检查文件
    for name, filepath in files.items():
        if not os.path.exists(filepath):
            print(f"错误: 找不到文件 {filepath}")
            return

    # 加载原始模型
    print("加载原始模型...")
    original = load_gaussian_from_ply(files['Original\n(345,600 points)'])
    scene_center = original.positions.mean(axis=0)
    scene_size = np.max(original.positions.max(axis=0) - original.positions.min(axis=0))

    print(f"场景中心: {scene_center}")
    print(f"场景大小: {scene_size:.2f}")

    # 创建3个视角
    angles = [0, 45, 90]
    angle_names = ['Front View', '45° View', 'Side View']

    print(f"\n开始渲染...")

    # 创建图像网格
    fig = plt.figure(figsize=(18, 20))
    gs = GridSpec(len(files), len(angles), figure=fig, hspace=0.25, wspace=0.05)

    for row, (model_name, filepath) in enumerate(files.items()):
        print(f"\n渲染 {model_name.replace(chr(10), ' ')}: {filepath}")
        gaussians = load_gaussian_from_ply(filepath)

        for col, (angle, angle_name) in enumerate(zip(angles, angle_names)):
            print(f"  {angle_name}...")

            camera = create_camera_for_scene(scene_center, scene_size, angle)
            image = simple_render(gaussians, camera, image_size=(800, 600))

            ax = fig.add_subplot(gs[row, col])
            ax.imshow(image)
            ax.axis('off')

            # 标题
            if row == 0:
                ax.set_title(angle_name, fontsize=13, fontweight='bold', pad=10)
            if col == 0:
                ax.text(-0.15, 0.5, model_name,
                       transform=ax.transAxes,
                       fontsize=11, fontweight='bold',
                       verticalalignment='center',
                       horizontalalignment='right')

    plt.suptitle('Aggressive Gaussian Splatting Degradation',
                fontsize=16, fontweight='bold', y=0.995)

    output_file = 'aggressive_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n激进退化对比图已保存到: {output_file}")
    plt.close()

    # 生成并排对比（45度视角）
    print("\n生成45度视角并排对比...")
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    camera = create_camera_for_scene(scene_center, scene_size, angle_deg=45)

    for ax, (name, filepath) in zip(axes, files.items()):
        print(f"  {name.replace(chr(10), ' ')}")
        gaussians = load_gaussian_from_ply(filepath)
        image = simple_render(gaussians, camera, image_size=(800, 600))

        ax.imshow(image)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle('Aggressive Degradation Effects (45° View)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = 'aggressive_comparison_single.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"45度对比图已保存到: {output_file}")
    plt.close()

    print("\n完成！")


if __name__ == '__main__':
    main()
