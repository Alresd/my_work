"""
渲染高斯点云模型并生成对比图
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
    """简单的点云投影渲染

    Args:
        gaussians: 高斯模型
        camera: 相机参数
        image_size: 图像尺寸 (width, height)

    Returns:
        image: 渲染的RGB图像
    """
    width, height = image_size

    # 转换到相机坐标系
    view_matrix = camera.get_view_matrix()
    positions_homo = np.concatenate([gaussians.positions, np.ones((len(gaussians.positions), 1))], axis=1)
    positions_cam = (view_matrix @ positions_homo.T).T[:, :3]

    # 深度值
    depths = -positions_cam[:, 2]

    # 过滤掉在相机后面的点
    mask = depths > 0
    positions_cam = positions_cam[mask]
    depths = depths[mask]
    colors = gaussians.colors[mask]
    opacities = gaussians.opacities[mask]
    scales = gaussians.scales[mask]

    print(f"  可见点数: {len(depths)} / {len(gaussians.positions)}")

    if len(depths) == 0:
        print("  警告: 没有可见的点！")
        return np.zeros((height, width, 3))

    # 投影到图像平面
    fx = fy = camera.focal_length
    cx = width / 2
    cy = height / 2

    x_proj = (positions_cam[:, 0] / -positions_cam[:, 2]) * fx + cx
    y_proj = (positions_cam[:, 1] / -positions_cam[:, 2]) * fy + cy

    # 创建图像
    image = np.zeros((height, width, 3))
    depth_buffer = np.full((height, width), np.inf)

    # 按深度排序（从远到近）
    sort_indices = np.argsort(-depths)

    # 渲染每个高斯点
    for idx in sort_indices:
        x, y = int(x_proj[idx]), int(y_proj[idx])
        depth = depths[idx]
        color = colors[idx]
        opacity = opacities[idx, 0]
        scale = np.mean(scales[idx])

        # 计算splat半径
        radius = max(1, int(scale * fx / depth * 10))
        radius = min(radius, 20)  # 限制最大半径

        # 在半径范围内splat
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                px, py = x + dx, y + dy

                if 0 <= px < width and 0 <= py < height:
                    # 高斯权重
                    dist_sq = dx*dx + dy*dy
                    weight = opacity * np.exp(-dist_sq / (2 * radius * radius))

                    # 深度测试和混合
                    if depth < depth_buffer[py, px] + 0.5:
                        # Alpha混合
                        alpha = weight
                        image[py, px] = image[py, px] * (1 - alpha) + color * alpha
                        depth_buffer[py, px] = min(depth_buffer[py, px], depth)

    return np.clip(image, 0, 1)


def create_camera_for_scene(scene_center, scene_size, angle_deg=0):
    """为场景创建相机

    Args:
        scene_center: 场景中心
        scene_size: 场景大小
        angle_deg: 相机角度（度）

    Returns:
        Camera对象
    """
    angle_rad = np.deg2rad(angle_deg)
    radius = scene_size * 1.5

    # 相机位置
    x = scene_center[0] + radius * np.cos(angle_rad)
    z = scene_center[2] + radius * np.sin(angle_rad)
    y = scene_center[1] + scene_size * 0.3
    camera_pos = np.array([x, y, z])

    # 计算相机朝向
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


def render_comparison():
    """渲染对比图"""

    # 文件列表
    files = {
        'Original': '3a79b9aefafb0b8d.ply',
        'Occlusion': 'output_occlusion_degraded.ply',
        'Flying Edge': 'output_flying_edge_degraded.ply',
        'Distortion': 'output_distortion_degraded.ply'
    }

    # 检查文件
    for name, filepath in files.items():
        if not os.path.exists(filepath):
            print(f"错误: 找不到文件 {filepath}")
            return

    # 加载原始模型以确定场景参数
    print("加载原始模型...")
    original = load_gaussian_from_ply(files['Original'])
    scene_center = original.positions.mean(axis=0)
    scene_size = np.max(original.positions.max(axis=0) - original.positions.min(axis=0))

    print(f"场景中心: {scene_center}")
    print(f"场景大小: {scene_size:.2f}")

    # 创建3个不同角度的相机
    angles = [0, 45, 90]
    angle_names = ['Front', '45°', 'Side']

    print(f"\n开始渲染，共 {len(files)} 个模型 x {len(angles)} 个视角...")

    # 创建图像网格
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(len(files), len(angles), figure=fig, hspace=0.3, wspace=0.1)

    for row, (model_name, filepath) in enumerate(files.items()):
        print(f"\n渲染 {model_name} ({filepath})...")
        gaussians = load_gaussian_from_ply(filepath)

        for col, (angle, angle_name) in enumerate(zip(angles, angle_names)):
            print(f"  视角: {angle_name} ({angle}°)")

            # 创建相机
            camera = create_camera_for_scene(scene_center, scene_size, angle)

            # 渲染
            image = simple_render(gaussians, camera, image_size=(800, 600))

            # 显示
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(image)
            ax.axis('off')

            # 添加标题
            if row == 0:
                ax.set_title(f'{angle_name}', fontsize=14, fontweight='bold')
            if col == 0:
                ax.text(-0.1, 0.5, model_name,
                       transform=ax.transAxes,
                       fontsize=14, fontweight='bold',
                       verticalalignment='center',
                       horizontalalignment='right',
                       rotation=90)

    plt.suptitle('Gaussian Splatting Degradation Comparison',
                fontsize=16, fontweight='bold', y=0.98)

    # 保存
    output_file = 'comparison_rendering.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n对比图已保存到: {output_file}")

    plt.close()


def render_single_view_comparison():
    """渲染单视角的并排对比"""

    files = {
        'Original': '3a79b9aefafb0b8d.ply',
        'Occlusion\n(28.8% pruned)': 'output_occlusion_degraded.ply',
        'Flying Edge\n(5x5 filter)': 'output_flying_edge_degraded.ply',
        'Distortion\n(15x15 filter)': 'output_distortion_degraded.ply'
    }

    # 检查文件
    for name, filepath in files.items():
        if not os.path.exists(filepath):
            print(f"错误: 找不到文件 {filepath}")
            return

    # 加载原始模型
    print("加载原始模型...")
    original = load_gaussian_from_ply(files['Original'])
    scene_center = original.positions.mean(axis=0)
    scene_size = np.max(original.positions.max(axis=0) - original.positions.min(axis=0))

    # 创建相机（45度视角）
    camera = create_camera_for_scene(scene_center, scene_size, angle_deg=45)

    # 渲染所有模型
    print("\n渲染所有模型...")
    images = {}
    for name, filepath in files.items():
        print(f"\n{name}: {filepath}")
        gaussians = load_gaussian_from_ply(filepath)
        images[name] = simple_render(gaussians, camera, image_size=(800, 600))

    # 创建并排对比图
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, (name, image) in zip(axes, images.items()):
        ax.imshow(image)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.suptitle('Gaussian Splatting Degradation Effects (45° View)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存
    output_file = 'comparison_single_view.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n单视角对比图已保存到: {output_file}")

    plt.close()


if __name__ == '__main__':
    print("="*60)
    print("高斯点云退化效果可视化")
    print("="*60)

    # 生成多视角对比图
    print("\n1. 生成多视角对比图...")
    render_comparison()

    # 生成单视角对比图
    print("\n2. 生成单视角对比图...")
    render_single_view_comparison()

    print("\n完成！请查看生成的PNG图片。")
