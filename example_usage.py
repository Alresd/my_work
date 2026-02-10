"""
使用示例：演示如何使用NeoVerse降质模拟
"""

import numpy as np
from gaussian_degradation import (
    GaussianModel, Camera, NeoVerseDegradation,
    create_camera_trajectory, save_gaussian_model
)


def create_dummy_gaussian_model(num_gaussians: int = 1000) -> GaussianModel:
    """创建一个测试用的高斯模型（球形点云）"""
    # 在半径为1的球内随机生成点
    positions = np.random.randn(num_gaussians, 3) * 0.5

    # 随机尺度
    scales = np.random.rand(num_gaussians, 3) * 0.02 + 0.01

    # 随机旋转（四元数）
    rotations = np.random.randn(num_gaussians, 4)
    rotations = rotations / np.linalg.norm(rotations, axis=1, keepdims=True)

    # 随机不透明度
    opacities = np.random.rand(num_gaussians, 1) * 0.5 + 0.5

    # 随机颜色
    colors = np.random.rand(num_gaussians, 3)

    return GaussianModel(
        positions=positions,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors
    )


def example_occlusion_simulation():
    """示例1: 遮挡模拟（图2a）"""
    print("=" * 60)
    print("示例1: 遮挡模拟 - 高斯剔除")
    print("=" * 60)

    # 创建测试高斯模型
    gaussians = create_dummy_gaussian_model(num_gaussians=5000)

    # 创建原始视角相机
    source_camera = Camera(
        position=np.array([3.0, 1.0, 3.0]),
        rotation=np.eye(3),  # 简化，实际需要计算朝向
        focal_length=500.0,
        width=800,
        height=600
    )

    # 创建新视角轨迹（环绕物体）
    novel_cameras = create_camera_trajectory(
        center=np.array([0, 0, 0]),
        radius=2.5,
        num_views=8,
        height=0.5
    )

    # 应用遮挡降质
    degradation = NeoVerseDegradation(occlusion_threshold=0.1)
    degraded_gaussians = degradation.apply_occlusion_degradation(
        gaussians, source_camera, novel_cameras
    )

    # 保存结果
    save_gaussian_model(degraded_gaussians, 'output_occlusion_degraded.npz')

    print(f"\n降质前: {len(gaussians.positions)} 个高斯点")
    print(f"降质后: {len(degraded_gaussians.positions)} 个高斯点")
    print(f"剔除比例: {(1 - len(degraded_gaussians.positions) / len(gaussians.positions)) * 100:.1f}%")


def example_flying_edge_simulation():
    """示例2: 飞边模拟（图2b）"""
    print("\n" + "=" * 60)
    print("示例2: 飞边模拟 - 平均几何滤波")
    print("=" * 60)

    # 创建测试高斯模型
    gaussians = create_dummy_gaussian_model(num_gaussians=3000)

    # 创建相机
    source_camera = Camera(
        position=np.array([3.0, 1.0, 3.0]),
        rotation=np.eye(3),
        focal_length=500.0,
        width=800,
        height=600
    )

    novel_camera = Camera(
        position=np.array([2.0, 1.5, 2.5]),
        rotation=np.eye(3),
        focal_length=500.0,
        width=800,
        height=600
    )

    # 应用飞边降质（小滤波器核）
    degradation = NeoVerseDegradation(filter_size=5)
    degraded_gaussians = degradation.apply_flying_edge_degradation(
        gaussians, source_camera, novel_camera
    )

    # 保存结果
    save_gaussian_model(degraded_gaussians, 'output_flying_edge_degraded.npz')

    # 计算位置变化
    position_changes = np.linalg.norm(
        degraded_gaussians.positions - gaussians.positions,
        axis=1
    )
    print(f"\n平均位置变化: {position_changes.mean():.4f}")
    print(f"最大位置变化: {position_changes.max():.4f}")


def example_distortion_simulation():
    """示例3: 严重畸变模拟（图2c）"""
    print("\n" + "=" * 60)
    print("示例3: 畸变模拟 - 大核平均滤波")
    print("=" * 60)

    # 创建测试高斯模型
    gaussians = create_dummy_gaussian_model(num_gaussians=3000)

    # 创建相机
    source_camera = Camera(
        position=np.array([3.0, 1.0, 3.0]),
        rotation=np.eye(3),
        focal_length=500.0,
        width=800,
        height=600
    )

    novel_camera = Camera(
        position=np.array([2.0, 1.5, 2.5]),
        rotation=np.eye(3),
        focal_length=500.0,
        width=800,
        height=600
    )

    # 应用严重畸变降质（大滤波器核）
    degradation = NeoVerseDegradation(filter_size=5)
    degraded_gaussians = degradation.apply_distortion_degradation(
        gaussians, source_camera, novel_camera, large_filter_size=15
    )

    # 保存结果
    save_gaussian_model(degraded_gaussians, 'output_distortion_degraded.npz')

    # 计算位置变化
    position_changes = np.linalg.norm(
        degraded_gaussians.positions - gaussians.positions,
        axis=1
    )
    print(f"\n平均位置变化: {position_changes.mean():.4f}")
    print(f"最大位置变化: {position_changes.max():.4f}")


def example_combined_degradation():
    """示例4: 组合降质流程"""
    print("\n" + "=" * 60)
    print("示例4: 组合降质 - 先遮挡剔除后几何滤波")
    print("=" * 60)

    # 创建测试高斯模型
    gaussians = create_dummy_gaussian_model(num_gaussians=5000)

    # 创建相机
    source_camera = Camera(
        position=np.array([3.0, 1.0, 3.0]),
        rotation=np.eye(3),
        focal_length=500.0,
        width=800,
        height=600
    )

    # 新视角轨迹
    novel_cameras = create_camera_trajectory(
        center=np.array([0, 0, 0]),
        radius=2.5,
        num_views=6
    )

    degradation = NeoVerseDegradation(
        occlusion_threshold=0.15,
        filter_size=7
    )

    # 步骤1: 遮挡剔除
    gaussians_after_occlusion = degradation.apply_occlusion_degradation(
        gaussians, source_camera, novel_cameras
    )

    # 步骤2: 飞边模拟
    gaussians_final = degradation.apply_flying_edge_degradation(
        gaussians_after_occlusion, source_camera, novel_cameras[0]
    )

    # 保存结果
    save_gaussian_model(gaussians_final, 'output_combined_degraded.npz')

    print(f"\n原始: {len(gaussians.positions)} 个高斯点")
    print(f"遮挡剔除后: {len(gaussians_after_occlusion.positions)} 个高斯点")
    print(f"最终: {len(gaussians_final.positions)} 个高斯点")


if __name__ == '__main__':
    # 运行所有示例
    example_occlusion_simulation()
    example_flying_edge_simulation()
    example_distortion_simulation()
    example_combined_degradation()

    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
