"""
NeoVerse风格的高斯模型降质模拟
实现遮挡模拟和飞边/畸变模拟
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from scipy.ndimage import uniform_filter


@dataclass
class GaussianModel:
    """3D Gaussian Splatting模型数据结构"""
    positions: np.ndarray  # [N, 3] 高斯中心位置
    scales: np.ndarray  # [N, 3] 高斯尺度
    rotations: np.ndarray  # [N, 4] 四元数旋转
    opacities: np.ndarray  # [N, 1] 不透明度
    colors: np.ndarray  # [N, 3] RGB颜色

    def copy(self):
        """深拷贝模型"""
        return GaussianModel(
            positions=self.positions.copy(),
            scales=self.scales.copy(),
            rotations=self.rotations.copy(),
            opacities=self.opacities.copy(),
            colors=self.colors.copy()
        )

    def to_torch(self, device='cuda'):
        """转换为PyTorch张量"""
        return GaussianModel(
            positions=torch.from_numpy(self.positions).float().to(device),
            scales=torch.from_numpy(self.scales).float().to(device),
            rotations=torch.from_numpy(self.rotations).float().to(device),
            opacities=torch.from_numpy(self.opacities).float().to(device),
            colors=torch.from_numpy(self.colors).float().to(device)
        )

    def to_numpy(self):
        """转换为NumPy数组"""
        if isinstance(self.positions, torch.Tensor):
            return GaussianModel(
                positions=self.positions.cpu().numpy(),
                scales=self.scales.cpu().numpy(),
                rotations=self.rotations.cpu().numpy(),
                opacities=self.opacities.cpu().numpy(),
                colors=self.colors.cpu().numpy()
            )
        return self


@dataclass
class Camera:
    """相机参数"""
    position: np.ndarray  # [3] 相机位置
    rotation: np.ndarray  # [3, 3] 旋转矩阵 (世界到相机)
    focal_length: float  # 焦距
    width: int  # 图像宽度
    height: int  # 图像高度

    def get_view_matrix(self):
        """获取视图矩阵"""
        view_matrix = np.eye(4)
        view_matrix[:3, :3] = self.rotation
        view_matrix[:3, 3] = -self.rotation @ self.position
        return view_matrix

    def get_projection_matrix(self, near=0.1, far=100.0):
        """获取投影矩阵"""
        aspect = self.width / self.height
        fov_y = 2 * np.arctan(self.height / (2 * self.focal_length))

        f = 1.0 / np.tan(fov_y / 2)
        proj = np.zeros((4, 4))
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2 * far * near) / (near - far)
        proj[3, 2] = -1
        return proj


class GaussianRenderer:
    """简化的高斯渲染器（主要用于深度渲染）"""

    def __init__(self, device='cuda'):
        self.device = device

    def render_depth(self, gaussians: GaussianModel, camera: Camera) -> np.ndarray:
        """渲染深度图

        Args:
            gaussians: 高斯模型
            camera: 相机参数

        Returns:
            depth_map: [H, W] 深度图
        """
        # 转换到相机坐标系
        view_matrix = camera.get_view_matrix()

        # 世界坐标 -> 相机坐标
        positions_homo = np.concatenate([gaussians.positions, np.ones((len(gaussians.positions), 1))], axis=1)
        positions_cam = (view_matrix @ positions_homo.T).T
        positions_cam = positions_cam[:, :3]

        # 深度值（相机坐标系中的Z值，正值表示在相机前方）
        depths = -positions_cam[:, 2]  # 注意：相机看向-Z方向，所以需要取负

        # 投影到图像平面
        fx = fy = camera.focal_length
        cx = camera.width / 2
        cy = camera.height / 2

        # 使用负Z进行投影（相机看向-Z）
        x_proj = (positions_cam[:, 0] / -positions_cam[:, 2]) * fx + cx
        y_proj = (positions_cam[:, 1] / -positions_cam[:, 2]) * fy + cy

        # 创建深度图
        depth_map = np.zeros((camera.height, camera.width))
        weight_map = np.zeros((camera.height, camera.width))

        for i in range(len(gaussians.positions)):
            if depths[i] <= 0:  # 在相机后面
                continue

            x, y = int(x_proj[i]), int(y_proj[i])
            if 0 <= x < camera.width and 0 <= y < camera.height:
                # 简化的高斯splat，使用不透明度作为权重
                opacity = gaussians.opacities[i, 0]

                # 在小范围内splat
                radius = max(1, int(np.max(gaussians.scales[i]) * camera.focal_length / max(depths[i], 0.01)))
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        px, py = x + dx, y + dy
                        if 0 <= px < camera.width and 0 <= py < camera.height:
                            # 高斯权重
                            dist_sq = dx*dx + dy*dy
                            weight = opacity * np.exp(-dist_sq / (2 * radius * radius))

                            # 深度混合（前景优先）
                            if weight > weight_map[py, px] * 0.1:
                                if weight_map[py, px] == 0 or depths[i] < depth_map[py, px]:
                                    depth_map[py, px] = depths[i]
                                    weight_map[py, px] = weight

        return depth_map


class OcclusionSimulator:
    """遮挡模拟（高斯剔除）"""

    def __init__(self, occlusion_threshold: float = 0.1):
        """
        Args:
            occlusion_threshold: 深度差阈值，超过此阈值认为被遮挡
        """
        self.occlusion_threshold = occlusion_threshold
        self.renderer = GaussianRenderer()

    def simulate(self, gaussians: GaussianModel,
                 source_camera: Camera,
                 novel_cameras: list) -> GaussianModel:
        """遮挡模拟

        Args:
            gaussians: 输入高斯模型
            source_camera: 原始视角相机
            novel_cameras: 新视角相机列表（相机轨迹）

        Returns:
            degraded_gaussians: 剔除遮挡高斯点后的模型
        """
        print(f"原始高斯点数量: {len(gaussians.positions)}")

        # 对每个新视角判断遮挡
        visible_mask = np.ones(len(gaussians.positions), dtype=bool)

        for cam_idx, novel_cam in enumerate(novel_cameras):
            print(f"处理新视角 {cam_idx + 1}/{len(novel_cameras)}...")

            # 渲染新视角的深度图
            depth_map = self.renderer.render_depth(gaussians, novel_cam)

            # 计算每个高斯点在新视角的深度
            view_matrix = novel_cam.get_view_matrix()
            positions_homo = np.concatenate([gaussians.positions, np.ones((len(gaussians.positions), 1))], axis=1)
            positions_cam = (view_matrix @ positions_homo.T).T
            gaussian_depths = -positions_cam[:, 2]  # 相机看向-Z

            # 投影到图像平面
            fx = fy = novel_cam.focal_length
            cx = novel_cam.width / 2
            cy = novel_cam.height / 2

            x_proj = (positions_cam[:, 0] / -positions_cam[:, 2]) * fx + cx
            y_proj = (positions_cam[:, 1] / -positions_cam[:, 2]) * fy + cy

            # 检查每个高斯点是否被遮挡
            for i in range(len(gaussians.positions)):
                if not visible_mask[i]:
                    continue

                x, y = int(x_proj[i]), int(y_proj[i])

                # 边界检查
                if not (0 <= x < novel_cam.width and 0 <= y < novel_cam.height):
                    continue

                # 深度比较
                rendered_depth = depth_map[y, x]
                if rendered_depth > 0:  # 有渲染的深度值
                    # 如果高斯点的深度远大于渲染深度，说明被遮挡
                    if gaussian_depths[i] - rendered_depth > self.occlusion_threshold:
                        visible_mask[i] = False

        # 剔除被遮挡的高斯点
        degraded_gaussians = GaussianModel(
            positions=gaussians.positions[visible_mask],
            scales=gaussians.scales[visible_mask],
            rotations=gaussians.rotations[visible_mask],
            opacities=gaussians.opacities[visible_mask],
            colors=gaussians.colors[visible_mask]
        )

        removed_count = np.sum(~visible_mask)
        print(f"剔除 {removed_count} 个被遮挡的高斯点")
        print(f"剩余高斯点数量: {len(degraded_gaussians.positions)}")

        return degraded_gaussians


class GeometricFilterSimulator:
    """飞边与畸变模拟（平均几何滤波器）"""

    def __init__(self, filter_size: int = 5):
        """
        Args:
            filter_size: 平均滤波器核大小，越大产生越严重的畸变
        """
        self.filter_size = filter_size
        self.renderer = GaussianRenderer()

    def simulate(self, gaussians: GaussianModel,
                 source_camera: Camera,
                 novel_camera: Camera) -> GaussianModel:
        """飞边与畸变模拟

        Args:
            gaussians: 输入高斯模型
            source_camera: 原始视角相机
            novel_camera: 新视角相机

        Returns:
            degraded_gaussians: 调整后的高斯模型
        """
        print(f"应用几何滤波模拟，滤波器大小: {self.filter_size}x{self.filter_size}")

        # 1. 渲染新视角的原始深度图
        depth_map = self.renderer.render_depth(gaussians, novel_camera)

        # 检查是否有有效深度值
        if np.any(depth_map > 0):
            print(f"原始深度图范围: [{depth_map[depth_map > 0].min():.2f}, {depth_map[depth_map > 0].max():.2f}]")
        else:
            print("警告: 深度图中没有有效值，跳过几何滤波")
            return gaussians.copy()

        # 2. 应用平均滤波器
        # 保留零值不参与滤波
        mask = depth_map > 0
        filtered_depth = uniform_filter(depth_map, size=self.filter_size, mode='constant')

        # 对于有效深度区域，使用局部平均
        weight_map = uniform_filter(mask.astype(float), size=self.filter_size, mode='constant')
        filtered_depth = np.where(weight_map > 0, filtered_depth / (weight_map + 1e-6), 0)

        if np.any(filtered_depth > 0):
            print(f"滤波后深度图范围: [{filtered_depth[filtered_depth > 0].min():.2f}, {filtered_depth[filtered_depth > 0].max():.2f}]")
        else:
            print("警告: 滤波后深度图中没有有效值")

        # 3. 调整高斯点位置
        degraded_gaussians = gaussians.copy()

        # 计算每个高斯点在新视角的投影
        view_matrix = novel_camera.get_view_matrix()
        positions_homo = np.concatenate([gaussians.positions, np.ones((len(gaussians.positions), 1))], axis=1)
        positions_cam = (view_matrix @ positions_homo.T).T[:, :3]

        # 投影到图像平面
        fx = fy = novel_camera.focal_length
        cx = novel_camera.width / 2
        cy = novel_camera.height / 2

        depths = -positions_cam[:, 2]  # 相机看向-Z
        x_proj = (positions_cam[:, 0] / -positions_cam[:, 2]) * fx + cx
        y_proj = (positions_cam[:, 1] / -positions_cam[:, 2]) * fy + cy

        adjusted_count = 0
        for i in range(len(gaussians.positions)):
            x, y = int(x_proj[i]), int(y_proj[i])

            if not (0 <= x < novel_camera.width and 0 <= y < novel_camera.height):
                continue

            original_depth = depths[i]
            filtered_depth_value = filtered_depth[y, x]

            if filtered_depth_value > 0 and original_depth > 0:
                # 计算深度调整比例
                depth_ratio = filtered_depth_value / original_depth

                # 沿着相机-点的方向调整位置
                # 在相机坐标系中调整深度
                adjusted_pos_cam = positions_cam[i] * depth_ratio

                # 转换回世界坐标
                view_matrix_inv = np.linalg.inv(view_matrix)
                adjusted_pos_homo = np.concatenate([adjusted_pos_cam, [1.0]])
                adjusted_pos_world = (view_matrix_inv @ adjusted_pos_homo)[:3]

                degraded_gaussians.positions[i] = adjusted_pos_world
                adjusted_count += 1

        print(f"调整了 {adjusted_count} 个高斯点的位置")

        return degraded_gaussians


class NeoVerseDegradation:
    """NeoVerse完整降质流程"""

    def __init__(self,
                 occlusion_threshold: float = 0.1,
                 filter_size: int = 5):
        """
        Args:
            occlusion_threshold: 遮挡判定阈值
            filter_size: 几何滤波器大小
        """
        self.occlusion_sim = OcclusionSimulator(occlusion_threshold)
        self.geometric_sim = GeometricFilterSimulator(filter_size)

    def apply_occlusion_degradation(self,
                                    gaussians: GaussianModel,
                                    source_camera: Camera,
                                    novel_cameras: list) -> GaussianModel:
        """应用遮挡降质（图2a）"""
        print("\n=== 应用遮挡降质 (图2a) ===")
        return self.occlusion_sim.simulate(gaussians, source_camera, novel_cameras)

    def apply_flying_edge_degradation(self,
                                      gaussians: GaussianModel,
                                      source_camera: Camera,
                                      novel_camera: Camera) -> GaussianModel:
        """应用飞边降质（图2b）"""
        print("\n=== 应用飞边降质 (图2b) ===")
        return self.geometric_sim.simulate(gaussians, source_camera, novel_camera)

    def apply_distortion_degradation(self,
                                     gaussians: GaussianModel,
                                     source_camera: Camera,
                                     novel_camera: Camera,
                                     large_filter_size: int = 11) -> GaussianModel:
        """应用严重畸变降质（图2c）"""
        print("\n=== 应用畸变降质 (图2c) ===")
        original_filter_size = self.geometric_sim.filter_size
        self.geometric_sim.filter_size = large_filter_size
        result = self.geometric_sim.simulate(gaussians, source_camera, novel_camera)
        self.geometric_sim.filter_size = original_filter_size
        return result


# ========== 辅助函数 ==========

def create_camera_trajectory(center: np.ndarray,
                            radius: float,
                            num_views: int,
                            height: float = 0.0,
                            focal_length: float = 500.0,
                            image_size: Tuple[int, int] = (800, 600)) -> list:
    """创建环绕相机轨迹

    Args:
        center: 环绕中心点
        radius: 环绕半径
        num_views: 视角数量
        height: 相机高度偏移
        focal_length: 焦距
        image_size: 图像尺寸 (width, height)

    Returns:
        cameras: 相机列表
    """
    cameras = []

    for i in range(num_views):
        angle = 2 * np.pi * i / num_views

        # 相机位置
        x = center[0] + radius * np.cos(angle)
        z = center[2] + radius * np.sin(angle)
        y = center[1] + height
        position = np.array([x, y, z])

        # 相机朝向中心
        forward = center - position
        forward = forward / np.linalg.norm(forward)

        # 构建旋转矩阵
        up = np.array([0, 1, 0])
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)

        rotation = np.stack([right, up, -forward], axis=0)  # 世界到相机

        camera = Camera(
            position=position,
            rotation=rotation,
            focal_length=focal_length,
            width=image_size[0],
            height=image_size[1]
        )
        cameras.append(camera)

    return cameras


def save_gaussian_model(gaussians: GaussianModel, filepath: str):
    """保存高斯模型到文件"""
    np.savez(filepath,
             positions=gaussians.positions,
             scales=gaussians.scales,
             rotations=gaussians.rotations,
             opacities=gaussians.opacities,
             colors=gaussians.colors)
    print(f"模型已保存到: {filepath}")


def load_gaussian_model(filepath: str) -> GaussianModel:
    """从文件加载高斯模型"""
    data = np.load(filepath)
    return GaussianModel(
        positions=data['positions'],
        scales=data['scales'],
        rotations=data['rotations'],
        opacities=data['opacities'],
        colors=data['colors']
    )
