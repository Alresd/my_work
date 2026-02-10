"""
简单的PLY文件加载器，无需外部依赖
支持 3D Gaussian Splatting 的二进制PLY格式
"""

import numpy as np
import struct


def read_ply_gaussian(filepath):
    """
    读取二进制PLY格式的3D Gaussian Splatting模型

    Args:
        filepath: PLY文件路径

    Returns:
        dict: 包含各属性的字典
    """
    with open(filepath, 'rb') as f:
        # 读取头部
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

        # 解析头部
        num_vertices = 0
        properties = []

        for line in header_lines:
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                prop_type = parts[1]
                prop_name = parts[2]
                properties.append((prop_name, prop_type))

        print(f"读取PLY文件: {filepath}")
        print(f"顶点数量: {num_vertices}")
        print(f"属性数量: {len(properties)}")

        # 构建读取格式
        type_map = {
            'float': ('f', 4),
            'double': ('d', 8),
            'uchar': ('B', 1),
            'uint': ('I', 4),
            'int': ('i', 4),
        }

        fmt = '<'  # 小端序
        byte_size = 0
        for _, prop_type in properties:
            if prop_type in type_map:
                fmt += type_map[prop_type][0]
                byte_size += type_map[prop_type][1]

        # 读取所有顶点数据
        data = {}
        for prop_name, _ in properties:
            data[prop_name] = []

        for i in range(num_vertices):
            vertex_bytes = f.read(byte_size)
            values = struct.unpack(fmt, vertex_bytes)

            for j, (prop_name, _) in enumerate(properties):
                data[prop_name].append(values[j])

            if (i + 1) % 50000 == 0:
                print(f"  已读取 {i + 1}/{num_vertices} 个顶点...")

        # 转换为numpy数组
        for key in data:
            data[key] = np.array(data[key], dtype=np.float32)

        print(f"读取完成！")

    return data


def write_ply_gaussian(filepath, data):
    """
    写入二进制PLY格式的3D Gaussian Splatting模型

    Args:
        filepath: 输出PLY文件路径
        data: 包含各属性的字典
    """
    num_vertices = len(data['x'])

    print(f"写入PLY文件: {filepath}")
    print(f"顶点数量: {num_vertices}")

    # 属性顺序（标准3DGS格式）
    properties = [
        ('x', 'float'), ('y', 'float'), ('z', 'float'),
        ('nx', 'float'), ('ny', 'float'), ('nz', 'float'),
        ('f_dc_0', 'float'), ('f_dc_1', 'float'), ('f_dc_2', 'float'),
        ('opacity', 'float'),
        ('scale_0', 'float'), ('scale_1', 'float'), ('scale_2', 'float'),
        ('rot_0', 'float'), ('rot_1', 'float'), ('rot_2', 'float'), ('rot_3', 'float')
    ]

    with open(filepath, 'wb') as f:
        # 写入头部
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write(f'element vertex {num_vertices}\n'.encode('ascii'))

        for prop_name, prop_type in properties:
            f.write(f'property {prop_type} {prop_name}\n'.encode('ascii'))

        f.write(b'end_header\n')

        # 写入数据
        for i in range(num_vertices):
            vertex_data = []
            for prop_name, _ in properties:
                vertex_data.append(float(data[prop_name][i]))

            # 打包为二进制
            vertex_bytes = struct.pack('<' + 'f' * len(vertex_data), *vertex_data)
            f.write(vertex_bytes)

            if (i + 1) % 50000 == 0:
                print(f"  已写入 {i + 1}/{num_vertices} 个顶点...")

    print(f"写入完成！")
