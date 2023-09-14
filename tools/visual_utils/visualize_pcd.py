import argparse
import numpy as np
import open3d as o3d

def load_bin_file(bin_file):
    point_cloud = np.fromfile(bin_file, dtype=np.float32)
    return point_cloud.reshape(-1, 4)

def load_label_file(label_file):
    labels = []
    with open(label_file, 'r') as file:
        for line in file:
            x, y, z, dx, dy, dz, yaw, class_name = line.split()
            labels.append((float(x), float(y), float(z), float(dx), float(dy), float(dz), float(yaw), class_name))
    return labels

def create_box_lineset(box, color=[1, 0, 0], line_width=0.01):
    vertices = np.asarray(box.get_box_points())
    edges = [[0, 1], [0, 3], [0, 4], [1, 2], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 7], [5, 6], [6, 7]]
    
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(vertices)
    lineset.lines = o3d.utility.Vector2iVector(edges)
    lineset.colors = o3d.utility.Vector3dVector([color for i in range(len(edges))])
    lineset.width = line_width
    
    return lineset

def visualize_point_cloud_with_labels(bin_file, label_file):
    point_cloud_data = load_bin_file(bin_file)
    labels = load_label_file(label_file)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])

    geometries = [pcd]

    for (x, y, z, dx, dy, dz, yaw, class_name) in labels:
        print(f'Label: {class_name} at location: (x={x}, y={y}, z={z}) with dimensions: (dx={dx}, dy={dy}, dz={dz}) and yaw: {yaw}')
        box = o3d.geometry.OrientedBoundingBox(center=(x, y, z),
                                               extent=(dx, dy, dz),
                                               R=o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw)))
        lineset = create_box_lineset(box)
        geometries.append(lineset)

    o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize a point cloud with labels.')
    parser.add_argument('--bin_file', required=True, help='Path to the .bin file')
    parser.add_argument('--label_file', required=True, help='Path to the label .txt file')
    args = parser.parse_args()

    visualize_point_cloud_with_labels(args.bin_file, args.label_file)