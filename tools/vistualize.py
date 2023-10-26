import os
import sys
import open3d
import copy
sys.path.append(os.getcwd())
from pcdet.utils.calibration_kitti import *
import shutil
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]

    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)
    box3d.color = (1.0,0.0,0.0)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)
    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d

def boxes3d_kitti_camera_to_lidar(boxes3d_camera, calib):
    """
    Args:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
        calib:

    Returns:
        boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    """
    boxes3d_camera_copy = copy.deepcopy(boxes3d_camera)
    xyz_camera, r = boxes3d_camera_copy[:, 0:3], boxes3d_camera_copy[:, 6:7]
    l, h, w = boxes3d_camera_copy[:, 3:4], boxes3d_camera_copy[:, 4:5], boxes3d_camera_copy[:, 5:6]

    xyz_lidar = calib.rect_to_lidar(xyz_camera)
    xyz_lidar[:, 2] += h[:, 0] / 2
    return np.concatenate([xyz_lidar, l, w, h, -(r + np.pi / 2)], axis=-1)

root_path = "/data/LOCO_annotated/untransformed/03101038/"

def vistualize(path):
    velodyne = [path + "velodyne/" + file for file in os.listdir(path + "velodyne/")]
    velodyne.sort()
    
    if os.path.exists(path + "labels"):
        labels = [path + "labels/" + file for file in os.listdir(path + "labels/")]
        labels.sort()
    
        for idx, label_file in enumerate(labels):
            image_file = path + "images/" + os.path.basename(label_file)[:-3] + "jpg"
            cv2.imshow("image", cv2.imread(image_file))
            cv2.waitKey(1)
            
            velodyne_file = path + "velodyne/" + os.path.basename(label_file)[:-3] + "bin"
            bin = np.fromfile(velodyne_file, dtype=np.float32)
            points = bin.reshape(-1, 4)
            points = points[:, :3]
            points = points.reshape(-1, 3)
            points = points.astype('float64')
            pcl = open3d.geometry.PointCloud()
            pcl.points = open3d.utility.Vector3dVector(points)
            with open(label_file, 'r') as f:
                label = []
                for line in f:
                    line = line.strip().split(' ')
                    cls = line[0]
                    dx, dy, dz, x, y, z, r = line[-7:]
                    custom_line = np.array([x, y, z, dx, dy, dz, r], dtype=np.float32)
                    lines, _ = translate_boxes_to_open3d_instance(custom_line)
                    if cls == "loco":
                        lines.paint_uniform_color(np.array([0, 1, 0]))
                    else:
                        lines.paint_uniform_color(np.array([1, 0, 0]))
                    label.append(lines)
                open3d.visualization.draw_geometries([pcl, *label])
    else:
        for points in velodyne:
            bin = np.fromfile(points, dtype=np.float32)
            points = bin.reshape(-1, 4)
            points = points[:, :3]
            points = points.reshape(-1, 3)
            points = points.astype('float64')
            pcl = open3d.geometry.PointCloud()
            pcl.points = open3d.utility.Vector3dVector(points)
            open3d.visualization.draw_geometries([pcl])

# for dir in transformed_paths:
#     kitti2custom(dir, transformed=True)
# for idx, dir in enumerate(untransformed_paths):
#     kitti2custom(dir, transformed=False)

# merge(processed_root_path, transformed_paths + untransformed_paths)

# get_split(processed_root_path)
vistualize(root_path)