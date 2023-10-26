import os
import sys
import open3d
import copy
sys.path.append(os.getcwd())
from pcdet.utils.calibration_kitti import *
import shutil
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

raw_root_path = "/data/LOCO_annotated/"
processed_root_path = "data/custom/"
transformed = raw_root_path + "transformed/"
untransformed = raw_root_path + "untransformed/"

transformed_paths = [transformed + dir + '/' for dir in os.listdir(transformed)]
untransformed_paths = [untransformed + dir + '/' for dir in os.listdir(untransformed)]

def kitti2custom(path, transformed):
    if os.path.exists(path + "custom_labels") == False:
        os.mkdir(path + "custom_labels")
    label = [path + "labels/" + file for file in os.listdir(path + "labels/")]
    velodyne = [path + "velodyne/" + file for file in os.listdir(path + "velodyne/")]
    label.sort()
    velodyne.sort()

    if transformed:
        calib = [path + "calib/" + file for file in os.listdir(path + "calib/")]
        img = [path + "images/" + file for file in os.listdir(path + "images/")]
        calib.sort()
        img.sort()


# sort files
    for idx, label_file in enumerate(label):
        if os.path.basename(label_file)[-3:] == "ini":
            continue
        velodyne_file = path + "velodyne/" + os.path.basename(label_file)[:-3] + "bin"
        # save to untransformed
        if transformed:
            kitti_calib = Calibration(calib[idx])
        label_file = label[idx]

        bin = np.fromfile(velodyne_file, dtype=np.float32)
        points = bin.reshape(-1, 4)
        points = points[:, :3]
        points = points.reshape(-1, 3)
        points = points.astype('float64')
        pcl = open3d.geometry.PointCloud()
        pcl.points = open3d.utility.Vector3dVector(points)

        custom_label_file = path + "custom_labels/" + os.path.basename(label_file)

        # if os.path.exists(custom_label_file):
        #     continue

        print(f"Processing {os.path.basename(velodyne_file)} in {path.split('/')[-2]}")

        # open custom label file
        with open(custom_label_file, 'w') as f_w:
            # Read labels from label file

            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip().split(' ')
                    dx, dy, dz, x, y, z, r = line[-7:]
                    if transformed:
                        x, y, z, dx, dy, dz, r = boxes3d_kitti_camera_to_lidar(np.array([[x, y, z, dx, dz, dy, r]], dtype=np.float32), kitti_calib)[0,:]
                    custom_line = np.array([x, y, z, dx, dy, dz, r], dtype=np.float32)
                    lines, _ = translate_boxes_to_open3d_instance(custom_line)
                    for i in custom_line:
                        f_w.write(str(i) + ' ')
                    f_w.write('Vehicle\n')


        # open3d.visualization.draw_geometries([pcl, lines])

def merge(root, dirs):
    if os.path.exists(root + "points") == False:
        os.mkdir(root + "points")
    if os.path.exists(root + "labels") == False:
        os.mkdir(root + "labels")
    for dir in dirs:
        labels = [dir + "custom_labels/" + label for label in os.listdir(dir + "custom_labels/")]
        points = [dir + "velodyne/" + point for point in os.listdir(dir + "velodyne/")]

        labels.sort()
        points.sort()

        # copy labels and points to merged folder
        for idx, label in enumerate(labels):
            print(f"copying {os.path.basename(label)[:-4]} to {root}")
            new_name = label.split('/')[-3] + "_" + label.split('/')[-1].split('_')[-1][:-3]
            shutil.copy(label, root + "labels/" + new_name + "txt")
            shutil.copy(dir + "velodyne/" + os.path.basename(label)[:-3] + "bin", root + "points/" + new_name + "bin")

def get_split(processed_root):
    if os.path.exists(processed_root + "ImageSets") == False:
        os.mkdir(processed_root + "ImageSets")
    labels = [processed_root + "labels/" + label[:-4] for label in os.listdir(processed_root + "labels/")]
    # Generate train.txt which is 20 precent of the labels samplied uniformly
    train, test = train_test_split(labels, test_size=0.2, random_state=42)
    with open(processed_root + "ImageSets/train.txt", 'w') as f:
        for label in train:
            f.write(os.path.basename(label) + '\n')
    
    with open(processed_root + "ImageSets/val.txt", 'w') as f:
        for label in test:
            f.write(os.path.basename(label) + '\n')

for dir in transformed_paths:
    kitti2custom(dir, transformed=True)
for idx, dir in enumerate(untransformed_paths):
    kitti2custom(dir, transformed=False)

merge(processed_root_path, transformed_paths + untransformed_paths)
get_split(processed_root_path)