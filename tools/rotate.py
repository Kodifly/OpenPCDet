import os
import cv2
import numpy as np
from tqdm import tqdm

def applyRT_with_I(scan,RT):
    #Extracting xyz and intensity
    points = scan[:,0:3]

    rotated = applyRT(points,RT)

    #Concatinating the only
    intensityd = scan[:, 3].reshape(1,scan.shape[0]).T
    rotated = np.concatenate([rotated, intensityd],axis=-1)

    return rotated

def applyRT(scan,RT):
    points = scan[:,0:3]
    velo = np.insert(points,3,1,axis=1).T
    rotated = RT.dot(velo)
    rotated = rotated.T[:,0:3]

    return rotated

bin_folder_path = "/data/13091115/pcl/"

bin_file = [f for f in os.listdir(bin_folder_path) if os.path.isfile(os.path.join(bin_folder_path, f))]
# label_file = [f for f in os.listdir(label_folder_path) if os.path.isfile(os.path.join(label_folder_path, f))]

for  bin in tqdm(bin_file):

    binFile = os.path.join(bin_folder_path, bin)
    lidarData = np.fromfile(binFile, dtype=np.float32)
    # print(lidarData.shape)
    lidarData = lidarData.reshape(-1, 4)

    RTStr = "0.9563047559630354 0.0 0.2923717047227367 0.0 0.0 1.0 0.0 0.0 -0.2923717047227367 0.0 0.9563047559630353 0.0 0.0 0.0 0.0 1.0"
    RT_str = RTStr.split(" ")[:]
    RT = np.reshape(np.array([float(p) for p in RT_str]), (4, 4)).astype(np.float32)
    RTi = np.linalg.inv(RT)
    lidarData2 = applyRT_with_I(lidarData, RT)
    t = os.path.basename(binFile)

    lidarData2.tofile(os.path.join('/data/13091115/pcl_/', t))
    # print(lidarData2.shape)