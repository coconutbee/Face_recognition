import os
import shutil
from collections import OrderedDict
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2

# from metrics.fid import calculate_fid_given_paths
# from metrics.lpips import calculate_lpips_given_images
# from core.data_loader_lm_perceptual import get_eval_loader_vgg, get_eval_loader_2
# from core import utils_lm
from PIL import Image
from ms1m_ir50.model_irse import IR_50
import math
# import network

# 定义FAKE和GT文件夹的路径
from math import cos, sin, atan2, asin, sqrt
import numpy as np
from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from TDDFA_ONNX import TDDFA_ONNX
import yaml
# 加载3DDFA-V2模型配置
cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)

# 初始化FaceBoxes和TDDFA
face_boxes = FaceBoxes_ONNX()
tddfa = TDDFA_ONNX(**cfg)

def plot_csim_distribution(csim_scores):
        bins = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        csim_scores_int = [csim_score.item() for csim_score in csim_scores]
        plt.hist(csim_scores_int, bins=bins, edgecolor='black', alpha=0.7)
        # plt.bar(edges[:-1], hist, width=np.diff(bins), edgecolor='black', alpha=0.7)
        plt.show()

def P2sRt(P):
    """ decompositing camera matrix P.
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation.
    """
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d
def matrix2angle(R):
    """ compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    refined by: https://stackoverflow.com/questions/43364900/rotation-matrix-to-euler-angles-with-opencv
    todo: check and debug
     Args:
         R: (3,3). rotation matrix
     Returns:
         x: yaw
         y: pitch
         z: roll
     """
    if R[2, 0] > 0.998:
        z = 0
        x = np.pi / 2
        y = z + atan2(-R[0, 1], -R[0, 2])
    elif R[2, 0] < -0.998:
        z = 0
        x = -np.pi / 2
        y = -z + atan2(R[0, 1], R[0, 2])
    else:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

    return x, y, z
def angle2matrix(angles):
    ''' get rotation matrix from three rotation angles(radian). The same as in 3DDFA.
    Args:
        angles: [3,]. x, y, z angles
        x: yaw.
        y: pitch.
        z: roll.
    Returns:
        R: 3x3. rotation matrix.
    '''
    # x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    # x, y, z = angles[0], angles[1], angles[2]
    y, x, z = angles[0], angles[1], angles[2]

    # x
    Rx = np.array([[1, 0, 0],
                   [0, cos(x), -sin(x)],
                   [0, sin(x), cos(x)]])
    # y
    Ry = np.array([[cos(y), 0, sin(y)],
                   [0, 1, 0],
                   [-sin(y), 0, cos(y)]])
    # z
    Rz = np.array([[cos(z), -sin(z), 0],
                   [sin(z), cos(z), 0],
                   [0, 0, 1]])
    R = Rz.dot(Ry).dot(Rx)
    return R.astype(np.float32)





# 定义函数计算两张图像之间的旋转误差
def calculate_rotation(image):

    # 使用FaceBoxes检测人脸
    boxes = face_boxes(image)
    
    n = len(boxes)
    if n == 0:
        print(f'No face detected, exit')
        return None

    # 使用3DDFA-V2进行3D姿势估计
    param_lst, roi_box_lst = tddfa(image, boxes)
    
    try:
        param_lst, roi_box_lst = tddfa(image, boxes)
        param = param_lst[0]
    except Exception as e:
        print(f'3D pose estimation failed, skipping')
        return None
    
    P1 = param[:12].reshape(3, -1).copy()  # camera matrix
    s, R1, t3d = P2sRt(P1)
    angle = matrix2angle(R1)
    yaw, pitch, roll = angle
    return yaw* (180/math.pi)


def calculate_csim_pairwise(image_path):
    """
    Compare all images in the folder pairwise using ArcFace and save the CSIM results and averages to a single text file.
    Also prints top 10 lowest and highest CSIM pairs.
    """
    convert_tensor = transforms.ToTensor()
    print("Loading Arcface model...")

    BACKBONE_RESUME_ROOT = './ms1m_ir50/backbone_ir50_ms1m_epoch63.pth'
    INPUT_SIZE = [112, 112]
    arcface = IR_50(INPUT_SIZE)

    if os.path.isfile(BACKBONE_RESUME_ROOT):
        arcface.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
        print(f"Loaded ArcFace model from {BACKBONE_RESUME_ROOT}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arcface = arcface.to(device)
    arcface.eval()

    # List and sort images
    image_files = sorted(os.listdir(image_path))
    csim_results = []  # List to store all results
    all_csim_values = []  # List to store all CSIM values for overall average calculation
    all_csim_pairs = []  # List to store all image pairs and their CSIM scores

    for i, image1_file in enumerate(tqdm(image_files, desc="Processing Images")):
        image1_path = os.path.join(image_path, image1_file)
        image1 = cv2.imread(image1_path)
        if image1 is None:
            print(f"Failed to load image: {image1_path}, skipping.")
            continue

        image1 = cv2.resize(image1, (112, 112))
        image1 = convert_tensor(image1).to(device).reshape(1, 3, 112, 112)

        # Normalize image
        image1 = nn.functional.interpolate(image1, size=(112, 112), mode='bilinear')

        # Calculate embedding for the first image
        with torch.no_grad():
            image1_emb = arcface(image1)

        # Initialize list to store CSIM values for the current image
        csim_values = []

        for j, image2_file in enumerate(image_files):
            if i == j:
                continue

            image2_path = os.path.join(image_path, image2_file)
            image2 = cv2.imread(image2_path)
            if image2 is None:
                print(f"Failed to load image: {image2_path}, skipping.")
                continue

            image2 = cv2.resize(image2, (112, 112))
            image2 = convert_tensor(image2).to(device).reshape(1, 3, 112, 112)

            # Normalize image
            image2 = nn.functional.interpolate(image2, size=(112, 112), mode='bilinear')

            # Calculate embedding for the second image
            with torch.no_grad():
                image2_emb = arcface(image2)

            # Compute cosine similarity
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            csim = cos(image1_emb, image2_emb).item()
            csim_values.append(csim)
            all_csim_values.append(csim)  # Add to overall list
            all_csim_pairs.append((image1_file, image2_file, csim))  # Add pair and score

        # Compute average CSIM for the current image
        avg_csim = sum(csim_values) / len(csim_values) if csim_values else 0
        csim_results.append({
            "image": image1_file,
            "average_csim": avg_csim,
            "pairwise_csim": csim_values
        })

    # Compute overall average CSIM
    overall_avg_csim = sum(all_csim_values) / len(all_csim_values) if all_csim_values else 0

    # Save results to a single text file
    output_file = os.path.join(
        image_path,
        f"black_male_75_{overall_avg_csim:.4f}.txt"  
    )
    with open(output_file, 'w') as f:
        for result in csim_results:
            f.write(f"Image: {result['image']}\n")
            f.write(f"Average CSIM: {result['average_csim']:.4f}\n")
            f.write(f"Pairwise CSIM: {', '.join(f'{csim:.4f}' for csim in result['pairwise_csim'])}\n")
            f.write("\n")
        f.write(f"Overall Average CSIM: {overall_avg_csim:.4f}\n")

    # Sort pairs by CSIM scores
    all_csim_pairs.sort(key=lambda x: x[2])

    # Print top 10 lowest CSIM pairs
    print("\nTop 10 Lowest CSIM Pairs:")
    for pair in all_csim_pairs[:10]:
        print(f"{pair[0]} <-> {pair[1]}: CSIM = {pair[2]:.4f}")

    # Print top 10 highest CSIM pairs
    print("\nTop 10 Highest CSIM Pairs:")
    for pair in all_csim_pairs[-10:]:
        print(f"{pair[0]} <-> {pair[1]}: CSIM = {pair[2]:.4f}")

    # Print overall average CSIM
    print(f"Overall Average CSIM: {overall_avg_csim:.4f}")
    print(f"CSIM results saved to {output_file}")


# Path to the folder containing images
image_path = '' # Add your image folder path here

# Run the CSIM calculation
calculate_csim_pairwise(image_path)

