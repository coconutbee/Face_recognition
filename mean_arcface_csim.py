# 用來計算兩個folder對應向同名稱的相似度，最後會輸出一個平均的相似度
# pip install -r requirements.txt

import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
from ms1m_ir50.model_irse import IR_50  # 假设 Arcface 模型 IR_50 定义在此处

# 初始化Arcface模型用于相似度计算
def load_arcface_model():
    model = IR_50([112, 112])
    BACKBONE_RESUME_ROOT = './ms1m_ir50/backbone_ir50_ms1m_epoch63.pth'
    
    if os.path.isfile(BACKBONE_RESUME_ROOT):
        model.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
        print("Loaded Arcface model weights.")
    else:
        raise FileNotFoundError(f"Arcface model weights not found at {BACKBONE_RESUME_ROOT}")

    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model

# 加载和转换图像的工具函数
convert_tensor = transforms.ToTensor()

# 计算单对图像的相似度
def calculate_similarity(arcface_model, fake_image, gt_image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fake_image = convert_tensor(fake_image).to(device).reshape(1, 3, 112, 112)
    gt_image = convert_tensor(gt_image).to(device).reshape(1, 3, 112, 112)
    
    with torch.no_grad():
        fake_emb = arcface_model(fake_image)
        gt_emb = arcface_model(gt_image)
    
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    similarity = cos(fake_emb, gt_emb).item()
    return similarity

# 从路径加载图像并调整大小
def load_and_prepare_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((112, 112))
    return image

# 示例调用方式，用于与 `generate_csim.py` 集成
# 使用方式：
# arcface_model = load_arcface_model()
# similarity = calculate_similarity(arcface_model, fake_image, gt_image)
