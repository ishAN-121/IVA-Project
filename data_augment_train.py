import os
import shutil
import random
import yaml
import torch
from torch import nn
from glob import glob
from ultralytics import YOLO
from tqdm import tqdm

if not torch.cuda.is_available():
    raise RuntimeError("🚨 CUDA GPU not available! Please run on a system with GPU.")
else:
    print("✅ GPU detected:", torch.cuda.get_device_name(0))

# ====== PATHS ======
base_dir = 'your_dataset'  
img_dir = os.path.join(base_dir, 'images/img')
lbl_dir = os.path.join(base_dir, 'labels/lbl')

# Output directories
out_img_train = os.path.join(base_dir, 'images/train')
out_img_val = os.path.join(base_dir, 'images/val')
out_lbl_train = os.path.join(base_dir, 'labels/train')
out_lbl_val = os.path.join(base_dir, 'labels/val')

# Create folders
for d in [out_img_train, out_img_val, out_lbl_train, out_lbl_val]:
    os.makedirs(d, exist_ok=True)

# ====== SPLIT DATA ======
image_files = glob(os.path.join(img_dir, '*.jpg'))  # change to .png if needed
random.seed(42)
random.shuffle(image_files)
split_idx = int(0.8 * len(image_files))
train_imgs = image_files[:split_idx]
val_imgs = image_files[split_idx:]

def move_pairs(img_list, dest_img, dest_lbl):
    count = 0
    for img in tqdm(img_list,desc="copying files"):
        base = os.path.basename(img)
        lbl = os.path.join(lbl_dir, base.replace('.jpg', '.txt'))
        shutil.copy(img, os.path.join(dest_img, base))
        if os.path.exists(lbl):
            shutil.copy(lbl, os.path.join(dest_lbl, os.path.basename(lbl)))
        else:
            count +=1
            # print(f"⚠️ No label found for {base}, skipping label.")
    print(f"Images skipped = ",count)

move_pairs(train_imgs, out_img_train, out_lbl_train)
move_pairs(val_imgs, out_img_val, out_lbl_val)

yaml_path = os.path.join(base_dir, 'data.yaml')
with open(yaml_path, 'r') as file:
    data = yaml.safe_load(file)
print("The data.yaml is as follows:")
print(data)

#====== TRAIN MODEL ON GPU ======
print("\n🚀 Starting training with YOLOv8m on GPU...\n")
model = YOLO('yolo11n.pt')  

model.train(
    data=yaml_path,
    epochs=5,
    imgsz=1280,
    batch=24,  # adjust based on your GPU memory
    device=0,  # explicitly use GPU 0
    name='yolov11_data_augment',
    project='yolo_runs',
    exist_ok=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    perspective=0.0005,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1
)
