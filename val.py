import os
from ultralytics import YOLO
import torch
from PIL import Image, ImageDraw, ImageFont

# Paths
runs_root = ''
val_data_path = ''
dataset_yaml = ''
save_root = ''

# Custom class label mapping (index 0 to 11)
class_labels = [
    'biker', 'car', 'pedestrian', 'trafficlight', 'trafficlight-green', 'trafficlight-greenleft',
    'trafficlight-red', 'trafficlight-redleft', 'trafficlight-yellow', 'trafficlight-yellowleft', 'truck', 'arret'
]

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Ensure results/val directory exists
os.makedirs(save_root, exist_ok=True)

for subdir in os.listdir(runs_root):
    model_dir = os.path.join(runs_root, subdir)
    weights_path = os.path.join(model_dir, 'weights', 'best.pt')

    if not os.path.isfile(weights_path):
        print(f"[!] Skipping {subdir}, no weights found at expected path.")
        continue

    print(f"[+] Evaluating model: {subdir}")

    # Load the model
    model = YOLO(weights_path)
    model.to(device)

    # Run validation and print mAP
    results = model.val(data=dataset_yaml, device=device)
    print(f"mAP@0.5: {results.box.map50:.4f}")

    # Run inference on validation images
    results = model(val_data_path, device=device)

    # Directory to save this model's predictions
    model_save_dir = os.path.join(save_root, subdir)
    os.makedirs(model_save_dir, exist_ok=True)
    os.rename("/home/ishan/IVA/runs/detect/val",os.path.join("/home/ishan/IVA/runs/detect",subdir))

    # Save images with bounding boxes and custom labels
    for i, pred in enumerate(results):
        img = Image.fromarray(pred.orig_img) 
        result_image_path = os.path.join(model_save_dir, f"predicted_{i}.jpg")

        # Convert to PIL image if needed
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        draw = ImageDraw.Draw(img)

        if pred.boxes is not None:
            for box in pred.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                label = f"{class_labels[cls_id]} {conf:.2f}"
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, y1 - 12), label, fill="white", font=font)

        img.save(result_image_path)
        print(f"Saved with labels: {result_image_path}")

print("\n✅ All models evaluated and predictions saved.")
