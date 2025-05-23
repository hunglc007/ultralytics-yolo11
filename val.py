from ultralytics import YOLO
import json
import os
from pathlib import Path
from tqdm import tqdm

# Load a pretrained YOLO11n model
model = YOLO("../yolo11m-1920.pt")
groundtruth_json = "groundtruth.json"
with open(groundtruth_json, "r") as f:
    coco_gt = json.load(f)

# Tạo ánh xạ từ file_name -> image_id
filename_to_id = {img["file_name"]: img["id"] for img in coco_gt["images"]}

test_images_dir = Path("/media/hungdv/Source/Data/ai-city-challenge-2024/track4/Fisheye8K/test/images")
image_paths = list(test_images_dir.glob("*.png"))
coco_results = []

for img_info in tqdm(coco_gt["images"], desc="Predicting"):
    file_name = img_info["file_name"]
    image_id = img_info["id"]
    img_path = test_images_dir / file_name

    # Bỏ qua nếu ảnh không tồn tại
    if not img_path.exists():
        print(f"⚠️ Missing file: {img_path}")
        continue

    results = model.predict(img_path, conf=0.001, iou=0.45,
              show=False, show_conf=False, show_labels=False, imgsz=1920, save=False)
    result = results[0]

    boxes = result.boxes
    for i in range(len(boxes)):
        box = boxes[i]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        width = x2 - x1
        height = y2 - y1
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        coco_results.append({
            "image_id": image_id,
            "category_id": cls,
            "bbox": [x1, y1, width, height],
            "score": conf
        })

# Ghi kết quả ra file JSON
with open("predictions_coco.json", "w") as f:
    json.dump(coco_results, f)

print("✅ Saved: predictions_coco.json")
