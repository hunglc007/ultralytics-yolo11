import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm  # Added tqdm for progress bar

def convert_coco_to_yolo(coco_json_path, image_folder, output_folder):
    # Load COCO JSON
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create output directories
    yolo_labels_folder = os.path.join(output_folder, 'labels')
    yolo_images_folder = os.path.join(output_folder, 'images')
    os.makedirs(yolo_labels_folder, exist_ok=True)
    os.makedirs(yolo_images_folder, exist_ok=True)

    # Map category IDs to indices
    category_mapping = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}

    # Process each image and its annotations with tqdm progress bar
    for image in tqdm(coco_data['images'], desc="Processing images"):
        image_id = image['id']
        image_name = image['file_name']
        image_path = os.path.join(image_folder, image_name)

        # Copy image to YOLO images folder
        shutil.copy(image_path, os.path.join(yolo_images_folder, image_name))

        # Prepare YOLO label file
        label_file_path = os.path.join(yolo_labels_folder, f"{Path(image_name).stem}.txt")
        with open(label_file_path, 'w') as label_file:
            for annotation in coco_data['annotations']:
                if annotation['image_id'] == image_id:
                    category_id = annotation['category_id']
                    bbox = annotation['bbox']  # COCO format: [x_min, y_min, width, height]

                    # Convert COCO bbox to YOLO format
                    x_min, y_min, width, height = bbox
                    x_center = x_min + width / 2
                    y_center = y_min + height / 2
                    image_width = image['width']
                    image_height = image['height']

                    # Normalize coordinates
                    x_center /= image_width
                    y_center /= image_height
                    width /= image_width
                    height /= image_height

                    # Write to label file
                    yolo_line = f"{category_mapping[category_id]} {x_center} {y_center} {width} {height}\n"
                    label_file.write(yolo_line)

if __name__ == "__main__":
    # Input paths
    coco_json_path = "/media/hungdv/Source/Data/AICity2025/Fisheye8K_all_including_train&test/train/train.json"
    image_folder = "/media/hungdv/Source/Data/AICity2025/Fisheye8K_all_including_train&test/train/images"
    output_folder = "/media/hungdv/Source/Data/AICity2025/yolo_format"

    # Convert COCO to YOLO
    convert_coco_to_yolo(coco_json_path, image_folder, output_folder)
