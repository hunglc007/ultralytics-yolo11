import cv2
import random
import json
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm

from collections import OrderedDict

class ImageCache:
    def __init__(self, max_size=2000):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, img_path):
        # If the image is in the cache, move it to the end (most recently used)
        if img_path in self.cache:
            self.cache.move_to_end(img_path)
            return self.cache[img_path]

        # Otherwise, load the image and add it to the cache
        img = cv2.imread(img_path)
        if img is not None:
            self.cache[img_path] = img
            # If the cache exceeds the max size, remove the oldest item
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
        return img

def weighted_random_choice(pool, target_index, tau=3.0):
    indices = np.array([p[-1] for p in pool])
    distances = np.abs(indices - target_index)
    weights = np.exp(-distances / tau)
    probs = weights / weights.sum()
    chosen_idx = np.random.choice(len(pool), p=probs)
    return pool[chosen_idx]

def iou(boxA, boxB):
    xa = max(boxA[0], boxB[0]); ya = max(boxA[1], boxB[1])
    xb = min(boxA[2], boxB[2]); yb = min(boxA[3], boxB[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)

def inplace_object_augment(img, labels, pool, index, image_cache, max_samples=500):
    """
    Augment image by sampling up to `max_samples` objects from 20 nearest-index images in pool
    and pasting those that do not overlap existing boxes.

    Args:
        img: np.ndarray HxWx3
        labels: list of [cls, x1, y1, x2, y2]
        pool: list of (img_path, cls, [x1,y1,x2,y2], index)
        index: int or str, current image index
        max_samples: number of objects to sample
    Returns:
        img_aug: augmented image
        new_labels: extended labels list
    """
    h, w = img.shape[:2]
    new_labels = labels.copy()
    current_boxes = [l[1:] for l in labels]

    center = int(index)

    pool_indices = [int(p[-1]) for p in pool]

    max_pool_idx = max(pool_indices)
    min_pool_idx = min(pool_indices)

    min_idx = max(min_pool_idx, center - 50)
    max_idx = min(max_pool_idx, center + 50)

    # Filter pool for 20-nearest images using pre-appended index
    filtered_pool = [
        p for p in pool
        if min_idx <= int(p[-1]) <= max_idx and not (center - 5 <= int(p[-1]) <= center + 5)
    ]

    # Sample from filtered pool safely
    sampled = random.sample(filtered_pool, min(max_samples, len(filtered_pool))) if filtered_pool else []

    start = time.time()
    for img_path, cls, bbox, _ in sampled:
        x1, y1, x2, y2 = map(int, bbox)
        width, height = x2 - x1, y2 - y1

        # Skip invalid or too large objects
        if x2 <= x1 or y2 <= y1:
            continue

        # Determine the probability of adding the object based on its size
        if width > 50 and height > 50:
            add_probability = 0.5
        elif 40 < width <= 50 and 40 < height <= 50:
            add_probability = 0.65
        elif 30 < width <= 40 and 30 < height <= 40:
            add_probability = 0.8
        else:
            add_probability = 1.0

        # Skip adding the object based on the probability
        if random.random() > add_probability:
            continue
        if any(iou([x1,y1,x2,y2], box) > 0 for box in current_boxes):
            continue
        obj_img = image_cache.get(img_path)
        if obj_img is None:
            continue
        patch = obj_img[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        img[y1:y2, x1:x2] = patch
        new_labels.append([cls, x1, y1, x2, y2])
        current_boxes.append([x1, y1, x2, y2])
    # print("Augmentation time:", time.time() - start)

    return img, new_labels

# Main function
def main():
    # Load index file
    with open('/media/hungdv/Source/Data/AICity2025/yolo_format/object_index.json') as f:
        raw_index = json.load(f)
    object_index = {tuple(k.split('_')): v for k, v in raw_index.items()}

    # Process all images in the folder
    image_folder = Path('/media/hungdv/Source/Data/AICity2025/yolo_format/images')
    output_folder = Path('/media/hungdv/Source/Data/AICity2025/yolo_format/augmented')
    output_folder.mkdir(exist_ok=True)

    image_cache = ImageCache(max_size=2000)

    image_paths = list(image_folder.glob('*.png'))

    for img_path in tqdm(image_paths, desc="Processing images", dynamic_ncols=True, total=len(image_paths)):
        img_name = img_path.name
        label_name = img_name.replace('.png', '.txt')

        img = image_cache.get(str(img_path))
        h, w = img.shape[:2]
        labels = []

        # Read YOLO labels
        label_path = Path(f'/media/hungdv/Source/Data/AICity2025/yolo_format/labels/{label_name}')
        if not label_path.exists():
            print(f"Label file {label_name} not found. Skipping...")
            continue

        with open(label_path) as f:
            for line in f:
                cls, xc, yc, bw, bh = map(float, line.strip().split())
                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w
                y2 = (yc + bh / 2) * h
                labels.append([int(cls), x1, y1, x2, y2])

        # Get object pool
        cam_id, time_seg, index = Path(img_name).stem.split('_')
        pool = object_index.get((cam_id, time_seg), [])

        # Augment
        img_aug, new_labels = inplace_object_augment(img.copy(), labels, pool, index, image_cache, max_samples=500)
        combined_img = cv2.hconcat([img, img_aug])

        # # Save the augmented image
        # output_path = output_folder / img_name
        # cv2.imwrite(str(output_path), combined_img)
        # # print(f"Augmented image saved to {output_path}")

        # Create output directories
        output_folder = Path('/media/hungdv/Source/Data/AICity2025/yolo_format/aug_fisheye')
        output_images_folder = output_folder / 'images'
        output_labels_folder = output_folder / 'labels'
        output_images_folder.mkdir(parents=True, exist_ok=True)
        output_labels_folder.mkdir(parents=True, exist_ok=True)

        # Save the augmented image in the images folder
        output_img_path = output_images_folder / f"{img_name.replace('.png', '_aug.png')}"
        cv2.imwrite(str(output_img_path), img_aug)

        # Save the augmented labels in the labels folder
        output_label_path = output_labels_folder / f"{label_name.replace('.txt', '_aug.txt')}"
        with open(output_label_path, 'w') as f:
            for label in new_labels:
                cls, x1, y1, x2, y2 = label
                xc = (x1 + x2) / 2 / w
                yc = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


# call main function
if __name__ == "__main__":
    main()

