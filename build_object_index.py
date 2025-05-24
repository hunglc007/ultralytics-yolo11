import os
import json
import cv2
import argparse
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm  # Added tqdm for progress bar

def main(img_dir, label_dir, out_json):
    group_objects = defaultdict(list)

    label_files = list(Path(label_dir).glob('*.txt'))  # Collect all label files
    for label_file in tqdm(label_files, desc="Processing label files"):  # Add tqdm progress bar
        stem = label_file.stem  # e.g., camera1_A_111
        print(stem)
        cam_id, time_seg, index = stem.split('_')

        # img_file = os.path.join(img_dir, stem + '.png')
        # img = cv2.imread(img_file)
        # if img is None:
        #     continue
        # h, w = img.shape[:2]

        # with open(label_file) as f:
        #     for line in f:
        #         cls, xc, yc, bw, bh = map(float, line.strip().split())
        #         x1 = (xc - bw/2) * w
        #         y1 = (yc - bh/2) * h
        #         x2 = (xc + bw/2) * w
        #         y2 = (yc + bh/2) * h
        #         group_objects[(cam_id, time_seg)].append(
        #             (img_file, int(cls), [x1, y1, x2, y2], index)
        #         )

        group_objects[(cam_id, time_seg)].append([stem, int(index)])

    # convert keys to strings to dump to JSON
    json_compatible = {f"{k[0]}_{k[1]}": v for k, v in group_objects.items()}
    with open(out_json, 'w') as f:
        json.dump(json_compatible, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build object index from labels and images.")
    parser.add_argument("--img_dir", default="/media/hungdv/Source/Data/AICity2025/yolo_format/train_data/images", help="Path to the directory containing images.")
    parser.add_argument("--label_dir",default="/media/hungdv/Source/Data/AICity2025/yolo_format/train_data/labels", help="Path to the directory containing label files.")
    parser.add_argument("--out_json",default="object_index.json", help="Path to the output JSON file.")
    args = parser.parse_args()

    main(args.img_dir, args.label_dir, args.out_json)

