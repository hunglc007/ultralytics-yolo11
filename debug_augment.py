from ultralytics import YOLO
from ultralytics.data import YOLODataset, YOLOMultiModalDataset
from ultralytics.data.augment import Mosaic, Replace, Compose, RandomFlip
import cv2

dataset = YOLODataset(cache=True, img_path='/mnt/e/Data/AICity2025/yolo_format/images/train', data={'channels': 3, 'names': {0: 'Bus', 1: 'Bike', 2: 'Car', 3: 'Pedestrian', 4: 'Truck'}, 'nc': 5,
                                                                                        'path': '/media/hungdv/Source/Data/AICity2025/yolo_format', 'train': '/mnt/e/Data/AICity2025/yolo_format/images/train',
                                                                                        'val': '/mnt/e/Data/AICity2025/yolo_format/images/val', 'yaml_file': '/media/hungdv/Source/Data/AICity2025/yolo_format/data.yaml'})


labels = dataset.get_image_and_label(7)
original_img = labels['img']
# print(labels)

# augmented_labels = dataset.__getitem__(7)
# augmented_img = augmented_labels['img'].to('cpu').numpy().transpose(1, 2, 0)  # Convert to HWC format


replace_aug = Replace(dataset, imgsz=1280, p=0.5)
pre_transform = Compose([
    replace_aug,
    RandomFlip(direction="vertical", p=0.25),
    RandomFlip(direction="horizontal", p=0.25, flip_idx=[])
])
# augmented_labels = replace_aug(labels)
# augmented_img = augmented_labels['img']

mosaic_aug = Mosaic(dataset, imgsz=640, p=1, n=4, pre_transform=pre_transform)

final_transform = Compose([pre_transform, mosaic_aug])

augmented_labels = final_transform(labels)
augmented_img = augmented_labels['img']
bboxes = augmented_labels['instances'].bboxes
# Draw bounding boxes on the augmented image
for bbox in bboxes:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(augmented_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)


# Resize augmented image to match the original image size
augmented_img = cv2.resize(augmented_img, (original_img.shape[1], original_img.shape[0]))
# Concatenate original and augmented images for comparison
concatenated_img = cv2.hconcat([original_img, augmented_img])
cv2.imwrite('augmented_image.png', concatenated_img)


# model = YOLO("yolo11n.pt")
# results = model.train(data="/media/hungdv/Source/Data/AICity2025/yolo_format/data.yaml", epochs=3)

# copypaste = CopyPaste(dataset, p=0.5)
# augmented_labels = copypaste(original_labels)
