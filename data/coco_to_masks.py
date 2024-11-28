from pycocotools.coco import COCO
import numpy as np
import cv2
import os

def create_instance_masks(coco_json_path, image_dir, mask_output_dir):
    coco = COCO(coco_json_path)
    if not os.path.exists(mask_output_dir):
        os.makedirs(mask_output_dir)

    for image_info in coco.imgs.values():
        image_id = image_info['id']
        image_file_name = image_info['file_name']
        mask_path = os.path.join(mask_output_dir, os.path.splitext(image_file_name)[0] + '.png')
        image_height = image_info['height']
        image_width = image_info['width']

        mask = np.zeros((image_height, image_width), dtype=np.uint8)

        # Get annotations for the image
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
        for i, ann in enumerate(annotations, start=1):
            if 'segmentation' in ann:
                # Create binary mask for each instance
                instance_mask = coco.annToMask(ann)
                mask[instance_mask > 0] = i  # Assign a unique label for each instance

        cv2.imwrite(mask_path, mask)
    print("Masks created successfully.")

# Example Usage
coco_annotation_file = "data/BUILDINGS/test/_annotations.coco.json"
image_directory = "data/BUILDINGS/test"
mask_output_directory = "data/BUILDINGS/masks/test"
create_instance_masks(coco_annotation_file, image_directory, mask_output_directory)