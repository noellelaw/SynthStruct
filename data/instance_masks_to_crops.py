import os
import cv2
import numpy as np
from tqdm import tqdm

def process_instance_segmentation_dataset(image_dir, mask_dir, output_dir):
    """
    Process an instance segmentation dataset to crop individual instances and save them.

    Parameters:
    - image_dir: Directory containing original images.
    - mask_dir: Directory containing instance masks (with unique labels for each instance).
    - output_dir: Directory to save cropped instances.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in tqdm(image_files, desc="Processing Images"):
        # Load the image and mask
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, os.path.splitext(image_file)[0] + '.png')
        
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        if mask is None or image is None:
            print(f"Skipping {image_file}: Missing image or mask.")
            continue
        
        # Get unique instance IDs from the mask (excluding background, assumed to be 0)
        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids != 0]
        
        for i, instance_id in enumerate(instance_ids):
            # Create a binary mask for the current instance
            instance_mask = (mask == instance_id).astype(np.uint8)
            
            # Find bounding box of the instance
            y_indices, x_indices = np.where(instance_mask > 0)
            if len(x_indices) == 0 or len(y_indices) == 0:
                continue
            
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            
            # Crop the image and apply the mask
            cropped_image = image[y_min:y_max+1, x_min:x_max+1]
            cropped_mask = instance_mask[y_min:y_max+1, x_min:x_max+1]
            cropped_instance = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_mask)
            
            # Save the cropped instance
            output_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_{i+1}.png")
            cv2.imwrite(output_path, cropped_instance)
    
    print(f"Processing complete. Cropped images saved in '{output_dir}'.")


image_directory = "data/BUILDINGS/valid"
mask_directory = "data/BUILDINGS/masks/valid"
output_directory = "data/BUILDINGS/crops/valid"

process_instance_segmentation_dataset(image_directory, mask_directory, output_directory)
