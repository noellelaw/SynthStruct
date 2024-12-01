import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse

def load_model():
    """
    Load a pre-trained deep learning segmentation model (e.g., DeepLabV3).
    """
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()
    return model

def preprocess_image(image_path):
    """
    Preprocess the image for segmentation: resize and normalize.
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or invalid format!")
    return transform(image), image

def segment_image(model, image_tensor):
    """
    Perform semantic segmentation using the pre-trained model.
    """
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))  # Add batch dimension
        output_predictions = output['out'].squeeze(0).argmax(0).cpu().numpy()
    return output_predictions

def classify_materials(image, segmented_mask, material_labels):
    """
    Classify materials based on texture or color analysis.
    Map segmentation labels to material types.
    """
    materials_detected = {}
    for material_label, material_name in material_labels.items():
        mask = (segmented_mask == material_label).astype(np.uint8) * 255
        # Resize the mask to match the image dimensions
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))  # Width x Height

        # Ensure the resized mask is binary and of type uint8
        mask_resized = (mask_resized > 0).astype(np.uint8) * 255
        
        if np.sum(mask) > 0:  # If material is present
            # Check dimensions of the mask and image
            print(f"Image shape: {image.shape}")
            print(f"Mask shape: {mask_resized.shape}")
            cropped_region = cv2.bitwise_and(image, image, mask=mask_resized)
            materials_detected[material_name] = cropped_region
    return materials_detected

def display_results(original_image, segmented_mask, material_results):
    """
    Visualize the segmentation results and classified materials.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title("Segmented Mask")
    plt.imshow(segmented_mask, cmap='jet')

    plt.show()

    # Display materials detected
    for material_name, material_image in material_results.items():
        plt.figure()
        plt.title(f"Material: {material_name}")
        plt.imshow(cv2.cvtColor(material_image, cv2.COLOR_BGR2RGB))
        plt.show()

def main(image_path):
    # Load pre-trained model
    model = load_model()

    # Preprocess input image
    image_tensor, original_image = preprocess_image(image_path)

    # Perform segmentation
    segmented_mask = segment_image(model, image_tensor)

    # Define material labels (customize based on fine-tuned model)
    material_labels = {
        0: "Other/Unclear",
        1: "Concrete",
        2: "Bricks",
        3: "Corrugated Metal Sheet",
        4: "Steel",
        5: "Glass",
        6: "Wood",
        7: "Thatch/Grass"
    }

    # Classify materials in the segmented regions
    material_results = classify_materials(original_image, segmented_mask, material_labels)

    # Display results
    display_results(original_image, segmented_mask, material_results)

if __name__ == "__main__":
    # Example usage: python3 deeplabv3.py /path/to/image
    parser = argparse.ArgumentParser(description="Segment and classify building materials from an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    args = parser.parse_args()

    main(args.image_path)
