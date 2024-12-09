import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import cv2
def generate_saliency(model, image_path, device):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad = True  # Enable gradient computation on input
    output = model(input_tensor)

    # Select the class with the highest score
    target_class = torch.argmax(output, dim=1).item()

    # Backpropagate the gradients of the target class
    model.zero_grad()
    target_score = output[0, target_class]
    target_score.backward()

    # Extract gradients of the input tensor
    gradients = input_tensor.grad.data.cpu().numpy()[0]  # Shape: (C, H, W)

    # Convert to grayscale saliency map (absolute max gradient across channels)
    saliency_map = np.max(np.abs(gradients), axis=0)

    # Normalize the saliency map to [0, 1]
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
    return display_saliency_map(image_path, saliency_map)

def display_saliency_map(image_path, saliency_map):
    # Open the original image and resize it to match the saliency map's dimensions
    original_image = Image.open(image_path).convert('RGB')
    original_image = original_image.resize(saliency_map.shape[::-1][0:2])  # Only take width, height (224, 224)
    original_image = np.array(original_image)

    # Normalize the saliency map to the range [0, 255]
    saliency_map_uint8 = (saliency_map * 255).astype(np.uint8)

    # Create a blue-tinted background image
    blue_image = np.zeros_like(original_image)
    blue_image[:, :, 2] = 255  # Set the blue channel to maximum

    # Apply the saliency map as a mask (this will give the saliency map in reddish tones)
    saliency_colored = cv2.applyColorMap(saliency_map_uint8, cv2.COLORMAP_HOT)  # Saliency in reddish tones
    saliency_colored = cv2.cvtColor(saliency_colored, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Normalize saliency map to create a blending mask
    saliency_mask = cv2.normalize(saliency_map, None, 0, 1, cv2.NORM_MINMAX)  # Normalize mask to [0, 1]

    # Reshape the saliency mask to have a 3rd dimension (224, 224, 1)
    saliency_mask = np.expand_dims(saliency_mask, axis=-1)

    # Now, we need to broadcast the saliency mask to have 3 channels for blending with images
    saliency_mask = np.repeat(saliency_mask, 3, axis=-1)  # Convert to shape (224, 224, 3)

    # Create a blended background by combining the blue image and the original image
    blended_background = cv2.addWeighted(blue_image, 0.3, original_image, 0.7, 0)

    # Overlay the saliency map on the background
    result_image = blended_background * (1 - saliency_mask) + saliency_colored * saliency_mask
    result_image = result_image.astype(np.uint8)  # Ensure proper format for visualization

    return result_image
