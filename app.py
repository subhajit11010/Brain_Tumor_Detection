import torch
import os
import torch.nn as nn
import torchvision
from torchvision import models, transforms
import sklearn
import cv2
import shutil
from training import *
from test import *
from predictor import *
from visualizeer import *
from saliency import *
from metrics import *
# zip_file_path = 'archive.zip'
# extraction_path = 'datasets'

# shutil.unpack_archive(zip_file_path, extraction_path, 'zip')
# print(f"Dataset extracted to {extraction_path}")

from torchvision import transforms

# Transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (for VGG16 input size)
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.ToTensor(),  # Convert to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.ToTensor(),  # Convert to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# from torchvision import datasets

# Dataset paths
# train_dir = 'img_dataset/train'
# val_dir = 'img_dataset/val'
# test_dir = 'img_dataset/test'
# def is_valid_file(x):
#     return not x.endswith('.ipynb_checkpoints')
# # Create datasets
# train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform, is_valid_file=is_valid_file)
# val_dataset = datasets.ImageFolder(root=val_dir, transform=val_test_transform, is_valid_file=is_valid_file)
# test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transform, is_valid_file=is_valid_file)

# Check class mappings
# print(f"Class mapping: {train_dataset.class_to_idx}")

# from torch.utils.data import DataLoader

# DataLoader parameters
batch_size = 32
num_workers = 4  # Number of subprocesses for data loading

# DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

model = models.vgg16(weights = 'VGG16_Weights.DEFAULT')
model.classifier[6] = nn.Linear(4096,4)
# print(model)

import torch.optim as optim
learning_rate  = 0.0001
epochs =10
batch_size  = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# training loop
# train(model, learning_rate, epochs,batch_size, device, train_loader, val_loader)
# testing loop
# test(model, test_loader, device)
# predict
  
image_path = "./img18.jpg"
prediction, outputs = predict(image_path, model, device)
# print(outputs, outputs.numpy())
# print(prediction)
# stats(outputs)
# overlay_heatmap(image_path, model, device)
# overlay_heatmap1(image_path, model, device)

image = generate_saliency(model, image_path, device)
plt.imshow(image)
plt.show()
# import cv2
# import numpy as np
# from PIL import Image
# import torch
# from torchvision import transforms

# # Load the image
# image_path = "img7.png"  # Update to your image path
# image = Image.open(image_path).convert('RGB')

# # Preprocess the image
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# image_tensor = transform(image).to(device)  # Add batch dimension

# # Generate heatmap
# heatmap = generate_gradcam_heatmap(model, image_tensor)  # Assuming this generates a 2D heatmap
# heatmap = np.array(heatmap)

# # Smooth the heatmap using Gaussian blur
# heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)

# # Normalize the heatmap values to [0, 1]
# heatmap = cv2.normalize(heatmap, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

# # Apply a threshold to create a binary mask
# threshold_value = 0.5
# binary_mask = heatmap > threshold_value

# # Resize binary mask to match the original image dimensions
# binary_mask_resized = cv2.resize(binary_mask.astype(np.uint8), (image.width, image.height))

# # Refine the binary mask using morphological operations
# kernel = np.ones((5, 5), np.uint8)
# binary_mask_refined = cv2.morphologyEx(binary_mask_resized, cv2.MORPH_CLOSE, kernel)

# # Create a color mask for the output
# color_mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)
# color_mask[:, :] = [139, 0, 0]  # Deep blue for the background

# # Draw overlapping circles within the detected region
# circle_radius_range = (3, 7)  # Range of circle radii
# for y in range(0, image.height, 2):
#     for x in range(0, image.width, 2):
#         if binary_mask_refined[y, x] > 0:
#             radius = np.random.randint(circle_radius_range[0], circle_radius_range[1])
#             cv2.circle(color_mask, (x, y), radius, (255, 255, 255), -1)

# # Convert original PIL image to OpenCV format
# original_image = np.array(image)
# original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

# # Combine the original image with the mask
# result_image = cv2.addWeighted(original_image, 0.7, color_mask, 0.5, 0)  # Adjust weights for density

# # Display the result
# cv2.imshow(result_image)

# print(predict('./img9.jpg', model, device))

