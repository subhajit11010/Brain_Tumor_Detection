import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
def generate_gradcam_heatmap(model, image, device, target_class=None):
 
    # Define the target layer (last convolutional layer in VGG16)
    target_layer = model.features[29]  # VGG16's last convolutional layer

    # Hook to get the feature maps and gradients
    def save_feature_maps(module, input, output):
        feature_maps.append(output)

    def save_grads(module, input, output):
        gradients.append(output[0])

    # Register hooks for feature maps and gradients
    feature_maps = []
    gradients = []
    target_layer.register_forward_hook(save_feature_maps)
    target_layer.register_backward_hook(save_grads)

    # Forward pass: get the model's output
    image = image.unsqueeze(0)  # Add batch dimension
    output = model(image)

    if target_class is None:
        target_class = torch.argmax(output, dim=1)  # Get the predicted class
    if target_class.item() == 2:
      return False
    # print(target_class.item())
    # Zero the gradients of the model's parameters
    model.zero_grad()

    # Backward pass: compute gradients w.r.t the target class
    target_class_score = output[0, target_class]
    target_class_score.backward()

    # Get the feature map and gradients
    feature_map = feature_maps[0]
    gradient = gradients[0]

    # Compute the weights
    weights = torch.mean(gradient, dim=[2, 3], keepdim=True)  # Global average pooling
    weighted_feature_map = weights * feature_map  # Weighting the feature map

    # Sum along the channels to get a single heatmap
    heatmap = torch.sum(weighted_feature_map, dim=1, keepdim=False)

    # Apply ReLU to the heatmap
    heatmap = F.relu(heatmap)

    # Normalize the heatmap to [0, 1]
    heatmap = heatmap - torch.min(heatmap)
    heatmap = heatmap / torch.max(heatmap)

    # Resize the heatmap to match the input image size
    heatmap = heatmap.squeeze().cpu().detach().numpy()
    heatmap = np.uint8(255 * heatmap)  # Convert to 0-255 scale
    heatmap = Image.fromarray(heatmap).resize((224, 224))

    return heatmap
def overlay_heatmap(image_path, model, device):
    # Open the image and convert it to RGB
    image = Image.open(image_path).convert('RGB')

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).to(device)

    # Generate Grad-CAM heatmap
    heatmap = generate_gradcam_heatmap(model, image_tensor, device = device)
    if(heatmap != False):
      # Convert original image to numpy array
      image = np.array(image)

      # Resize the heatmap to the image size
      heatmap_resized = np.array(heatmap.resize(image.shape[1::-1]))

      # Normalize the heatmap to the same scale as the original image
      heatmap_resized = np.uint8(255 * (heatmap_resized / np.max(heatmap_resized)))

      # Apply the heatmap (overlay) on the original image
      overlay = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
      overlay = np.float32(overlay) / 255
      result = np.float32(image) / 255
      result = result + 0.4 * overlay  # Adjust the intensity of the heatmap

      # Clip the result to avoid values outside [0, 1]
      result = np.clip(result, 0, 1)
      return result
      # Show the result
      # plt.imshow(result)
      # plt.axis('off')
      # plt.show()
    else:
      return image

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def overlay_heatmap1(image_path, model, device):
    # Open the image and convert it to RGB
    image = Image.open(image_path).convert('RGB')

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).to(device)

    # Generate Grad-CAM heatmap
    heatmap = generate_gradcam_heatmap(model, image_tensor, device = device)
    if(heatmap != False):
    # Convert original image to numpy array
      image = np.array(image)

      # Resize the heatmap to the image size
      heatmap_resized = np.array(heatmap.resize(image.shape[1::-1]))

      # Normalize the heatmap to the same scale as the original image
      heatmap_resized = np.uint8(255 * (heatmap_resized / np.max(heatmap_resized)))

      # Threshold the heatmap to create a binary mask for detected areas
      _, binary_mask = cv2.threshold(heatmap_resized, 127, 255, cv2.THRESH_BINARY)

      # Apply Gaussian Blur to create a soft fading effect on the binary mask
      blurred_mask = cv2.GaussianBlur(binary_mask, (21, 21), 0)

      # Normalize the blurred mask to range [0, 1] (for blending)
      blurred_mask = blurred_mask / np.max(blurred_mask)

      # Create a light blue background (RGB values)
      light_blue = np.array([173, 0, 0], dtype=np.uint8)  # Light blue color in RGB
      blue_background = np.full_like(image, light_blue)

      # Create a white overlay for the detected area with fading effect
      fading_overlay = np.zeros_like(image, dtype=np.uint8)
      for i in range(3):  # For each color channel (RGB)
          fading_overlay[:,:,i] = (blurred_mask * 255).astype(np.uint8)

      # Combine the light blue background with the fading white overlay
      result_image = cv2.addWeighted(blue_background, 0.8, fading_overlay, 0.9, 0)

      # Combine with original image to preserve the brain structure while applying the blue tint
      result_image = cv2.addWeighted(result_image, 0.8, image, 0.8, 0)
      result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
      return result_image
      # Display the result image
      plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
      plt.axis('off')
      plt.show()

    else:
      return image


# overlay_heatmap1(image_path, model, device)
# overlay_heatmap(image_path, model, device)
