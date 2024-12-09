from torchvision import transforms
import torch
from PIL import Image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.ToTensor(),  # Convert to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])
def predict(image_path,model, device):
  image = Image.open(image_path).convert("RGB")
  image = transform(image).unsqueeze(0).to(device)
  model.load_state_dict(torch.load('./model.pth', map_location=device, weights_only=True))
  model.eval()
  with torch.no_grad():
    output = model(image)
    return_result(output)
  _ , predicted = torch.max(output,1)
  if predicted.item() == 0:
    return 'glioma', output
  elif predicted.item() == 1:
    return 'meningioma' ,output
  elif predicted.item() == 2:
    return 'no tumor',output
  else:
    return 'pituitary',output
  
def return_result(outputs):
  return outputs
