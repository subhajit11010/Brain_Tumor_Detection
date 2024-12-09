from saliency import *
from visualizeer import *
def processor(model, image_path, device):
    model.load_state_dict(torch.load('./model.pth', map_location=device, weights_only=True))
    model.eval()
    s_image = generate_saliency(model, image_path,device)
    g_image= overlay_heatmap(image_path, model, device)
    b_image = overlay_heatmap1(image_path, model, device)
    return s_image, b_image, g_image