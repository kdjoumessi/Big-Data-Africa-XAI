import os
import torch
import numpy as np

from PIL import Image
from torchvision import transforms

from captum.attr import visualization as viz
from captum.attr import LayerGradCam, GuidedGradCam

####----------------------------- get_img ---------------
def get_ts_img(fname, img_path, IMG_SIZE = (600, 450)):
    img_path = os.path.join(root,  img_path, fname + '.jpg')
    image = Image.open(img_path).convert('RGB')

    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_img = val_transform(image)
    
    np_img = np.array(image, dtype=np.float32) / 255.0
    
    return transform_img.unsqueeze(0), np_img


####----------------------------- Generate attributions for a single image ---------------
def compute_attributions(model, image_tensor, device, target_class=None):
    """
    image_tensor: [1, C, H, W], already normalized, on the correct device
    """
    model.to(device)
    model.eval()

    # target the last conv block, same choice as before
    target_layer = model.layer4[-1]
    
    gradcam = LayerGradCam(model, target_layer)
    guided_gradcam = GuidedGradCam(model, target_layer)
    
    input_tensor = image_tensor.clone().to(device).requires_grad_()

    if target_class is None:
        with torch.no_grad():
            output = model(input_tensor)
            target_class = output.argmax(dim=1).item()

    # Grad-CAM: coarse, low-res localization map (size of target_layer's feature map)
    gradcam_attr = gradcam.attribute(input_tensor, target=target_class)

    # upsample Grad-CAM to input resolution for visualization
    gradcam_attr_upsampled = LayerGradCam.interpolate(
        gradcam_attr, image_tensor.shape[2:]
    )

    # Guided Grad-CAM: pixel-level, high-res attribution
    guided_attr = guided_gradcam.attribute(input_tensor, target=target_class)

    return gradcam_attr_upsampled, guided_attr, target_class

####----------------------------- Denormalization ---------------
def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean