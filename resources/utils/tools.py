import os
import torch
import numpy as np

from PIL import Image
from torchvision import transforms

from captum.attr import visualization as viz
from captum.attr import LayerGradCam, GuidedGradCam

####----------------------------- get_img ---------------
def get_ts_img(root, fname, img_path, batch=True, IMG_SIZE = (600, 450)):
    img_path = os.path.join(root,  img_path, fname + '.jpg')
    image = Image.open(img_path).convert('RGB')

    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_img_ = val_transform(image)
    transform_img = transform_img_.unsqueeze(0) if batch else transform_img_
    
    np_img = np.array(image, dtype=np.float32) / 255.0
    
    return transform_img, np_img


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

####----------------------------- plot attribution ---------------
def show_captum_attributions(image_tensor, orig_image, label_map, device, true_label=None):
    """
    image_tensor: [1, C, H, W] normalized tensor for the model
    orig_image: denormalized HWC numpy array in [0,1] for display
    """
    gradcam_attr, guided_attr, pred_class = compute_attributions(model, image_tensor, device)

    # Grad-CAM: single-channel heatmap
    gradcam_np = gradcam_attr.squeeze().cpu().detach().numpy()

    # Guided Grad-CAM: multi-channel, same shape as input image
    guided_np = guided_attr.squeeze().cpu().detach().permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(orig_image)
    axes[0].set_title(f"{label_map[true_label]},  Pred: {label_map[pred_class]}")
    axes[0].axis('off')

    axes[1].imshow(orig_image)
    axes[1].imshow(gradcam_np, cmap='jet', alpha=0.5)
    axes[1].set_title('Grad-CAM')
    axes[1].axis('off')

    # Guided Grad-CAM uses Captum's built-in visualization (handles normalization/clipping)
    viz.visualize_image_attr(
        guided_np, orig_image, method='heat_map', sign='absolute_value',
        show_colorbar=True, title='Guided Grad-CAM', plt_fig_axis=(fig, axes[2]), use_pyplot=False
    )

    plt.tight_layout()
    plt.show()