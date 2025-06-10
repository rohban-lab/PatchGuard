import os
import sys
import cv2
import torch
import shutil
import numpy as np
from pathlib import Path
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from attack import pgd_attack
from utils import get_dataloader, label_patch, patchify, load_model
from patchguard import PatchGuard

def image_transform(image):
     return np.clip(image* 255, 0, 255).astype(np.uint8)
    
def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam) 

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)
    
def get_heatmap(raw_image, localization):
    ano_map = gaussian_filter(localization, sigma=4)
    ano_map = min_max_norm(ano_map)
    ano_map = cvt2heatmap(ano_map * 255.0)
    raw_image = image_transform(raw_image.detach().cpu().numpy())
    image_cv2 = np.uint8(np.transpose(raw_image,(1,2,0)))
    ano_map = show_cam_on_image(image_cv2[..., ::-1], ano_map)
    ano_map = ano_map[..., ::-1]
    return ano_map

def transparent_cmap(cmap, N=255):
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap

def visualize_heatmap(args):
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    _, test_loader = get_dataloader(args.image_size, args.dataset_dir, args.dataset, args.class_name, args.train_batch_size, args.test_batch_size, args.num_workers, args.seed)

    model = PatchGuard(args, device)
    load_model(model, args.checkpoint_dir+f"patchguard_{args.dataset}_{args.class_name}.pth")    

    Path(f"./plots").mkdir(exist_ok=True, parents=True)
    plot_path = Path(f"./plots")

    Path(f"{plot_path}/clean_image").mkdir(exist_ok=True, parents=True)
    clean_image_path = Path(f"{plot_path}/clean_image")

    Path(f"{plot_path}/adv_image").mkdir(exist_ok=True, parents=True)
    adv_image_path = Path(f"{plot_path}/adv_image")
    
    Path(f"{plot_path}/clean_heatmap").mkdir(exist_ok=True, parents=True)
    clean_heatmap_path = Path(f"{plot_path}/clean_heatmap")
    
    Path(f"{plot_path}/adv_heatmap").mkdir(exist_ok=True, parents=True)
    adv_heatmap_path = Path(f"{plot_path}/adv_heatmap")

    Path(f"{plot_path}/mask").mkdir(exist_ok=True, parents=True)
    mask_path = Path(f"{plot_path}/mask")
    
    cmap = transparent_cmap(plt.cm.jet)
    
    with torch.no_grad():
        i = 0
        for images, _, masks, _ in test_loader:
            images, masks = images.to(device), masks.to(device)
            
            for mode in ["clean", "adv"]:
                image_path = clean_image_path if mode == "clean" else adv_image_path
                heatmap_path = clean_heatmap_path if mode == "clean" else adv_heatmap_path

                if mode == "adv":
                    with torch.set_grad_enabled(True):
                        images = pgd_attack(model, images, label_patch(patchify(masks, model.patch_size)), args.epsilon_visualization, args.step_visualization)
                        
                scores = model(images)
    
                batch_size, num_patches = scores.shape
                image_size = images.shape[-1]
                patches_per_side = int(np.sqrt(num_patches))

                j = i
                for b in range(batch_size):
                    patch_scores = scores[b].reshape((patches_per_side, patches_per_side))
                    scores_interpolated = F.interpolate(patch_scores.unsqueeze(0).unsqueeze(0),
                                                        size=image_size,
                                                        mode='bilinear',
                                                        align_corners=False
                                                        ).squeeze(0).squeeze(0)
    
                    localization = gaussian_filter(scores_interpolated.cpu().detach().numpy(), sigma=args.smoothing_sigma, radius=args.smoothing_radius)  
                    localization = gaussian_filter(localization, sigma=4)
                    localization = min_max_norm(localization)

                    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                    ax.imshow(images[b].cpu().detach().permute(1, 2, 0).numpy())
                    ax.axis('off')
                    plt.savefig(os.path.join(image_path, f'img{j}.png'),  bbox_inches='tight', pad_inches=0, format='png')
                    plt.close(fig)
                    
                    # Save the mask
                    if mode == "clean":
                        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                        modified_image = images[b].clone()
                        modified_image[0, masks[b] > 0] = 1.0
                        modified_image[1, masks[b] > 0] = 0.0
                        modified_image[2, masks[b] > 0] = 0.0
    
                        ax.imshow(modified_image.cpu().detach().permute(1, 2, 0).numpy())
                        ax.axis('off')
                        plt.savefig(os.path.join(mask_path, f'img{j}.png'), bbox_inches='tight', pad_inches=0, format='png')
                        plt.close(fig)
                    
                    # Save the heatmap
                    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                    ax.imshow(images[b].cpu().detach().permute(1, 2, 0).numpy())
                    ax.imshow(localization, cmap=cmap, interpolation='bilinear')
                    ax.axis('off')
                    plt.savefig(os.path.join(heatmap_path, f'img{j}.png'), bbox_inches='tight', pad_inches=0, format='png')
                    plt.close(fig)
    
                    j = j + 1

            i += args.test_batch_size
                
        print("Visualization complete.")
        shutil.make_archive(f'visualization', 'zip', plot_path)
        return