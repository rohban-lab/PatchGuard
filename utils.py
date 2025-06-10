import sys
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from rich.console import Console
from rich.table import Table

from dataset import *


def patchify(x, patch_size):
    if len(x.shape) == 3:  # If single-channel image, add channel dimension
        x = x.unsqueeze(1)

    bs, c, h, w = x.shape

    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    x = unfold(x)  # Shape: (B, C * patch_size * patch_size, num_patches)

    num_patches = (h // patch_size) * (w // patch_size)
    x = x.view(bs, c, patch_size, patch_size, num_patches).permute(0, 4, 1, 2, 3)
    return x

def label_patch(x):
    labels = torch.any(x > 0, dim=(2, 3, 4)).float()
    return labels

def get_dataloader(image_size, path, dataset_name, class_name, batch_size, test_batch_size, num_workers, seed):
    transform = transforms.Compose([transforms.Resize((image_size, image_size), Image.LANCZOS), transforms.ToTensor()])
    mask_transform = transforms.Compose([transforms.Resize((image_size, image_size), Image.LANCZOS),transforms.ToTensor()])
    
    if dataset_name == 'mvtec':
        train_set = MVTec(path, class_name, transform=transform, mask_transform=mask_transform, seed=seed, split='train', size=image_size)
        test_set = MVTec(path, class_name, transform=transform, mask_transform=mask_transform, seed=seed, split='test', size=image_size)
    elif dataset_name == 'visa':
        train_set = VisA(path, class_name, transform=transform, mask_transform=mask_transform, seed=seed, split='train', size=image_size)
        test_set = VisA(path, class_name, transform=transform, mask_transform=mask_transform, seed=seed, split='test', size=image_size)
    elif dataset_name == 'mpdd':
        train_set = MPDD(path, class_name, transform=transform, mask_transform=mask_transform, seed=seed, split='train', size=image_size)
        test_set = MPDD(path, class_name, transform=transform, mask_transform=mask_transform, seed=seed, split='test', size=image_size)
    elif dataset_name == 'btad':
        train_set = BTAD(path, class_name, transform=transform, mask_transform=mask_transform, seed=seed, split='train', size=image_size)
        test_set = BTAD(path, class_name, transform=transform, mask_transform=mask_transform, seed=seed, split='test', size=image_size)
    elif dataset_name == "dtd":
        train_set = DTD(path, class_name, transform=transform, mask_transform=mask_transform, seed=seed, split='train', size=image_size)
        test_set = DTD(path, class_name, transform=transform, mask_transform=mask_transform, seed=seed, split='test', size=image_size)
    elif dataset_name == "brats2021":
        train_set = BraTS2021(path, transform=transform, mask_transform=mask_transform, seed=seed, split='train', size=image_size)
        test_set = BraTS2021(path, transform=transform, mask_transform=mask_transform, seed=seed, split='test', size=image_size)
    elif dataset_name == "headct":
        train_set = HeadCT(path, transform=transform, mask_transform=mask_transform, seed=seed, split='train', size=image_size)
        test_set = HeadCT(path, transform=transform, mask_transform=mask_transform, seed=seed, split='test', size=image_size)
    elif dataset_name == 'wfdd':
        train_set = WFDD(path, class_name, transform=transform, mask_transform=mask_transform, seed=seed, split='train', size=image_size)
        test_set = WFDD(path, class_name, transform=transform, mask_transform=mask_transform, seed=seed, split='test', size=image_size)

    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    print(f"Dataloaders for dataset {dataset_name} and class {class_name} have been prepared.")

    return train_loader, test_loader

def get_auc(test_scores, test_labels, test_masks, patches_per_side, sigma, radius, k):
    scores = torch.cat(test_scores, dim=0)

    topk_values, _ = torch.topk(scores, k, dim=1)
    pred_labels = torch.mean(topk_values, dim=1)
    image_labels = torch.cat(test_labels, dim=0)

    image_auroc = roc_auc_score(image_labels.view(-1).cpu().numpy(), pred_labels.view(-1).cpu().numpy())

    masks = torch.cat(test_masks, dim=0)
    patch_scores = scores.reshape(-1, patches_per_side, patches_per_side)
    pixel_scores = F.interpolate(patch_scores.unsqueeze(1), size=(masks.shape[-1], masks.shape[-1]), mode='bilinear', align_corners=False)
    localization = gaussian_filter(pixel_scores.squeeze(1).cpu().detach().numpy(), sigma=sigma, radius=radius, axes=(1,2))

    pixel_auroc = roc_auc_score(masks.view(-1).cpu().numpy(), localization.reshape(-1))

    return image_auroc, pixel_auroc


def save_model(model, filepath="./model.pth"):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model, filepath="./model.pth"):
    try:
        model.load_state_dict(torch.load(filepath))
        print(f"Model loaded from {filepath}")
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)

def log_loss(epoch, loss, filepath="./loss_log.txt"):
    with open(filepath, "a") as f:
        f.write(f"Epoch {epoch} : {loss}\n")

def display_results(metrics_dict, description):
    console = Console()
    table = Table(title=f"{description}")

    table.add_column("Metric", style="cyan", justify="center")
    table.add_column("Value", style="magenta", justify="center")

    for metric, value in metrics_dict.items():
        table.add_row(f"{metric}", f"{value:.4f}")

    console.print(table)

