import torch
import random
import numpy as np
from tqdm import tqdm
import torch.optim as optim

from patchguard import PatchGuard
from utils import get_dataloader, save_model, log_loss, patchify, label_patch
from loss import Loss
from attack import pgd_attack
from pseudo_anomaly import AnomalyGenerator

import matplotlib.pyplot as plt
import os 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_step(model, anomaly_generator, train_loader, optimizer, criterion, use_reg, device, args):
    total_sample = 0
    total_loss = 0

    batch_iterator = tqdm(train_loader, disable=not args.use_tqdm, desc="Training Batches")
    for batch in batch_iterator:
        loss = 0

        normal_data = [batch[0].to(device)]

        if args.adv_train:
            images = batch[0].to(device).clone()
            adv_normal_images = pgd_attack(model, images, torch.zeros(images.shape[0], model.num_patches).to(device), args.epsilon_train, args.step_train)
            normal_data.append(adv_normal_images)

        for imgs in normal_data:
            features, attn_weights = model.feature_extractor(imgs, use_reg)

            scores_true = model.discriminator(features)

            masks_true = torch.zeros(features.shape[0], features.shape[1]).to(device)
            loss += criterion(scores_true, masks_true, attn_weights)

        images = batch[0].clone()
        foreground_masks = batch[3]

        augmented_images, augmented_masks = anomaly_generator(images, foreground_masks)
        augmented_masks = label_patch(patchify(augmented_masks, model.patch_size))
        augmented_images, augmented_masks = augmented_images.to(device), augmented_masks.to(device)

        anomaly_data = [augmented_images]
        if args.adv_train:
            adv_distorted_images = pgd_attack(model, augmented_images.clone(), augmented_masks, args.epsilon_train, args.step_train)
            anomaly_data.append(adv_distorted_images)

        for imgs in anomaly_data:
            features, attn_weights = model.feature_extractor(imgs, use_reg)
            scores_aug = model.discriminator(features)

            loss += criterion(scores_aug, augmented_masks, attn_weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_sample += images.shape[0]
        total_loss += loss.item() * images.shape[0]
    
    return total_loss / total_sample


def train(model, anomaly_generator, train_loader, optimizer, lr_scheduler, criterion, use_reg, epochs, device, ckpt_path, args):
    model.train()
    
    epoch_iterator = tqdm(range(epochs), disable=not args.use_tqdm, desc="Epochs")
    for epoch in epoch_iterator:
        total_loss = train_step(model, anomaly_generator, train_loader, optimizer, criterion, use_reg, device, args)
        lr_scheduler.step()

        epoch_iterator.set_postfix(loss=total_loss)
        log_loss(epoch, total_loss)

        if (epoch % 20 == 0) and (epoch > int(epochs / 2)):
            save_model(model, ckpt_path + f"/patchguard_epoch_{epoch}.pth")


def run_train(args):
    set_seed(args.seed)

    ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoints_{args.dataset}_{args.class_name}_pgd{args.step_train}")
    os.makedirs(ckpt_path, exist_ok=True)


    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    model = PatchGuard(args, device).to(device)

    train_loader, _ = get_dataloader(args.image_size, args.dataset_dir, args.dataset, args.class_name, args.train_batch_size, args.test_batch_size, args.num_workers, args.seed)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * args.lr_decay_factor)
    
    criterion = Loss(args.reg_type, device, model.num_patches)

    anomaly_generator = AnomalyGenerator(args.dataset, args.class_name, args.seed)  

    train(model, anomaly_generator, train_loader, optimizer, lr_scheduler, criterion, args.use_reg, args.epochs, device, ckpt_path, args)

    save_model(model, args.checkpoint_dir+f"patchguard_{args.dataset}_{args.class_name}_pgd{args.step_train}_last_epoch.pth")
