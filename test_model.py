import torch

from patchguard import PatchGuard
from utils import get_dataloader, load_model, patchify, label_patch, get_auc, display_results
from attack import pgd_attack


def test(model, test_loader, device, args, adv_test, epsilon=8/255, steps=10):
    model.eval()

    test_scores = []
    test_labels = []
    test_masks = []

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            if adv_test:
                masks = label_patch(patchify(batch[2], model.patch_size)).to(device)
                with torch.set_grad_enabled(True):
                    if args.attack_type == "PGD":
                        images = pgd_attack(model, images, masks, epsilon, steps)

            masks = batch[2].to(device)
            scores = model(images)

            test_scores.append(scores.cpu())
            test_labels.append(labels.cpu())
            test_masks.append(masks.cpu())

    image_auc, pixel_auc = get_auc(test_scores, test_labels, test_masks, model.patches_per_side, args.smoothing_sigma, args.smoothing_radius, args.top_k)

    return image_auc, pixel_auc

def run_test(args):
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    model = PatchGuard(args, device).to(device)
    load_model(model, args.checkpoint_dir+f"patchguard_{args.dataset}_{args.class_name}.pth")
    _, test_loader = get_dataloader(args.image_size, args.dataset_dir, args.dataset, args.class_name, args.train_batch_size, args.test_batch_size, args.num_workers, args.seed)

    image_auc, pixel_auc = test(model, test_loader, device, args, False)
    display_results({"Image-level AUC":image_auc, "Pixel-level AUC":pixel_auc}, "Clean Performance")

    if args.adv_test:
        epsilons = args.epsilon_test
        step = args.step_test

        for epsilon in epsilons:
            image_auc, pixel_auc = test(model, test_loader, device, args, True, epsilon, step)
            display_results({"Image-level AUC":image_auc, "Pixel-level AUC":pixel_auc}, f"{args.attack_type} attack (eps={epsilon}, step={step})")
