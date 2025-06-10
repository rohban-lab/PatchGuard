import torch
import torch.nn.functional as F

def pgd_attack(model, images, masks, epsilon, num_iter):
    alpha =  (2.5 * epsilon) / num_iter

    X = images.clone().detach()
    original_X = X.data

    for i in range(num_iter) :    
        X.requires_grad = True

        scores = model(X)

        zeros_count = (masks == 0).sum(dim=1)
        non_zeros_count = (masks != 0).sum(dim=1)

        anomalous_loss = (masks * scores).sum(dim=1) / (non_zeros_count + 1e-8)
        normal_loss = ((1 - masks) * scores).sum(dim=1) / (zeros_count + 1e-8)
        loss = normal_loss.sum() - anomalous_loss.sum()

        loss.backward()

        adv_X = X + alpha * X.grad.sign()
        delta = torch.clamp(adv_X - original_X, min=-epsilon, max=epsilon)
        X = torch.clamp(original_X + delta, min=0, max=1).detach_()

    return X
