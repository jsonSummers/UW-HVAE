# loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F



def make_objective_function(device, use_vgg, lambda_kl, lambda_l1, lambda_con):


    if use_vgg:
        from torchvision import models
        vgg_weights_path = '../Model/vgg19-dcbb9e9d.pth'
        vgg_model = models.vgg19(pretrained=False)
        vgg_model.load_state_dict(torch.load(vgg_weights_path))
        vgg_model.eval().to(device)
    else:
        vgg_model = None

    def Loss(enhanced_images, target_images, mu, logvar):
        # KL divergence loss
        kl_loss_val = lambda_kl * kl_loss(mu, logvar)

        # Calculate reconstruction loss
        l1_loss_val = lambda_l1 * l1_loss(enhanced_images, target_images)

        # Content loss
        con_loss_val = lambda_con * content_loss(enhanced_images, target_images)

        return kl_loss_val, l1_loss_val, con_loss_val

    return Loss


def kl_loss(mu, logvar):
    # Calculate KL divergence loss using the provided formula
    kl_loss_val = 0.5 * torch.mean(torch.exp(logvar) + mu**2 - 1 - logvar)
    return kl_loss_val


def l1_loss(enhanced_images, target_images):
    # Calculate L1 loss between enhanced and target images
    l1_loss_val = nn.L1Loss(enhanced_images, target_images)
    return l1_loss_val


def content_loss_vgg(vgg_model, enhanced_image, clean_image):
    # minimise
    # Extract features from the block5_conv2 layer
    features_enhanced = vgg_model(enhanced_image)
    features_clean = vgg_model(clean_image)

    # Compute the content loss
    loss_content = F.mse_loss(features_enhanced, features_clean)

    return loss_content


def content_loss_non_deep(enhanced_image, clean_image):
    # Compute the content loss using mean squared error (MSE)
    loss_content = F.mse_loss(enhanced_image, clean_image)
    return loss_content


def content_loss(enhanced_image, clean_image, vgg_model=None):
    if vgg_model is not None:
        return content_loss_vgg(vgg_model, enhanced_image, clean_image)
    else:
        return content_loss_non_deep(enhanced_image, clean_image)
