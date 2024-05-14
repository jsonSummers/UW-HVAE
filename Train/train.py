# train.py
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import Subset
import numpy as np
import sys
import random
from skimage.color import lab2rgb

import torch
import torchvision.transforms as transforms
import numpy as np
import torchvision.transforms.functional as F
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab, lab2xyz, xyz2rgb
from skimage import color
from sklearn.preprocessing import FunctionTransformer
import random
from torchvision.transforms import ToTensor

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
dataset_path = os.path.join(os.getcwd(), '..', '..', 'Data', 'Paired')
print(f"Dataset path: {dataset_path}")

from Model.model import HierarchicalVAE
from Model.model_config import model_config
from Utils.loss import make_objective_function
from Utils.dataloader import GetTrainingPairs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
torch.cuda.empty_cache()

seed = 16
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
writer = SummaryWriter()

batch_size = 32
num_evaluation_pairs = 5
num_epochs = 10




def initialize_models(learning_rate, patience=5):
    model = HierarchicalVAE(model_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)
    return scheduler, model, optimizer


def initialize_data_loader():
    train_dataset = GetTrainingPairs(root=dataset_path, dataset_name='EUVP')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader


def get_evaluation_images(train_dataset, num_evaluation_pairs):
    dataset_length = len(train_dataset)
    print(f"Number of samples in the dataset: {dataset_length}")
    evaluation_indices = np.random.choice(dataset_length, size=num_evaluation_pairs, replace=False).tolist()
    eval_input_images = []
    eval_target_images = []
    for idx in evaluation_indices:
        sample = train_dataset[idx]
        eval_input_images.append(sample['input_rgb'])
        eval_target_images.append(sample['target_rgb'])

    eval_input_images = torch.stack(eval_input_images).to(device)
    eval_target_images = torch.stack(eval_target_images)

    return eval_input_images, eval_target_images


def make_directories(experiment_id):
    save_path = os.path.join(os.getcwd(), '..', 'Test', 'experiments', experiment_id)

    os.makedirs(os.path.join(save_path, 'checkpoints', 'model_weights'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'final_weights'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'evaluation_images'), exist_ok=True)

    return save_path


def save_evaluation_images(eval_input_images, eval_output_images, model, save_path, seed, epoch):
    with torch.no_grad():
        enhanced_samples, _, _ = model(eval_input_images)

        # Concatenate along dimension 3
        side_by_side = torch.cat([eval_input_images.cpu(), eval_output_images.cpu(), enhanced_samples.cpu()], dim=3)

        save_image(side_by_side,
                   os.path.join(save_path, f"evaluation_images/evaluation_samples_seed_{seed}_epoch_{epoch}.png"),
                   normalize=False)


def train(model, train_loader, optimizer, device, objective_function):
    model.train()
    running_loss = 0.0
    kl_loss_val = 0.0
    l1_loss_val = 0.0
    con_loss_val = 0.0

    for batch_idx, data in enumerate(train_loader):
        inputs, targets = data['input_rgb'].to(device), data['target_rgb'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs, mu, logvar = model(inputs)

        # Calculate losses
        kl_loss_val, l1_loss_val, con_loss_val = objective_function(outputs, inputs, mu, logvar)

        # Total loss
        total_loss = kl_loss_val + l1_loss_val + con_loss_val

        # Backward pass
        total_loss.backward()

        # Update weights
        optimizer.step()

        running_loss += total_loss.item()

    return running_loss / len(train_loader), kl_loss_val, l1_loss_val, con_loss_val


def main(args):

    # Initialize models
    scheduler, model, optimizer = initialize_models(args.lr, args.patience)
    print("Models initialized")

    Objective_function = make_objective_function(device, args.use_vgg, args.lambda_kl, args.lambda_l1, args.lambda_con)

    # Initialize data loader
    train_loader = initialize_data_loader()
    print("Data loader initialized")

    # Get evaluation images
    eval_input_images, eval_output_images = get_evaluation_images(train_loader.dataset, num_evaluation_pairs)

    # Create directories to save the model and results
    save_path = make_directories(args.experiment_id)

    # Training loop
    print("Training started")
    for epoch in range(num_epochs):
        train_loss, kl_loss_val, l1_loss_val, con_loss_val = train(model, train_loader, optimizer, device, Objective_function)

        # Adjust learning rate based on validation loss
        scheduler.step(train_loss)

        writer.add_scalar('Train Loss', train_loss.item(), epoch * len(train_loader) + i)
        writer.add_scalar('L1 Loss', l1_loss_val.item(), epoch * len(train_loader) + i)
        writer.add_scalar('KL Loss', kl_loss_val.item(), epoch * len(train_loader) + i)
        writer.add_scalar('Content Loss', con_loss_val.item(), epoch * len(train_loader) + i)
        # Print training statistics
        print(f'Epoch [{epoch + 1}/{args.num_epochs}], Loss: {train_loss:.4f}')
        save_evaluation_images(eval_input_images, eval_output_images, model, save_path, seed, epoch)

    # Save trained model
    torch.save(model.state_dict(), args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HVAE/NVAE Training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--factor', type=float, default=0.1, help='Factor by which the learning rate will be reduced')
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of epochs with no improvement after which learning rate will be reduced')

    parser.add_argument("experiment_id", type=str, help="Experiment name.")
    parser.add_argument("--lambda_l1", type=float, default=0.55, help="Weight for L1 loss.")
    parser.add_argument("--lambda_con", type=float, default=0.7, help="Weight for content loss.")
    parser.add_argument("--lambda_kl", type=float, default=0.0005, help="Weight for KL divergence loss.")
    parser.add_argument("--use_vgg", action="store_true", help="Whether to use VGG for content loss.")
    args = parser.parse_args()

    make_directories(args.experiment_id)

    args = parser.parse_args()

    main(args)
