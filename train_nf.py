# Train the normalizing flows model using the normflows package
# It uses the SystematicDataset class and SystematicFlows class in utils

import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import normflows as nf
from utils.dataset_class import SystematicDataset
from utils.nf_class import SystematicFlow
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import tqdm
import torch.nn as nn
import time
import sys
import pickle
import torch.cuda as cuda
import tqdm


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default='Dataset/pickle_files/configMarginalise_Fit_configOa2021_Asimov_12Pars_1M.pkl', help='Path to the dataset file')
parser.add_argument('--batch_size', type=int, default=10000, help='Batch size')
parser.add_argument('--nflows', type=int, default=5, help='Number of flows')
parser.add_argument('--nhidden', type=int, default=256, help='Number of hidden units in the neural networks')
parser.add_argument('--nepochs', type=int, default=1000, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--num_val', type=int, default=10000, help='Number of validation samples')
parser.add_argument('--list_dim_phase_space', type=list, default=[8,9,10,11], help='List of the dimensions of the phase space')
parser.add_argument('--tail_bound', type=float, default=5.0, help='Tail bounds for all dimension in sigma')
parser.add_argument('--num_val_show', type=int, default=20000, help='Number of validation samples')
args = parser.parse_args()

# Set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)


# empty cache and useless memory
torch.cuda.empty_cache()
# Create output directory
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)




def plot_subplot_hist_model(z, dimension_names, epoch):
    ndim=z.shape[1]
    fig, axes = plt.subplots(nrows=ndim, ncols=ndim, figsize=(10, 9))
    for i in range(ndim):
        for j in range(ndim):
            if i == j:
                axes[i, j].hist(z[:, i], bins=100, alpha=0.7)
                axes[i, j].set_xlabel(dimension_names[i])
            else:
                axes[i, j].hist2d(z[:, j], z[:, i], bins=50, density=True, cmap='viridis')
                axes[i, j].set_xlabel(dimension_names[j])
                axes[i, j].set_ylabel(dimension_names[i])

    plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(f'img/learned_distributions.png')
    plt.close()
num_dim = len(args.list_dim_phase_space)
# Load the dataset
dataset = SystematicDataset(args.data_file, args.list_dim_phase_space)
dimension_names = [dataset.titles[i].split('/')[-1] for i in args.list_dim_phase_space]

# Initialize the normalizing flow model
target = SystematicDataset(args.data_file, args.list_dim_phase_space)
base = nf.distributions.DiagGaussian(num_dim)
flow_layers =[nf.flows.AutoregressiveRationalQuadraticSpline(num_dim, 1, args.nhidden, num_context_channels=12-num_dim, num_bins=9, tail_bound=torch.ones(num_dim)*args.tail_bound, permute_mask=True) for _ in range(args.nflows)]
model = SystematicFlow(base, flow_layers, target)

def checkpoint_and_plot_losses(losses, val_losses, output_dir):
    # Save the model
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
    # Plot the losses
    plt.figure()
    plt.plot(losses, label='Training')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'losses.png'))
    plt.close()

if torch.cuda.is_available():
    device = 'cuda'
    print('CUDA available, using GPU')
else:
    device = 'cpu'
    print('CUDA not available, using CPU')
model.to(device)

# Initialize the optimizer
optimizer = Adam(model.parameters(), lr=args.lr)
scheduler = ExponentialLR(optimizer, 0.999)




# Train the model
val_losses = []
train_losses = []
start = time.time()
val_idx = np.random.choice(len(dataset), args.num_val, replace=False)

train_idx = np.setdiff1d(np.arange(len(dataset)), val_idx)
alpha = 0
beta = 1

for epoch in tqdm.tqdm(range(args.nepochs)):
    idx = np.random.choice(train_idx, args.batch_size, replace=False)
    optimizer.zero_grad()
    loss = model.forward_kld_importance(idx, verbose=False)
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_losses.append(loss.item())
    with torch.no_grad():
        val_loss = model.symmetric_kld_importance(val_idx, alpha=alpha, beta=beta)
        val_losses.append(val_loss.item())
    if (epoch+1) % 100 == 0:
        with torch.no_grad():
            val_loss = model.symmetric_kld_importance(val_idx, alpha=alpha, beta=beta, verbose = True, plot_hist_weight = True)
        print('Epoch %d, loss = %.2f, val_loss = %.2f' % (epoch, loss.item(), val_loss.item()))
        checkpoint_and_plot_losses(train_losses, val_losses, args.output_dir)
        context_test= torch.randn(args.num_val_show, dataset.ndim-num_dim).to(device)
        z, _= model.sample(args.num_val_show, context=context_test)
        z = z.detach().cpu().numpy()
        plot_subplot_hist_model(z, dimension_names, epoch+1)
        del(z,context_test)

end = time.time()
print('Training time: %.2f s' % (end - start))
