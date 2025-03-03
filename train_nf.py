#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  Title: Normalizing Flows Training Script
#  Author: Mathias El Baz
#  Date: 28/01/2025
#  Description:
#       This script trains a normalizing flows model on specified systematic 
#       dimensions using the normflows package, the SystematicDataset class 
#       for data loading, and the SystematicFlow class that wraps the 
#       normflows model. 
#       - Checkpoints are saved with a custom naming scheme indicating:
#           * Date
#           * Total number of systematics
#           * Number of systematics used for training
#       - Training and validation log-losses are plotted and saved in the img/ 
#         directory.
# =============================================================================

import logging
import sys
sys.path.append(r'../utils/normalizing-flows')
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
import pickle
import torch.cuda as cuda
from datetime import datetime
import random


if torch.cuda.is_available():
    device = 'cuda'
    print('CUDA available, using GPU')
else:
    device = 'cpu'
    print('CUDA not available, using CPU')

def parse_list(arg):
    return [int(x) for x in arg.strip('[]').split(',')]

# =============================================================================
#  Parse Arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default='Dataset/npz_files', help='Path to the dataset directory')
parser.add_argument('--batch_size', type=int, default=100000, help='Batch size')
parser.add_argument('--nflows', type=int, default=10, help='Number of flows')
parser.add_argument('--nhidden', type=int, default=512, help='Number of hidden units in the neural networks')
parser.add_argument('--nepochs', type=int, default=10000, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--num_val', type=int, default=50000, help='Number of validation samples')
parser.add_argument('--list_dim_phase_space', type=parse_list, default=range(652,711), help='List of the dimensions of the phase space')
parser.add_argument('--tail_bound', type=float, default=5.0, help='Tail bounds for all dimensions in sigma')
parser.add_argument('--num_val_show', type=int, default=1000, help='Number of validation samples for final plots')
parser.add_argument('--load_hyperparameters', type=bool, default=False, help='Load hyperparameters ?')
args = parser.parse_args()

def set_seed(seed: int):
    """Set seed for reproducibility across NumPy, PyTorch, and Python random module."""
    random.seed(seed)  
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)  


if args.load_hyperparameters==False:
    import os
    sys.path.append(r'../utils/optuna')
    os.system('pip install github optunahub cmaes')
    os.system('pip install --upgrade scipy')
    import optuna 
    import optunahub

    set_seed(42)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "training_systematic_nd_oa2022"  
    file_path = "logs_slurm/hyperparameter_search_oa2022_asimov.log"
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(file_path),  
    )
    study = optuna.create_study(study_name=study_name, direction='maximize', sampler=optunahub.load_module("samplers/auto_sampler").AutoSampler(), storage=storage, load_if_exists=True) 

# =============================================================================
#  Preparation & Utilities
# =============================================================================
torch.cuda.empty_cache()
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
if not os.path.exists("img"):
    os.makedirs("img")

TOTAL_SYSTEMATICS = 711
num_dim = len(args.list_dim_phase_space)

# =============================================================================
#  Dataset Loading and Model Initialization
# =============================================================================

target = SystematicDataset(args.data_file, args.list_dim_phase_space)
target.plot_weights_histogram()
dimension_names = [target.titles[i].split('/')[-1] for i in args.list_dim_phase_space]
base = nf.distributions.DiagGaussian(num_dim)



# =============================================================================
#  Checkpoint and Loss Plot Function
# =============================================================================
def checkpoint_and_plot_logloss(train_log_loss, ess_ratio, ess_epoch, Best=None):
    """
    Saves the model and plots:
      - Training log-loss on the left y-axis (in log10 scale).
      - ESS ratio on the right y-axis.
    Args:
        train_log_loss : Training losses, one per epoch.
        ess_ratio      : ESS ratios, only computed at certain epochs.
        ess_epoch      : The epoch indices where ESS was computed.
    """

    date_str = datetime.now().strftime("%Y-%m-%d")
    if Best=='Best':
        model_name = f"trained_model/{date_str}_model_total{TOTAL_SYSTEMATICS}_trainedOn{num_dim}_Best.pth"
    else :
        model_name = f"trained_model/{date_str}_model_total{TOTAL_SYSTEMATICS}_trainedOn{num_dim}.pth"
    torch.save(model.state_dict(), os.path.join(args.output_dir, model_name))

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Log10(|Loss|)', color='blue')
    ax1.plot([np.log10(np.abs(l)) for l in train_log_loss], label='Train LogLoss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('ESS Ratio', color='green')
    ax2.plot(ess_epoch, ess_ratio, label='ESS Ratio', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

    plt.title('Log-Loss and ESS Ratio')
    plt.tight_layout()
    loss_plot_name = "img/logloss.png"
    plt.savefig(loss_plot_name)
    plt.close(fig)

# =============================================================================
#  Hyperparameters tuning
# =============================================================================

def objective(trial):
    lr = trial.suggest_float('lr', 1e-6, 1e-4)   
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    nflows = trial.suggest_int('nflows', 5, 10)
    if nflows >= 8:
        nbins = trial.suggest_int('nbins', 7, 11)
    else:
        nbins = trial.suggest_int('nbins', 7, 15)

    nlayers = trial.suggest_int('nlayers', 1, 2)

    if nflows >= 8 or nlayers == 2:
        nhidden = trial.suggest_int('nhidden', 256, 384)
    else:
        nhidden = trial.suggest_int('nhidden', 256, 512)

    flow_layers = [
        nf.flows.AutoregressiveRationalQuadraticSpline(
            num_dim, nlayers, nhidden, 
            num_context_channels=TOTAL_SYSTEMATICS - num_dim,
            num_bins=nbins, 
            tail_bound=torch.ones(num_dim)*args.tail_bound, 
            permute_mask=True
        ) 
        for _ in range(nflows)]
  
    model = SystematicFlow(base, flow_layers, target)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, amsgrad=True)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    best_ess = 0 
    for epoch in range(2000):  
        idx = np.random.choice(train_idx, 100000, replace=False)
        optimizer.zero_grad()
        loss = model.exponential_loss_importance(idx)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0:
            
            with torch.no_grad():
                _,ess = model.plot_nll_ess(val_idx, weight_cap=1, verbose=False)
            if ess.item()/100 > best_ess:
                best_ess = ess.item()/100

    return best_ess 

# =============================================================================
#  Initializations before training
# =============================================================================


Ratio_ess, train_losses, epoch_val = [], [], []
start = time.time()
np.random.seed(42)
val_idx = np.random.choice(len(target), args.num_val, replace=False)
train_idx = np.setdiff1d(np.arange(len(target)), val_idx)
alpha = 1
beta = 1

## Early stopping
patience = 10  
min_delta = 1e-2  
wait = 0 
best_ess_ratio = 0  

if args.load_hyperparameters==False :
    study.optimize(objective, n_trials=5)  
    best_params = study.best_params
    for param, value in best_params.items():
        print(f"Best {param}: {value}")
    best_lr, best_gamma, best_nflows, best_nbins, best_nhidden, best_nlayers = (
        best_params['lr'], best_params['gamma'], best_params['nflows'], 
        best_params['nbins'], best_params['nhidden'], best_params['nlayers']
    )
    flow_layers = [
        nf.flows.AutoregressiveRationalQuadraticSpline(
            num_dim, best_nlayers, best_nhidden, 
            num_context_channels=TOTAL_SYSTEMATICS - num_dim,
            num_bins=best_nbins, 
            tail_bound=torch.ones(num_dim)*args.tail_bound, 
            permute_mask=True
        ) 
        for _ in range(best_nflows)]
    model = SystematicFlow(base, flow_layers, target)
    optimizer = Adam(model.parameters(), lr=best_lr, amsgrad=True)
    scheduler = ExponentialLR(optimizer, gamma=best_gamma)
    np.savez("config/hyperparameter.npz", 
         lr=best_lr, gamma=best_gamma, 
         best_nflows=best_nflows, best_nbins=best_nbins, 
         best_nlayers=best_nlayers, best_nhidden=best_nhidden)

else :
    hyperparams = np.load("config/hyperparameter.npz")
    best_lr, best_gamma, best_nhidden, best_nflows, best_nbins, best_nlayers = (
        float(hyperparams["lr"]), float(hyperparams["gamma"]), 
        int(hyperparams["nhidden"]), int(hyperparams["nflows"]), 
        int(hyperparams["nbins"]), int(hyperparams["nlayers"]))

    flow_layers = [
        nf.flows.AutoregressiveRationalQuadraticSpline(
            num_dim, best_nlayers, best_nhidden, 
            num_context_channels=TOTAL_SYSTEMATICS - num_dim,
            num_bins=best_nbins, 
            tail_bound=torch.ones(num_dim)*args.tail_bound, 
            permute_mask=True
        ) 
        for _ in range(best_nflows)]
    model = SystematicFlow(base, flow_layers, target)
    optimizer = Adam(model.parameters(), lr=best_lr, amsgrad=True)
    scheduler = ExponentialLR(optimizer, gamma=best_gamma)

model.to(device)


# =============================================================================
#  Training Loop
# =============================================================================

for epoch in tqdm.tqdm(range(args.nepochs)):
    idx = np.random.choice(train_idx, args.batch_size, replace=False)
    optimizer.zero_grad()
    loss = model.exponential_loss_importance(idx)
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_losses.append(loss.item())

    # with torch.no_grad():
    #     val_loss = model.symmetric_kld_importance(val_idx, alpha=alpha, beta=beta)
    #     val_losses.append(val_loss.item())

    if (epoch+1) % 100 == 0:
        epoch_val.append(epoch)
        with torch.no_grad():
            _,ratio_ess = model.plot_nll_ess(val_idx, weight_cap=1, verbose=True)
        Ratio_ess.append(ratio_ess)
        print(f'Epoch {epoch}, loss = {loss.item():.2f}, ratio_ess = {ratio_ess.item():.4f}')
        checkpoint_and_plot_logloss(train_losses, Ratio_ess, epoch_val)
        if (ratio_ess > best_ess_ratio + min_delta): 
            best_ess_ratio = ratio_ess
            wait = 0  
            checkpoint_and_plot_logloss(train_losses, Ratio_ess, epoch_val, 'Best')
        else:
            wait += 1  

        if (wait >= patience) and (epoch > 2000):
            print(f"Early stopping triggered at epoch {epoch}. No significant ESS improvement for {patience} checks.")
            break  

end = time.time()
print('Training time: %.2f s' % (end - start))

# =============================================================================
#  Final Plots and Cleanup
# =============================================================================
def plot_subplot_hist_model(z, dimension_names, epoch):
    ndim = z.shape[1]
    fig, axes = plt.subplots(nrows=ndim, ncols=ndim, figsize=(10, 9))
    for i in range(ndim):
        for j in range(ndim):
            if i == j:
                axes[i, j].hist(z[:, i], bins=100, alpha=0.7)
                if ndim < 9:
                    axes[i, j].set_xlabel(dimension_names[i])
                axes[i, j].tick_params(axis='x', labelsize=6)
                axes[i, j].tick_params(axis='y', left=False, labelleft=False)
            else:
                axes[i, j].hist2d(z[:, j], z[:, i], bins=50, density=True, cmap='viridis')
                if ndim < 9:
                    axes[i, j].set_xlabel(dimension_names[j])
                    axes[i, j].set_ylabel(dimension_names[i])
                axes[i, j].tick_params(axis='both', which='both', bottom=False, left=False,
                                       labelbottom=False, labelleft=False)
    plt.suptitle(f'Epoch {epoch}')
    plt.savefig('img/learned_distributions.png')
    plt.close()

context_test = torch.randn(args.num_val_show, target.ndim - num_dim).to(device)
z, _ = model.sample(args.num_val_show, context=context_test)
z = z.detach().cpu().numpy()
plot_subplot_hist_model(z, dimension_names, epoch+1)
torch.cuda.empty_cache()
del z, context_test


