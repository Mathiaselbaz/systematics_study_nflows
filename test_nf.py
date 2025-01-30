#!/usr/bin/env python3
# =============================================================================
#  ___   ___  ____   ____       _   _                            
# |   \ / _ \|___ \ | ___|_ __ | |_(_) ___  _ __  _   _ _ __ ___ 
# | |) | | | | __) ||___ \| '_ \| __| |/ _ \| '_ \| | | | '__/ _ \
# |___/| |_| |/ __/  ___) | |_) | |_| | (_) | | | | |_| | | |  __/
# |___/ \___/|_____| |____/| .__/ \__|_|\___/|_| |_|\__,_|_|  \___|
#                         |_|                                      
# -----------------------------------------------------------------------------
# This script is to test the previously trained Normalizing Flow model. 
# It:
#   1) Plots an n*n grid of generated samples (1D histogram on diagonals, 2D on off-diagonals).
#   2) Plots some 1D histograms for selected variables.
#   3) Allows automatic variable selection based on the largest deviation 
#      (via K-S test) from a Gaussian with ad-hoc mean & sigma, 
#      computed from the latent samples (before real-space transform).
# -----------------------------------------------------------------------------

import sys
sys.path.append(r'../normalizing-flows')
import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import normflows as nf
import math
import time

# We import kstest from scipy, to measure deviation from Normal.
from scipy.stats import kstest

from utils.dataset_class import SystematicDataset
from utils.nf_class import SystematicFlow

# -----------------------------------------------------------------------------
# Helper function to parse lists from argument strings
# -----------------------------------------------------------------------------
def parse_list(arg):
    return [int(x) for x in arg.strip('[]').split(',')]

# -----------------------------------------------------------------------------
# Load the trained NF model
# -----------------------------------------------------------------------------
def load_nf_model(ds, args, num_dim, device):
    base = nf.distributions.DiagGaussian(num_dim)
    flows = []
    for _ in range(args.nflows):
        flows.append(
            nf.flows.AutoregressiveRationalQuadraticSpline(
                num_dim,
                1,
                args.nhidden,
                num_context_channels=ds.ndim - num_dim,
                num_bins=9,
                tail_bound=torch.ones(num_dim) * args.tail_bound,
                permute_mask=True
            )
        )
    model = SystematicFlow(base, flows, ds)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# Generate NF samples in batches
# -----------------------------------------------------------------------------
def generate_samples_in_batches(model, total_samples, gen_batch_size, context_dim, device):
    start = time.time()
    z_all = []
    done = 0
    while done < total_samples:
        b = min(gen_batch_size, total_samples - done)
        c = torch.randn(b, context_dim, device=device) if context_dim > 0 else None
        with torch.no_grad():
            z_batch, _ = model.sample(b, context=c)
        z_all.append(z_batch.cpu().numpy())
        done += b
        del z_batch, c
    end = time.time()
    print('Sampling time: %.2f s' % (end - start))
    return np.concatenate(z_all, axis=0)

# -----------------------------------------------------------------------------
# Transform latent samples z to real space, given mean and covariance
# -----------------------------------------------------------------------------
def transform_to_real_space(z, mean_prior, cov_prior):
    z = np.asarray(z, dtype=np.float64)
    mean_prior = np.asarray(mean_prior, dtype=np.float64)
    cov_prior = np.asarray(cov_prior, dtype=np.float64)
    L = np.linalg.cholesky(cov_prior)
    return np.dot(z, L.T) + mean_prior

# -----------------------------------------------------------------------------
# Plot pairwise histograms (2D off-diagonal, 1D on-diagonal)
# -----------------------------------------------------------------------------
def plot_pairwise(z, names, vars_to_plot, prior_means, prior_stds, out_name):
    fig, axes = plt.subplots(len(vars_to_plot), len(vars_to_plot), 
                             figsize=(3*len(vars_to_plot), 3*len(vars_to_plot)))
    bins = 50
    for i in range(len(vars_to_plot)):
        for j in range(len(vars_to_plot)):
            ax = axes[i, j]
            ax.set_xlabel("")
            ax.set_ylabel("")
            if i == j:
                i_ind = vars_to_plot[i]
                ax.hist(z[:, i_ind], bins=bins, histtype='step', 
                        color='navy', linewidth=1.5)
                xvals = np.linspace(z[:, i_ind].min(), z[:, i_ind].max(), 200)
                pdf = (1/(prior_stds[i_ind]*np.sqrt(2*np.pi))) * np.exp(
                      -0.5 * ((xvals - prior_means[i_ind]) / prior_stds[i_ind])**2)
                w = (z[:, i_ind].max() - z[:, i_ind].min()) / bins
                ax.plot(xvals, pdf * len(z[:, i_ind]) * w, 
                        color='red', linewidth=1.5)
                ax.set_xlabel(names[i_ind])
            else:
                i_ind = vars_to_plot[i]
                j_ind = vars_to_plot[j]
                ax.hist2d(z[:, j_ind], z[:, i_ind], bins=bins, 
                          density=True, cmap='plasma')
                ax.set_xlabel(names[j_ind])
                ax.set_ylabel(names[i_ind])
                ax.set_xticks([])
                ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_name, dpi=150)
    plt.close(fig)

# -----------------------------------------------------------------------------
# Plot 1D histograms for selected variables
# -----------------------------------------------------------------------------
def plot_1d(z, names, vars_to_plot, prior_means, prior_stds, out_name):
    k = len(vars_to_plot)  # Number of variables to plot
    
    # Automatically find the best grid size (rows x cols)
    cols = math.ceil(math.sqrt(k))  # Number of columns based on square root
    rows = math.ceil(k / cols)      # Number of rows needed to fit all variables

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    # Flatten axes for easier indexing and handle single-variable case
    axes = axes.flatten() if k > 1 else [axes]

    bins = 50
    for n, idx in enumerate(vars_to_plot):
        ax = axes[n]
        
        # Histogram of NF samples
        ax.hist(z[:, idx], bins=bins, histtype='step', 
                color='darkorange', linewidth=1.5, label='NF samples')
        
        # Plot the theoretical Gaussian PDF
        xvals = np.linspace(z[:, idx].min(), z[:, idx].max(), 200)
        pdf = (1 / (prior_stds[idx] * np.sqrt(2 * np.pi))) * np.exp(
               -0.5 * ((xvals - prior_means[idx]) / prior_stds[idx])**2)
        w = (z[:, idx].max() - z[:, idx].min()) / bins
        ax.plot(xvals, pdf * len(z[:, idx]) * w, 
                color='red', linewidth=1.5, label='MINUIT Gaussian')
        
        # Vertical dashed line at the mean of the NF samples (no label)
        ax.axvline(np.mean(z[:, idx]), color='black', linestyle='--')

        # Set labels and remove y ticks
        ax.set_yticks([])
        ax.set_xlabel(names[idx], fontsize=10)
        ax.set_ylabel("Counts", fontsize=10)

    # Turn off any unused subplots
    for ax in axes[k:]:
        ax.axis('off')

    plt.legend()
    plt.tight_layout()
    plt.savefig(out_name, dpi=150)
    plt.close(fig)

# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, 
                        default='Dataset/pickle_files')
    parser.add_argument('--list_dim_phase_space', type=parse_list, 
                        default=range(652,711))
    parser.add_argument('--model_path', type=str, default='output/trained_model/2025-01-30_model_total711_trainedOn59_Best.pth')
    parser.add_argument('--output_dir', type=str, default='img')
    parser.add_argument('--nflows', type=int, default=5)
    parser.add_argument('--nhidden', type=int, default=512)
    parser.add_argument('--tail_bound', type=float, default=5.0)
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--gen_batch_size', type=int, default=200)
    parser.add_argument('--vars_to_plot', type=parse_list, 
                        default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    parser.add_argument('--transform_to_real_space', type=bool, default=True)

    # -------------------------------------------------------------------------
    # New arguments for automatic variable selection based on largest K-S 
    # deviation from a Gaussian with dimension-specific mean and std 
    # (computed from latent samples).
    # -------------------------------------------------------------------------
    parser.add_argument('--auto_var_select', type=bool, default=True,
                        help="If set, we override vars_to_plot by selecting the dimensions that deviate the most from a normal distribution in latent space.")
    parser.add_argument('--auto_var_select_count', type=int, default=12, 
                        help="Number of dimensions to select if auto_var_select is True.")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # Load dataset info for dimension naming and real-space transform
    ds = SystematicDataset(args.data_file, args.list_dim_phase_space, batch_index=0)
    names = [ds.titles[i].split('/')[-1] for i in args.list_dim_phase_space]
    num_dim = len(args.list_dim_phase_space)
    context_dim = ds.ndim - num_dim

    # Load NF model
    model = load_nf_model(ds, args, num_dim, device)

    # Generate latent samples z in the NF's latent space (i.e. untransformed)
    z = generate_samples_in_batches(model, args.n_samples, 
                                    args.gen_batch_size, context_dim, device)

    # -------------------------------------------------------------------------
    # If requested, automatically select the variables that deviate most from 
    # a normal distribution N(mean_i, std_i), where mean_i, std_i are computed 
    # from the latent samples for dimension i.
    # We measure deviation via the K-S statistic.
    # -------------------------------------------------------------------------
    if args.auto_var_select:
        ks_stats = []
        for dim_idx in range(z.shape[1]):
            sample_vals = z[:, dim_idx]
            mean_i = sample_vals.mean()
            std_i = sample_vals.std()
            if std_i == 0:
                # If std is zero (degenerate), consider deviation huge
                ks_stat = float('inf')
            else:
                # Perform the K-S test comparing to N(mean_i, std_i)
                stat, _ = kstest(sample_vals, 'norm', args=(mean_i, std_i))
                ks_stat = stat
            ks_stats.append((dim_idx, ks_stat))
        
        # Sort dimensions by largest K-S statistic
        ks_stats.sort(key=lambda x: x[1], reverse=True)
        # Pick top auto_var_select_count
        top_dims = [x[0] for x in ks_stats[:args.auto_var_select_count]]
        print(f"Auto-selected dimensions (largest K-S deviation): {top_dims}")
        args.vars_to_plot = top_dims

    # Retrieve dataset mean/cov
    ds_mean = ds.mean.detach().cpu().numpy() if torch.is_tensor(ds.mean) else ds.mean
    ds_cov = ds.cov.detach().cpu().numpy()/20 if torch.is_tensor(ds.cov) else ds.cov

    # Transform to real space if requested
    if args.transform_to_real_space:
        mean_prior = ds_mean[args.list_dim_phase_space]
        cov_prior = ds_cov[np.ix_(args.list_dim_phase_space, args.list_dim_phase_space)]
        z = transform_to_real_space(z, mean_prior, cov_prior)
        prior_means = mean_prior
        prior_stds = np.sqrt(np.diag(cov_prior))
    else:
        prior_means = np.zeros(num_dim)
        prior_stds = np.ones(num_dim)

    # Plot pairwise distributions
    pairwise_out = os.path.join(args.output_dir, "distribution_prediction.png")
    plot_pairwise(z, names, args.vars_to_plot, prior_means, prior_stds, pairwise_out)

    # Plot 1D histograms
    hist_out = os.path.join(args.output_dir, "1d_histograms.png")
    plot_1d(z, names, args.vars_to_plot, prior_means, prior_stds, hist_out)

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
