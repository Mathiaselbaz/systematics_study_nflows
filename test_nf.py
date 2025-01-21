# This script is to test the previously trained NF. It plots n*n subplot of generated samples
# (1D histogram in diagonal and 2D histogram off diagonal).
# It also plots some 1D histogram for some variables.

#!/usr/bin/env python3

import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import normflows as nf
from utils.dataset_class import SystematicDataset
from utils.nf_class import SystematicFlow

def parse_list(arg):
    return [int(x) for x in arg.strip('[]').split(',')]

def load_nf_model(args, num_dim, device):
    ds_temp = SystematicDataset(args.data_file, args.list_dim_phase_space)
    base = nf.distributions.DiagGaussian(num_dim)
    flows = []
    for _ in range(args.nflows):
        flows.append(
            nf.flows.AutoregressiveRationalQuadraticSpline(
                num_dim,
                1,
                args.nhidden,
                num_context_channels=ds_temp.ndim - num_dim,
                num_bins=9,
                tail_bound=torch.ones(num_dim) * args.tail_bound,
                permute_mask=True
            )
        )
    model = SystematicFlow(base, flows, ds_temp)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    del ds_temp
    return model

def generate_samples_in_batches(model, total_samples, gen_batch_size, context_dim, device):
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
    return np.concatenate(z_all, axis=0)

def transform_to_real_space(z, mean_prior, cov_prior):
    z = np.asarray(z, dtype=np.float64)
    mean_prior = np.asarray(mean_prior, dtype=np.float64)
    cov_prior = np.asarray(cov_prior, dtype=np.float64)
    L = np.linalg.cholesky(cov_prior)
    return np.dot(z, L.T) + mean_prior

def plot_pairwise(z, names, prior_means, prior_stds, out_name):
    d = z.shape[1]
    fig, axes = plt.subplots(d, d, figsize=(3*d, 3*d))
    bins = 50
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            ax.set_xlabel("")
            ax.set_ylabel("")
            if i == j:
                ax.hist(z[:, i], bins=bins, histtype='step', color='navy', linewidth=1.5)
                xvals = np.linspace(z[:, i].min(), z[:, i].max(), 200)
                pdf = (1/(prior_stds[i]*np.sqrt(2*np.pi))) * np.exp(-0.5*((xvals - prior_means[i])/prior_stds[i])**2)
                w = (z[:, i].max() - z[:, i].min()) / bins
                ax.plot(xvals, pdf * len(z[:, i]) * w, color='red', linewidth=1.5)
                ax.set_xlabel(names[i])
                
            else:
                ax.hist2d(z[:, j], z[:, i], bins=bins, density=True, cmap='plasma')
                ax.set_xlabel(names[j])
                ax.set_ylabel(names[i])
                ax.set_xticks([])
                ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_name, dpi=150)
    plt.close(fig)

def plot_1d(z, names, vars_to_plot, prior_means, prior_stds, out_name):
    k = len(vars_to_plot)
    fig, axes = plt.subplots(k, 1, figsize=(6, 2.5*k))
    if k == 1:
        axes = [axes]
    bins = 50
    for n, idx in enumerate(vars_to_plot):
        ax = axes[n]
        ax.hist(z[:, idx], bins=bins, histtype='step', color='darkorange', linewidth=1.5, label='Normalizing Flows samples')
        xvals = np.linspace(z[:, idx].min(), z[:, idx].max(), 200)
        pdf = (1/(prior_stds[idx]*np.sqrt(2*np.pi))) * np.exp(-0.5*((xvals - prior_means[idx])/prior_stds[idx])**2)
        w = (z[:, idx].max() - z[:, idx].min()) / bins
        ax.plot(xvals, pdf * len(z[:, idx]) * w, color='red', linewidth=1.5, label='MINUIT Gaussian')
        ax.set_yticks([])
        ax.set_xlabel(names[idx], fontsize=10)
        ax.set_ylabel("Counts", fontsize=10)
        ax.legend(loc='best', fontsize=8)
        
    plt.tight_layout()
    plt.savefig(out_name, dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='Dataset/pickle_files/configMarginalise_Fit_configOa2021_Asimov_12Pars_1M.pkl')
    parser.add_argument('--list_dim_phase_space', type=parse_list, default=[8,9,10,11])
    parser.add_argument('--model_path', type=str, default='output/model.pth')
    parser.add_argument('--output_dir', type=str, default='img')
    parser.add_argument('--nflows', type=int, default=5)
    parser.add_argument('--nhidden', type=int, default=256)
    parser.add_argument('--tail_bound', type=float, default=5.0)
    parser.add_argument('--n_samples', type=int, default=1000000)
    parser.add_argument('--gen_batch_size', type=int, default=10000)
    parser.add_argument('--vars_to_plot', type=parse_list, default=[0,1,2,3])
    parser.add_argument('--transform_to_real_space', type=bool, default=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = SystematicDataset(args.data_file, args.list_dim_phase_space)
    names = [ds.titles[i].split('/')[-1] for i in args.list_dim_phase_space]
    num_dim = len(args.list_dim_phase_space)
    context_dim = ds.ndim - num_dim

    model = load_nf_model(args, num_dim, device)
    z = generate_samples_in_batches(model, args.n_samples, args.gen_batch_size, context_dim, device)

    ds_mean = ds.mean.detach().cpu().numpy() if torch.is_tensor(ds.mean) else ds.mean
    ds_cov = ds.cov.detach().cpu().numpy() if torch.is_tensor(ds.cov) else ds.cov

    if args.transform_to_real_space:
        mean_prior = ds_mean[args.list_dim_phase_space]
        cov_prior = ds_cov[np.ix_(args.list_dim_phase_space, args.list_dim_phase_space)]
        z = transform_to_real_space(z, mean_prior, cov_prior)
        prior_means = mean_prior
        prior_stds = np.sqrt(np.diag(cov_prior))
    else:
        prior_means = np.zeros(num_dim)
        prior_stds = np.ones(num_dim)

    pairwise_out = os.path.join(args.output_dir, "distribution_prediction.png")
    plot_pairwise(z, names, prior_means, prior_stds, pairwise_out)

    hist_out = os.path.join(args.output_dir, "1d_histograms.png")
    plot_1d(z, names, args.vars_to_plot, prior_means, prior_stds, hist_out)

if __name__ == "__main__":
    main()
