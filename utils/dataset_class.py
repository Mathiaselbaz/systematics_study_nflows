#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  Title: SystematicDataset Class
#  Author: Mathias El Baz
#  Date: 28/01/2025
#  Description:
#     Dataset class to train a Normalizing Flow model. It loads a pickle file 
#     containing:
#       - Nsample x Ndim dataset in the eigenspace of the covariance matrix
#       - The true probability (not assuming Gaussianity)
#       - Covariance matrix & mean of the dataset before standardization
#     It inherits from torch.nn.Module and is used in the training script.
# =============================================================================

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import glob


class SystematicDataset(torch.nn.Module):
    def __init__(
        self, 
        data_dir,           # Directory containing batch_i.pkl files
        phase_space_dim, 
        batch_index=None    # If None, load all batches; otherwise load only batch_{batch_index}.pkl
    ):
        """
        Loads the dataset from one or more pickle (.pkl) files.

        Args:
            data_dir (str): Path to the directory containing batch_*.pkl files.
            phase_space_dim (list[int]): Indices of the phase space dimensions. 
                                         The remaining dimensions are treated as conditional variables.
            batch_index (int or None): If an integer, load only batch_{batch_index}.pkl.
                                       If None, load and concatenate all files matching batch_*.pkl.
        """
        super(SystematicDataset, self).__init__()

        # 1. Determine which files to load
        if batch_index is None:
            file_list = sorted(glob.glob(os.path.join(data_dir, "batch*.pkl")))
            if len(file_list) == 0:
                raise FileNotFoundError(
                    f"No files found in {data_dir} matching 'batch*.pkl'"
                )
        else:
            single_file = os.path.join(data_dir, f"batch{batch_index}.pkl")
            if not os.path.isfile(single_file):
                raise FileNotFoundError(f"File {single_file} does not exist.")
            file_list = [single_file]

        # 2. Load the first file once and extract shared metadata
        first_file = file_list[0]
        first_data = np.load(first_file, allow_pickle=True)
        
        # a) Common metadata
        self.mean = torch.tensor(first_data['mean'], dtype=torch.float32)
        
        self.titles = first_data['par_names']
        
        # b) We'll assume the same covariance for all files
        

        # 3. Prepare lists for final concatenation
        data_list = []
        log_p_list = []

        # -- Process the first file --
        norm_first = np.median(np.std(first_data['data'], axis=0))
        data_first = torch.tensor(first_data['data'], dtype=torch.float32) / norm_first
        self.cov = torch.tensor(first_data['cov'], dtype=torch.float32)
        self.cov = self.cov * norm_first**2
        self.cholesky = torch.linalg.cholesky(self.cov)
        
        log_p_first = torch.tensor(first_data['log_p'], dtype=torch.float32)
        
        # Compute per-file log_g and shift
        log_g_first = self.n_log_g(data_first)
        shift_first = torch.median(log_g_first) - torch.median(log_p_first)
        log_p_first += shift_first
        
        data_list.append(data_first)
        log_p_list.append(log_p_first)

        # 4. Process remaining files
        for fpath in file_list[1:]:
            loaded_data = np.load(fpath, allow_pickle=True)
            
            # Compute norm for this file
            norm_file = np.median(np.std(loaded_data['data'], axis=0))
            
            # Normalize data
            data_file = torch.tensor(loaded_data['data'], dtype=torch.float32) / norm_file
            log_p_file = torch.tensor(loaded_data['log_p'], dtype=torch.float32)
            
            # Compute log_g for the current file & apply shift
            log_g_file = self.n_log_g(data_file)
            shift_file = torch.median(log_g_file) - torch.median(log_p_file)
            log_p_file += shift_file
            
            data_list.append(data_file)
            log_p_list.append(log_p_file)

        # 5. Concatenate
        self.data = torch.cat(data_list, dim=0)
        self.log_p = torch.cat(log_p_list, dim=0)

        # 6. Final bookkeeping
        self.nsample, self.ndim = self.data.shape
        self.phase_space_dim = phase_space_dim
        self.list_dim_conditionnal = [i for i in range(self.ndim) if i not in phase_space_dim]

        print(f"Number of files loaded: {len(file_list)}")
        print(f"Dataset shape: {self.nsample}*{self.ndim}")

    def __len__(self):
        return self.nsample

    def __getitem__(self, idx):
        """Returns a batch of data points and the log density of the data distribution

        Args:
          idx: Index of the batch
        """
        return (
            self.data[idx, self.phase_space_dim],
            self.data[idx, self.list_dim_conditionnal],
            -self.n_log_g(self.data[idx, :]),
            -self.n_log_g(self.data[idx, self.list_dim_conditionnal]),
            -self.log_p[idx]
        )

    def get_cov(self):
        return self.cov

    def get_mean(self):
        return self.mean

    def ind_n_log_g(self, x):
        """Computes the log density of the data distribution for one dimension assuming gaussianity"""
        return 0.5 * (np.log(2 * np.pi) + x**2)

    def n_log_g(self, x):
        """Computes the log density of the data distribution assuming gaussianity
        by summing the log density of each data point

        Args:
          x: Batch of data points

        Returns:
          Log density of the data distribution
        """
        n_dim = x.shape[1]
        return 0.5 * (n_dim * np.log(2 * np.pi) + torch.sum(x**2, dim=1))

    def log_prob(self, gundam=False, idx=None):
        """Returns the log density of the data distribution from the file 
        or by computing it with GUNDAM (not implemented)
        """
        if gundam:
            raise NotImplementedError
        else:
            if idx is None:
                raise ValueError("Please provide an index to compute the log probability")
            return (
                self.data[idx[:, None], self.phase_space_dim],
                self.data[idx[:, None], self.list_dim_conditionnal],
                -self.n_log_g(self.data[idx, :]),
                -self.n_log_g(self.data[idx[:, None], self.list_dim_conditionnal]),
                -self.log_p[idx]
            )

    def transform_eigen_space_to_data_space(self, x):
        """Transforms a batch of data points from the eigenspace to the data space"""
        return torch.mm(x, self.cholesky.T) + self.mean

    def plot_weights_histogram(self, save_path='img/hist_weights_all'):
        """
        Plots the histogram of weights for the entire dataset.
        
        The weights are defined as:
            weight[i] = exp( n_log_g(data[i]) - log_p[i] ).

        Args:
            save_path (str or None): If not None, saves the figure to this path.
        """
        nlogg_vals = self.n_log_g(self.data)
        weights = nlogg_vals - self.log_p
        plt.figure(figsize=(8, 6))
        plt.hist(weights.numpy(), color='blue', alpha=0.7)
        plt.xlabel("Weight ")
        plt.ylabel("Frequency")
        plt.yscale('log')
        plt.title("Histogram of Weights for Entire Dataset")

        if save_path is not None:
            plt.savefig(save_path)
        plt.close()
