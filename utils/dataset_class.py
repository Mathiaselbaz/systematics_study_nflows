# Dataset class to train the normalizing flow model. 
# It loads a pickle file with a numpy dictionnary.
# It contains the nsample*ndim dataset in the eigenspace of the covariance matrix, the true probability not assuming gaussianity, the covariance matrix and the mean of the dataset before standardization.
# This class is used in the training script. It inherits from torch.utils.data.Dataset.
# """

import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns



class SystematicDataset(Dataset):
    def __init__(self, data_file, start_conditional):
        """Loads the dataset from a pickle file

        Args:
          data_file: Path to the pickle file containing the dataset
        """
        data = np.load(data_file)
        self.start_conditional = start_conditional
        self.data = torch.tensor(data['data'], dtype=torch.float32)
        self.mean = torch.tensor(data['mean'], dtype=torch.float32)
        self.cov = torch.tensor(data['cov'], dtype=torch.float32)
        self.log_p = torch.tensor(data['log_p'], dtype=torch.float32)
        self.cholesky = torch.linalg.cholesky(self.cov)
        self.nsample, self.ndim = self.data.shape

    def __len__(self):
        return self.nsample

    def __getitem__(self, idx):
        return self.data[idx,:self.start_conditional], self.data[idx,self.start_conditional:], self.ind_n_log_g(self.data[idx,:]), self.n_log_p(gundam=False, idx=idx)
    
    def get_cov(self):
        return self.cov
    
    def get_mean(self):
        return self.mean
    
    def ind_n_log_g(self, x):
        """Computes the log density of the data distribution for one dimension assuming gaussianity"""
        return 0.5 * (np.log(2 * np.pi) + x**2)

    
    def n_log_g(self, x):
        """Computes the log density of the data distribution assuning gaussianity by summing the log density of each data point

        Args:
          x: Batch of data points

        Returns:
          Log density of the data distribution
        """
        return 0.5 * (self.ndim * np.log(2 * np.pi) + torch.sum(x**2, dim=1))
    
    def n_log_p(self, gundam=False, idx=None):
        """Returns the log density of the data distribution from the file or by computing it with GUNDAM (not implemented)"""
        if gundam:
            raise NotImplementedError
        else:
            if idx is None:
                raise ValueError("Please provide an index to compute the log probability")
            return self.log_p[idx]
        
    def transform_eigen_space_to_data_space(self, x):
        """Transforms a batch of data points from the eigenspace to the data space"""
        return torch.mm(x, self.cholesky.T) + self.mean
    
    def plot_histo_data(self, n_sample=100, n_bins=100, eigen=False, gaussian_reweight=False):
        """
        Plots standard 1D and 2D histograms of the data (without Seaborn),
        using a "plasma" colormap for the 2D plots.

        Args:
            n_sample (int): Number of samples to plot.
            n_bins (int): Number of bins for histograms.
            eigen (bool): If True, plot self.data (i.e., 'eigenspace');
                        otherwise transform to 'data space' first.
            gaussian_reweight (bool): If True, weight each sample by exp(-log_g).
        """

        # Decide which data to plot
        if eigen:
            data_plot = self.data[:n_sample, :]
        else:
            data_plot = self.transform_eigen_space_to_data_space(self.data[:n_sample, :])

        # Optionally compute sample weights
        if gaussian_reweight:
            # Assuming self.log_p is already negative log-likelihood (shape: [nsample])
            weights = torch.exp(-self.n_log_g(data_plot))
        else:
            weights = None

        # Create a figure and a grid of subplots
        fig, ax = plt.subplots(self.ndim, self.ndim, figsize=(3*self.ndim, 3*self.ndim))

        # Handle 1D case
        if self.ndim == 1:
            # Single 1D histogram
            ax.hist(
                data_plot[:, 0].numpy(),
                bins=n_bins,
                weights=None if weights is None else weights.numpy(),
                color="blue",
                alpha=0.7
            )
        else:
            # Higher-dimensional case: plot grid of subplots
            for i in range(self.ndim):
                for j in range(self.ndim):
                    if i == j:
                        # Diagonal: 1D histogram
                        ax[i, j].hist(
                            data_plot[:, i].numpy(),
                            bins=n_bins,
                            weights=None if weights is None else weights.numpy(),
                            color="blue",
                            alpha=0.7
                        )
                    else:
                        # Off-diagonal: 2D histogram with "plasma" colormap
                        h = ax[i, j].hist2d(
                            x=data_plot[:, j].numpy(),
                            y=data_plot[:, i].numpy(),
                            bins=n_bins,
                            weights=None if weights is None else weights.numpy(),
                            cmap="plasma"
                        )
                        # Optionally, you could add a colorbar for each subplot:
                        # fig.colorbar(h[3], ax=ax[i, j])

        plt.tight_layout()
        plt.show()
        plt.savefig('../img/histo_data.png')
        plt.close()