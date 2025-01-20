
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from normflows.core import NormalizingFlow
from normflows import utils, distributions


class SystematicFlow(NormalizingFlow):
    """
    Conditional normalizing flow model, providing condition,
    which is also called context, to both the base distribution
    and the flow layers
    """
    def __init__(self, base, flows, target):
        """Initializes the normalizing flow model

        Args:
          base: Base distribution
          flows: List of flow layers
          target: Target distribution
        """
        super(SystematicFlow, self).__init__(base, flows,target)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bins = np.logspace(-2, 2, 100)


    def forward(self, z, context=None):
        """Transforms latent variable z to the flow variable x

        Args:
          z: Batch in the latent space
          context: Batch of conditions/context

        Returns:
          Batch in the space of the target distribution
        """
        for flow in self.flows:
            z, _ = flow(z, context=context)
        return z

    def forward_and_log_det(self, z, context=None):
        """Transforms latent variable z to the flow variable x and
        computes log determinant of the Jacobian

        Args:
          z: Batch in the latent space
          context: Batch of conditions/context

        Returns:
          Batch in the space of the target distribution,
          log determinant of the Jacobian
        """
        log_det = torch.zeros(len(z), device=z.device)
        for flow in self.flows:
            z, log_d = flow(z, context=context)
            log_det += log_d
        return z, log_det

    def inverse(self, x, context=None):
        """Transforms flow variable x to the latent variable z

        Args:
          x: Batch in the space of the target distribution
          context: Batch of conditions/context

        Returns:
          Batch in the latent space
        """
        for i in range(len(self.flows) - 1, -1, -1):
            x, _ = self.flows[i].inverse(x, context=context)
        return x

    def inverse_and_log_det(self, x, context=None):
        """Transforms flow variable x to the latent variable z and
        computes log determinant of the Jacobian

        Args:
          x: Batch in the space of the target distribution
          context: Batch of conditions/context

        Returns:
          Batch in the latent space, log determinant of the
          Jacobian
        """
        log_det = torch.zeros(len(x), device=x.device)
        for i in range(len(self.flows) - 1, -1, -1):
            x, log_d = self.flows[i].inverse(x, context=context)
            log_det += log_d
        return x, log_det

    def sample(self, num_samples=1, context=None):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw
          context: Batch of conditions/context

        Returns:
          Samples, log probability
        """
        z, log_q = self.q0(num_samples, context=context)
        for flow in self.flows:
            z, log_det = flow(z, context=context)
            log_q -= log_det
        return z, log_q

    def log_prob(self, x, context=None):
        """Get log probability for batch

        Args:
          x: Batch
          context: Batch of conditions/context

        Returns:
          log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z, context=context)
            log_q += log_det
        log_q += self.q0.log_prob(z, context=context)
        return log_q

    
    def forward_kld_importance(self, idx, verbose = False, plot_hist_weight = False):
        """Estimates forward KL divergence  with importance sampling  for a given index"""

        device = self.device
        zb, context, log_g, log_g_cond, log_p = self.p.log_prob(idx=idx)
        log_g= log_g.unsqueeze(1).to(device)
        log_g_cond= log_g_cond.unsqueeze(1).to(device)
        log_p = log_p.unsqueeze(1).to(device)
        zb = zb.to(device)
        context = context.to(device)
        log_q = self.log_prob(zb, context=context).unsqueeze(1)
        if verbose :
            print('log_q : ', torch.mean(log_q).item())
            print('log_p : ', torch.mean(log_p).item())
            print('log_g : ', torch.mean(log_g).item())
            print('log_g_context : ', torch.mean(log_g_cond).item())
            print('Forward kld :', torch.mean(torch.exp((log_p - log_g ))*(log_p - log_q - log_g_cond)).item())
            print(' Mean Weights :', torch.mean(torch.exp(log_p - log_g )).item())
        if plot_hist_weight :
          plt.figure()
          weights = torch.exp(log_p - log_q - log_g_cond ).detach().cpu().numpy()
          ess = (np.sum(weights)**2) / np.sum(weights**2)
          ess_percentage = ess / len(weights) * 100
          plt.hist(weights, bins = self.bins)
          plt.text(0.05, 0.95, f'ESS: {ess_percentage:.2f}%', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
          plt.title('Weights LLH/NF')
          plt.xscale('log')
          plt.yscale('log')
          plt.savefig('img/Weight_nf_vs_llh')
          plt.close()

          plt.figure()
          weights = torch.exp(log_p - log_g ).detach().cpu().numpy()
          ess = (np.sum(weights)**2) / np.sum(weights**2)
          ess_percentage = ess / len(weights) * 100
          plt.hist(weights, bins = self.bins)
          plt.text(0.05, 0.95, f'ESS: {ess_percentage:.2f}%', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
          plt.title('Weights LLH/Gaussian')
          plt.xscale('log')
          plt.yscale('log')
          plt.savefig('img/Weights_gaussian_vs_llh')
          plt.close()
        return torch.mean(torch.exp((log_p - log_g ))*(log_p - log_q - log_g_cond))
    
    def reverse_kld_importance(self, idx, verbose = False, plot_hist_weight = False):
        """Estimates reverse KL divergence  with importance sampling  for a given index"""

        device = self.device
        zb, context, log_g, log_g_cond, log_p = self.p.log_prob(idx=idx)
        log_g= log_g.unsqueeze(1).to(device)
        log_g_cond= log_g_cond.unsqueeze(1).to(device)
        log_p = log_p.unsqueeze(1).to(device)
        zb = zb.to(device)
        context = context.to(device)
        log_q = self.log_prob(zb, context=context).unsqueeze(1)
        
        if verbose :
            print('log_q : ', torch.mean(log_q).item())
            print('log_p : ', torch.mean(log_p).item())
            print('log_g : ', torch.mean(log_g).item())
            print('log_g_context : ', torch.mean(log_g_cond).item())
            print(' Mean Weights :', torch.mean(torch.exp(log_q - log_g +log_g_cond)).item())
            print('Reverse kld :', torch.mean(torch.exp((log_q - log_g +log_g_cond))*(log_q - log_p + log_g_cond)).item())
        if plot_hist_weight :
          weights = (torch.exp(log_q - log_g +log_g_cond)).detach().cpu().numpy()
          plt.figure()
          plt.hist(weights, bins = self.bins)
          ess = (np.sum(weights)**2) / np.sum(weights**2)
          ess_percentage = ess / len(weights) * 100
          plt.text(0.05, 0.95, f'ESS: {ess_percentage:.2f}%', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
          plt.title('Weights NF/gaussian')
          plt.xscale('log')
          plt.yscale('log')
          plt.savefig('img/Weight_nf_vs_gaussian')   
        return torch.mean(torch.exp((log_q - log_g +log_g_cond))*(log_q - log_p + log_g_cond))
    
    def symmetric_kld_importance(self, idx, alpha=1, beta=1, verbose=False, plot_hist_weight = False):
        """Estimates symmetric KL divergence  with importance sampling  for a given index using the two functions above"""
        
        return alpha*self.reverse_kld_importance(idx, verbose, plot_hist_weight=plot_hist_weight) + beta*self.forward_kld_importance(idx, verbose, plot_hist_weight = plot_hist_weight)
   