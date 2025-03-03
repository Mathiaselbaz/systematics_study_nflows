#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#  Title: SystematicFlow - Normalizing Flow Class
#  Author: Mathias El Baz
#  Date: 28/01/2025
#  Description:
#     This module defines a conditional Normalizing Flow model (SystematicFlow),
#     leveraging conditional information in the flow layers.
#     It includes several importance-sampling-based losses:
#     exponential loss, forward KL, reverse KL, and a symmetric KL. 
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from normflows.core import NormalizingFlow
from normflows import utils, distributions


class SystematicFlow(NormalizingFlow):
    """
    Conditional normalizing flow model, providing condition (also called context)
    to both the base distribution and the flow layers.
    """

    def __init__(self, base, flows, target):
        """Initializes the normalizing flow model

        Args:
          base: Base distribution
          flows: List of flow layers
          target: Target distribution
        """
        super(SystematicFlow, self).__init__(base, flows, target)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
          Batch in the latent space, log determinant of the Jacobian
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
          Log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z, context=context)
            log_q += log_det
        log_q += self.q0.log_prob(z, context=context)
        return log_q

    def plot_nll_ess(self, idx, weight_cap=np.exp(150), verbose=False):
        device = self.device
        zb, context, log_g, log_g_cond, log_p = self.p.log_prob(idx=idx)
        log_g = log_g.unsqueeze(1).to(device)
        log_g_cond = log_g_cond.unsqueeze(1).to(device)
        log_p = log_p.unsqueeze(1).to(device)
        zb = zb.to(device)
        context = context.to(device)
        log_q = self.log_prob(zb, context=context).unsqueeze(1)
        weight_cap = torch.tensor(weight_cap, device=log_q.device).expand_as(log_q)

        if verbose:
            print('log_q : ', torch.mean(log_q).item())
            print('log_p : ', torch.mean(log_p).item())
            print('log_g : ', torch.mean(log_g).item())
            print('log_g_context : ', torch.mean(log_g_cond).item())
            print(' Mean Forward Weights :', torch.mean(torch.min(torch.exp(log_p - log_g), weight_cap)).item())
            print(' Mean Reverse Weights :', torch.mean(torch.exp(log_q - log_g + log_g_cond)).item())
            print('Forward Exponential loss : ', torch.mean(torch.min(torch.exp(log_p - log_g), weight_cap) * (log_p - log_q - log_g_cond) ** 2).item())
            print('Reverse exponential loss :', torch.mean(torch.exp((log_q - log_g + log_g_cond)) *(log_q - log_p + log_g_cond)**2).item())
            print('Forward kld : ', torch.mean(torch.min(torch.exp(log_p - log_g), weight_cap) * (log_p - log_q - log_g_cond)).item())
            print('Reverse kld :', torch.mean(torch.exp((log_q - log_g + log_g_cond)) *(log_q - log_p + log_g_cond)).item())

        weights = torch.exp(log_p - log_q - log_g_cond).detach().cpu().numpy()
        ess = (np.sum(weights)**2) / np.sum(weights**2)
        ess_percentage = ess / len(weights) * 100

        plt.figure(figsize=(8, 6))
        hist = plt.hist2d(-log_p.squeeze(1).detach().cpu().numpy(),
                          -(log_q.squeeze(1) + log_g_cond.squeeze(1)).detach().cpu().numpy(),
                          bins=50, norm=LogNorm(), cmap='viridis')
        cbar = plt.colorbar(hist[3])
        plt.text(
            0.05, 0.95,
            f'ESS: {ess_percentage:.2f}%',
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment='top',
            horizontalalignment='left',
        )
        plt.title('-log(p) vs -log(NF)')
        plt.xlabel('-log(p)')
        plt.ylabel('-log(NF)')
        plt.savefig('img/NLLH_vs_NLNF.png')
        plt.close()

        plt.figure(figsize=(8, 6))
        hist = plt.hist2d(-log_p.squeeze(1).detach().cpu().numpy(),
                          -log_g.squeeze(1).detach().cpu().numpy(),
                          bins=50, norm=LogNorm(), cmap='viridis')
        cbar = plt.colorbar(hist[3])
        cbar.set_label('Weight Density (log scale)')

        weights = torch.exp(log_p - log_g).detach().cpu().numpy()
        ess_2 = (np.sum(weights)**2) / np.sum(weights**2)
        ess_percentage = ess_2 / len(weights) * 100
        plt.text(
            0.05, 0.95,
            f'ESS: {ess_percentage:.2f}%',
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment='top',
            horizontalalignment='left',
        )
        plt.title('-log(p) vs -log(g)')
        plt.xlabel('-log(p)')
        plt.ylabel('-log(g)')
        plt.savefig('img/NLLH_vs_NLg.png')
        plt.close()
        print('Ratio of ESS : ', ess/ess_2)

        return ess/ess_2, ess
    
    def exponential_loss_importance(self, idx, weight_cap=np.exp(150), power=1, verbose=False, plot_hist_weight=False):
        device = self.device
        zb, context, log_g, log_g_cond, log_p = self.p.log_prob(idx=idx)
        log_g = log_g.unsqueeze(1).to(device)
        log_g_cond = log_g_cond.unsqueeze(1).to(device)
        log_p = log_p.unsqueeze(1).to(device)
        zb = zb.to(device)
        context = context.to(device)
        log_q = self.log_prob(zb, context=context).unsqueeze(1)
        weight_cap = torch.tensor(weight_cap, device=log_q.device).expand_as(log_q)

        if verbose:
            print('log_q : ', torch.mean(log_q).item())
            print('log_p : ', torch.mean(log_p).item())
            print('log_g : ', torch.mean(log_g).item())
            print('log_g_context : ', torch.mean(log_g_cond).item())
            print(' Mean Weights :', torch.mean(torch.min(torch.exp(log_p - log_g), weight_cap)).item())
            print('Exponential loss : ', torch.mean(torch.min(torch.exp(log_p - log_g), weight_cap) * (log_p - log_q - log_g_cond) ** 2).item())

        if plot_hist_weight:
            weights = torch.exp(log_p - log_q - log_g_cond).detach().cpu().numpy()
            ess = (np.sum(weights)**2) / np.sum(weights**2)
            ess_percentage = ess / len(weights) * 100

            plt.figure(figsize=(8, 6))
            hist = plt.hist2d(-log_p.squeeze(1).detach().cpu().numpy(),
                              -(log_q.squeeze(1) + log_g_cond.squeeze(1)).detach().cpu().numpy(),
                              bins=50, norm=LogNorm(), cmap='viridis')
            cbar = plt.colorbar(hist[3])
            plt.text(
                0.05, 0.95,
                f'ESS: {ess_percentage:.2f}%',
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment='top',
                horizontalalignment='left',
            )
            plt.title('-log(p) vs -log(NF)')
            plt.xlabel('-log(p)')
            plt.ylabel('-log(NF)')
            plt.savefig('img/NLLH_vs_NLNF.png')
            plt.close()

            plt.figure(figsize=(8, 6))
            hist = plt.hist2d(-log_p.squeeze(1).detach().cpu().numpy(),
                              -log_g.squeeze(1).detach().cpu().numpy(),
                              bins=50, norm=LogNorm(), cmap='viridis')
            cbar = plt.colorbar(hist[3])
            cbar.set_label('Weight Density (log scale)')

            weights = torch.exp(log_p - log_g).detach().cpu().numpy()
            ess_2 = (np.sum(weights)**2) / np.sum(weights**2)
            ess_percentage = ess_2 / len(weights) * 100
            plt.text(
                0.05, 0.95,
                f'ESS: {ess_percentage:.2f}%',
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment='top',
                horizontalalignment='left',
            )
            plt.title('-log(p) vs -log(g)')
            plt.xlabel('-log(p)')
            plt.ylabel('-log(g)')
            plt.savefig('img/NLLH_vs_NLg.png')
            plt.close()
            print('Ratio of ESS : ', ess_2/ess)

        exp_loss = torch.pow(torch.mean(
            torch.min(torch.exp(log_p - log_g), weight_cap) *
            (log_p - log_q - log_g_cond)**2
        ), power)
        del(log_p, log_g, log_g_cond, log_q)

        return exp_loss
    
    def reverse_exponential_loss_importance (self, idx, verbose=False):
        """Estimates reverse exp loss with importance sampling for a given index."""

        device = self.device
        zb, context, log_g, log_g_cond, log_p = self.p.log_prob(idx=idx)
        log_g = log_g.unsqueeze(1).to(device)
        log_g_cond = log_g_cond.unsqueeze(1).to(device)
        log_p = log_p.unsqueeze(1).to(device)
        zb = zb.to(device)
        context = context.to(device)
        log_q = self.log_prob(zb, context=context).unsqueeze(1)

        if verbose:
            print(' Mean Weights :', torch.mean(torch.exp(log_q - log_g + log_g_cond)).item())
            print('Reverse exponential loss :', torch.mean(torch.exp((log_q - log_g + log_g_cond)) *(log_q - log_p + log_g_cond)**2).item())

        exp_loss = torch.mean(torch.exp((log_q - log_g + log_g_cond)) *(log_q - log_p + log_g_cond)**2)
        del(log_p, log_g, log_g_cond, log_q)
        return exp_loss
    
    def symmetric_exponential_loss (self, idx, weight_cap=np.exp(150), power=1, alpha=1, beta=1, verbose=False, plot_hist_weight=False):
        return (
              alpha * self.reverse_exponential_loss_importance(idx, verbose=verbose) +
              beta * self.exponential_loss_importance(idx,weight_cap=weight_cap, power=power, verbose=verbose, plot_hist_weight=plot_hist_weight)
          )
    

    
        
    
    

    def forward_kld_importance(self, idx, weight_cap=np.exp(20), power=50/711, verbose=False, plot_hist_weight=False):
        """Estimates forward KL divergence with importance sampling for a given index."""

        device = self.device
        zb, context, log_g, log_g_cond, log_p = self.p.log_prob(idx=idx)
        log_g = log_g.unsqueeze(1).to(device)
        log_g_cond = log_g_cond.unsqueeze(1).to(device)
        log_p = log_p.unsqueeze(1).to(device)
        zb = zb.to(device)
        context = context.to(device)
        log_q = self.log_prob(zb, context=context).unsqueeze(1)
        weight_cap = torch.tensor(weight_cap, device=log_q.device).expand_as(log_q)

        if verbose:
            print('log_q : ', torch.mean(log_q).item())
            print('log_p : ', torch.mean(log_p).item())
            print('log_g : ', torch.mean(log_g).item())
            print('log_g_context : ', torch.mean(log_g_cond).item())
            print('Forward kld :', torch.pow(torch.mean(
                torch.min(torch.exp(log_p - log_g), weight_cap) * (log_p - log_q - log_g_cond)),power).item())
            print(' Mean Weights :', torch.pow(torch.mean(torch.min(torch.exp(log_p - log_g), weight_cap)),power).item())

        if plot_hist_weight:
            weights = torch.exp(log_p - log_q - log_g_cond).detach().cpu().numpy()
            ess = (np.sum(weights)**2) / np.sum(weights**2)
            ess_percentage = ess / len(weights) * 100

            plt.figure(figsize=(8, 6))
            hist = plt.hist2d(-log_p.squeeze(1).detach().cpu().numpy(),
                              -(log_q.squeeze(1) + log_g_cond.squeeze(1)).detach().cpu().numpy(),
                              bins=50, norm=LogNorm(), cmap='viridis')
            cbar = plt.colorbar(hist[3])
            plt.text(
                0.05, 0.95,
                f'ESS: {ess_percentage:.2f}%',
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment='top',
                horizontalalignment='left',
            )
            plt.title('-log(p) vs -log(NF)')
            plt.xlabel('-log(p)')
            plt.ylabel('-log(NF)')
            plt.savefig('img/NLLH_vs_NLNF.png')
            plt.close()

            plt.figure(figsize=(8, 6))
            hist = plt.hist2d(-log_p.squeeze(1).detach().cpu().numpy(),
                              -log_g.squeeze(1).detach().cpu().numpy(),
                              bins=50, norm=LogNorm(), cmap='viridis')
            cbar = plt.colorbar(hist[3])
            cbar.set_label('Weight Density (log scale)')

            weights = torch.exp(log_p - log_g).detach().cpu().numpy()
            ess_2 = (np.sum(weights)**2) / np.sum(weights**2)
            ess_percentage = ess_2 / len(weights) * 100
            plt.text(
                0.05, 0.95,
                f'ESS: {ess_percentage:.2f}%',
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment='top',
                horizontalalignment='left',
            )
            plt.title('-log(p) vs -log(g)')
            plt.xlabel('-log(p)')
            plt.ylabel('-log(g)')
            plt.savefig('img/NLLH_vs_NLg.png')
            plt.close()
            print('Ratio of ESS : ', ess/ess_2)
            kld = torch.pow(torch.mean(torch.min(torch.exp((log_p - log_g)), weight_cap) * (log_p - log_q - log_g_cond)),power)
            return kld, ess/ess_2

        kld = torch.pow(torch.mean(torch.min(torch.exp((log_p - log_g)), weight_cap) * (log_p - log_q - log_g_cond)),power)
        del(log_p, log_g, log_g_cond, log_q)
        return kld

    def reverse_kld_importance(self, idx, verbose=False):
        """Estimates reverse KL divergence with importance sampling for a given index."""

        device = self.device
        zb, context, log_g, log_g_cond, log_p = self.p.log_prob(idx=idx)
        log_g = log_g.unsqueeze(1).to(device)
        log_g_cond = log_g_cond.unsqueeze(1).to(device)
        log_p = log_p.unsqueeze(1).to(device)
        zb = zb.to(device)
        context = context.to(device)
        log_q = self.log_prob(zb, context=context).unsqueeze(1)

        if verbose:
            print(' Mean Weights :', torch.mean(torch.exp(log_q - log_g + log_g_cond)).item())
            print('Reverse kld :', torch.mean(torch.exp((log_q - log_g + log_g_cond)) *(log_q - log_p + log_g_cond)).item())

        kld = torch.mean(torch.exp((log_q - log_g + log_g_cond)) *(log_q - log_p + log_g_cond))
        del(log_p, log_g, log_g_cond, log_q)
        return kld

    def symmetric_kld_importance(self, idx, weight_cap=np.exp(150), power=1, alpha=1, beta=1, verbose=False, plot_hist_weight=False):
        """Estimates symmetric KL divergence with importance sampling for a given index."""
        if plot_hist_weight :
          forward_kld, ratio_ess = self.forward_kld_importance(idx, weight_cap=weight_cap, power=power, verbose=verbose, plot_hist_weight=plot_hist_weight)
          return (
              alpha * self.reverse_kld_importance(idx, verbose=verbose) +
              beta * forward_kld
          ), ratio_ess
        else :
            return (
              alpha * self.reverse_kld_importance(idx, verbose=verbose) +
              beta * self.forward_kld_importance(idx,weight_cap=weight_cap, power=power, verbose=verbose, plot_hist_weight=plot_hist_weight)
          )
            

   