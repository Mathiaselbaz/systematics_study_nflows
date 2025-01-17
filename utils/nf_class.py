
import numpy as np
import torch
import torch.nn as nn


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

    def forward_kld(self, x, context=None):
        """Estimates forward KL divergence

        Args:
          x: Batch sampled from target distribution
          context: Batch of conditions/context

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z, context=context)
            log_q += log_det
        log_q += self.q0.log_prob(z, context=context)
        return -torch.mean(log_q)
    
    def forward_kld_importance(self, idx, verbose):
        """Estimates forward KL divergence  with importance sampling  for a given index"""

        device = self.device
        zb, context, ind_log_g, log_p = self.p.log_prob(idx=idx)
        ind_log_g= ind_log_g.to(device)
        dim_context = context.shape[1]
        dim_data = zb.shape[1]
        zb = zb.to(device)
        log_g = torch.sum(ind_log_g, dim=1, keepdim=True).to(device)
        log_p = log_p.unsqueeze(1).to(device)
        context = context.to(device)
        log_g_context= torch.sum(ind_log_g[:,dim_data:], dim=1, keepdim=True).to(device)
        log_q = self.log_prob(zb, context=context)
        if verbose :
            print('log_q : ', torch.mean(log_q).item())
            print('log_p : ', torch.mean(log_p).item())
            print('log_g : ', torch.mean(log_g).item())
            print('log_g_context : ', torch.mean(log_g_context).item())
            print('Forward kld :', torch.mean(torch.exp((log_p - log_g ))*(log_p - log_q - log_g_context)).item())
            print('Weights :', torch.mean(torch.exp(log_p - log_g )).item())
        return torch.mean(torch.exp((log_p - log_g ))*(log_p - log_q - log_g_context))
    
    def reverse_kld_importance(self, idx, verbose):
        """Estimates reverse KL divergence  with importance sampling  for a given index"""

        device = self.device
        zb, context, ind_log_g, log_p = self.p.log_prob(idx=idx)
        ind_log_g= ind_log_g.to(device)
        dim_context = context.shape[1]
        dim_data = zb.shape[1]
        zb = zb.to(device)
        log_g = torch.sum(ind_log_g, dim=1, keepdim=True).to(device)
        log_p = log_p.unsqueeze(1).to(device)
        context = context.to(device)
        log_g_context= torch.sum(ind_log_g[:,dim_data:],dim=1, keepdim=True).to(device)
        log_q = self.log_prob(zb, context=context)
        
        if verbose :
            print('log_q : ', torch.mean(log_q).item())
            print('log_p : ', torch.mean(log_p).item())
            print('log_g : ', torch.mean(log_g).item())
            print('log_g_context : ', torch.mean(log_g_context).item())
            print('Weights :', torch.mean(torch.exp(log_q - log_g +log_g_context)).item())
            print('Reverse kld :', torch.mean(torch.exp((log_q - log_g +log_g_context))*(log_q - log_p + log_g_context)).item())
            
        return torch.mean(torch.exp((log_q - log_g +log_g_context))*(log_q - log_p + log_g_context))
    
    def symmetric_kld_importance(self, idx, alpha=1, beta=1, verbose=False):
        """Estimates symmetric KL divergence  with importance sampling  for a given index using the two functions above"""
        
        return alpha*self.reverse_kld_importance(idx, verbose) + beta*self.forward_kld_importance(idx, verbose)
    

    def reverse_kld(self, num_samples=1, context=None, beta=1.0, score_fn=True):
        """Estimates reverse KL divergence

        Args:
          num_samples: Number of samples to draw from base distribution
          context: Batch of conditions/context
          beta: Annealing parameter, see [arXiv 1505.05770](https://arxiv.org/abs/1505.05770)
          score_fn: Flag whether to include score function in gradient, see [arXiv 1703.09194](https://arxiv.org/abs/1703.09194)

        Returns:
          Estimate of the reverse KL divergence averaged over latent samples
        """
        z, log_q_ = self.q0(num_samples, context=context)
        log_q = torch.zeros_like(log_q_)
        log_q += log_q_
        for flow in self.flows:
            z, log_det = flow(z, context=context)
            log_q -= log_det
        if not score_fn:
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            utils.set_requires_grad(self, False)
            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_, context=context)
                log_q += log_det
            log_q += self.q0.log_prob(z_, context=context)
            utils.set_requires_grad(self, True)
        log_p = self.p.log_prob(z, context=context)
        return torch.mean(log_q) - beta * torch.mean(log_p)
        
    def symmetric_kld(self, i, num_samples=1, pedestal=0.005, beta=1.0, gamma=1.0, verbose=False, score_fn=True):
        """Estimates symmetric KL divergence
    
        Args:
          num_samples: Number of samples to draw from base distribution
    
        Returns:
          Estimate of the symmetric KL divergence averaged over latent samples
        """
    
        # Sample from the base distribution
        z, _ = self.q0(num_samples)
    
        # Get log probability and context from p.log_prob()
        zb, context, log_p = self.p.log_prob(i, pedestal)
    
        # Move tensors to the same device
        zb = zb.to(z.device)
        log_p = log_p.to(z.device)
        context = context.to(z.device)
    
        # Compute log probabilities
        log_q = self.log_prob(zb, context=context)
    
        # Compute a constant log_qb (log probability of a uniform base distribution)
        log_qb = torch.full_like(log_q, np.log(1 / (2 ** 4)))
    
        # Disable gradient tracking for ratio computations
        with torch.no_grad():
            pq_ratio = torch.exp(log_p - log_qb)
            qbq_ratio = torch.exp(log_q - log_qb)
    
        # Verbose output for debugging
        if verbose:
            print('log_p : ', torch.mean(log_p).item())
            print('log_q : ', torch.mean(log_q).item())
            print('pq_ratio : ', torch.mean(pq_ratio).item())
            print('qbq_ratio : ', torch.mean(qbq_ratio).item())
            print('Reverse kld :', torch.mean(qbq_ratio * (log_q - log_p)).item())
            print('Forward kld :', torch.mean(pq_ratio * (log_p - log_q)).item())
            print('Symmetric kld :', (torch.mean(qbq_ratio * (log_q - log_p)) + torch.mean(pq_ratio * (log_p - log_q))).item())
    
        # Compute the symmetric KLD
        forward_kld = torch.mean(pq_ratio * (log_p - log_q))
        reverse_kld = torch.mean(qbq_ratio * (log_q - log_p))
        symmetric_kld = beta * reverse_kld + gamma * forward_kld
    
        # Explicitly delete intermediate tensors to free memory
        del pq_ratio, qbq_ratio, log_q, log_p, log_qb, zb, context
        torch.cuda.empty_cache()
    
        return symmetric_kld
