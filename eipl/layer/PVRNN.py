import torch
import torch.nn as nn
from torch.distributions import Independent, Normal, kl_divergence
from eipl.utils import get_activation_fn


class PVRNNCell(nn.Module):
    def __init__(
        self,
        z_dim,
        d_dim,
        w,
    ):
        super(PVRNNCell, self).__init__()
        
        self.z_dim = z_dim
        self.d_dim = d_dim
        self.w = w
        
        self.recurrent = nn.GRUCell(z_dim, d_dim)
        self.prior_d2z_mu = nn.Linear(d_dim, z_dim)
        self.prior_d2z_logvar = nn.Linear(d_dim, z_dim)
        self.posterior_d2z_mu = nn.Linear(d_dim, z_dim)
        self.posterior_d2z_logvar = nn.Linear(d_dim, z_dim)
    
    def prior(self, d):
        prior_mu = torch.tanh(self.prior_d2z_mu(d))
        prior_std = torch.exp(self.prior_d2z_logvar(d))
        prior = Independent(Normal(prior_mu, prior_std))
        return prior
    
    def posterior(self, A, d):
        A_mu, A_logvar = A.chunk(2, 1)
        posterior_mu = torch.tanh(self.posterior_d2z_mu(d) + A_mu)
        posterior_std = torch.exp(self.posterior_d2z_logvar(d) + A_logvar)
        posterior = Independent(Normal(posterior_mu, posterior_std))
        return posterior
        
    def forward(self, A, d=None):
        batch_size = A.shape[0]
        device = A.device
        if d is None:
            d = self._initial_d(batch_size, device)
        
        prior = self.prior(d)
        posterior = self.posterior(A, d)
        
        z = posterior.rsample()
        d = self.recurrent(z, d)
        
        kld = kl_divergence(posterior, prior)
        wkld = self.w * kld
        
        return z, d, kld, wkld
    
    def _initial_d(self, batch_size, device):
        d = torch.zeros(batch_size, self.d_dim).to(device)
        return d
