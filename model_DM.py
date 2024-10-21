import torch
import torch.nn as nn
import numpy as np

class DiffusionModel(nn.Module):
    def __init__(self, input_dim, num_timesteps, device):
        super(DiffusionModel, self).__init__()
        self.input_dim = input_dim
        self.num_timesteps = num_timesteps
        self.device = device

        # Define the architecture for the network
        # Match the input_dim+1 to account for the context/energy Dimension
        self.network = nn.Sequential(
		    nn.Linear(input_dim + 1, 256),
    		nn.ReLU(),
    		nn.Dropout(0.3),
    		nn.Linear(256, 256),
    		nn.ReLU(),
    		nn.Dropout(0.3),
    		nn.Linear(256, input_dim)
	    )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        beta_t = torch.cos(((t + 1) / self.num_timesteps) * (np.pi / 2)) ** 2
        return x_start + beta_t * noise

    def p_sample(self, x_t, t, context):
        beta_t = torch.cos(((t + 1) / self.num_timesteps) * (np.pi / 2)) ** 2
        noise_pred = self.network(torch.cat((x_t, context), dim=1))
        return x_t - beta_t * noise_pred

    def compute_loss(self, x, context):
        noise = torch.randn_like(x[:, :self.input_dim])
        t = torch.rand(x.size(0), device=x.device).view(-1, 1).expand(x.size(0), self.input_dim)
        x_t = self.q_sample(x[:, :self.input_dim], t, noise)
        noise_pred = self.network(torch.cat((x_t, context), dim=1))

        # Weight the loss based on the energy level
        energy_weights = context.squeeze() / torch.max(context.squeeze())
        loss = torch.mean(((noise_pred - noise) ** 2) * energy_weights.unsqueeze(1))
        return loss

