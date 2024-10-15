import torch
import torch.nn as nn
import numpy as np

class DiffusionModel(nn.Module):
    def __init__(self, input_dim, num_timesteps, device):
        super(DiffusionModel, self).__init__()
        self.input_dim = input_dim
        self.num_timesteps = num_timesteps
        self.device = device
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

        # Define a cosine noise schedule
        self.betas = torch.from_numpy(self.get_noise_schedule(num_timesteps)).float().to(device)

    def get_noise_schedule(self, num_timesteps):
        # A cosine noise schedule can help smooth out noise
        return np.cos(np.linspace(0, np.pi / 2, num_timesteps))**2

    def q_sample(self, x_start, t, noise):
        return x_start + noise * t  # Simple perturbation, modify as needed for your model

    def p_sample(self, x_t, t, context):
        # Predict the noise
        noise_pred = self.network(torch.cat((x_t, context), dim=1))

        # Subtract the predicted noise to reverse one step of the diffusion process
        x_t = x_t - noise_pred

        # Enforce non-negative output
        x_t = torch.relu(x_t)  # Ensure no negative values in output

        return x_t

    def compute_loss(self, x):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        noise = torch.randn_like(x)

        # Sample from the forward process (adding noise)
        x_t = self.q_sample(x, t, noise)

        # Predict noise with the network
        noise_pred = self.network(torch.cat((x_t, t.unsqueeze(1)), dim=1))

        # Calculate the loss as mean squared error between the predicted noise and actual noise
        loss = torch.mean((noise_pred - noise) ** 2)

        # Add a penalty for negative values in x_t to prevent unphysical outputs
        penalty = torch.mean(torch.relu(-x_t))  # Penalize negative values

        return loss + penalty

