import torch
import torch.nn as nn
import math

# Cosine noise schedule function
def cosine_noise_schedule(timesteps, s=0.008):
    """Cosine noise schedule for diffusion models."""
    steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float32)
    alpha_bar = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    return alpha_bar[:-1] / alpha_bar[1:]  # Return schedule ratios

class DiffusionModel(nn.Module):
    def __init__(self, input_dim, num_timesteps, device, noise_scale=0.001):
        super(DiffusionModel, self).__init__()
        self.input_dim = input_dim
        self.num_timesteps = num_timesteps
        self.device = device
        self.noise_scale = noise_scale  # Store noise_scale as an instance variable

        # Define the architecture for the network
        self.network = nn.Sequential(
            nn.Linear(input_dim + 1, 128),  # Add context (energy) into the input
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),  # Output matches the input data without context (4 dims)
            nn.Softplus()  # Enforce non-negative outputs
        )

        # Precompute the cosine noise schedule
        self.noise_schedule = cosine_noise_schedule(num_timesteps) * noise_scale

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return x_start + noise * t

    def p_sample(self, x_t, t, context):
        # Predict noise and subtract it to denoise the sample
        noise_pred = self.network(torch.cat((x_t, context), dim=1))  # Context added here
        return x_t - noise_pred

    def compute_loss(self, x, context, noise_schedule):
        # Similar to how Normalizing Flows handle inputs, split data and context
        noise = torch.randn_like(x[:, :self.input_dim])  # Ignore the context in noise generation
        t = torch.randint(0, self.num_timesteps, (x.size(0),), device=x.device).float() / self.num_timesteps
        t = t.view(-1, 1).expand(x.size(0), self.input_dim)  # Shape (batch_size, input_dim)

        # Add noise to the input data (4 dims), excluding the context dimension (5th dim)
        x_t = self.q_sample(x[:, :self.input_dim], t, noise)

        # Concatenate the context with the noisy data
        noise_pred = self.network(torch.cat((x_t, context), dim=1))

        # Compute loss between predicted and actual noise
        return torch.mean((noise_pred - noise) ** 2)

