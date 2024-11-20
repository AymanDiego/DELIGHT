import torch
import torch.nn as nn
import math

def cosine_noise_schedule(timesteps, s=0.008):
    """Cosine noise schedule."""
    steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float32)
    alpha_bar = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    return alpha_bar[:-1] / alpha_bar[1:]

def linear_noise_schedule(timesteps):
    """Linear noise schedule."""
    return torch.linspace(1, 0, timesteps, dtype=torch.float32)

def constant_noise_schedule(timesteps, scale=0.1):
    """Constant noise schedule."""
    return torch.full((timesteps,), scale, dtype=torch.float32)

class DiffusionModel(nn.Module):
    def __init__(self, input_dim, num_timesteps, device, noise_schedule=None):
        super(DiffusionModel, self).__init__()
        self.input_dim = input_dim
        self.num_timesteps = num_timesteps
        self.device = device

        # Define the architecture for the network
        self.network = nn.Sequential(
            nn.Linear(input_dim + 1, 128),  # Add context (energy) into the input
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),  # Output matches the input data without context (4 dims)
            nn.Softplus()  # Enforce non-negative outputs
        )

        # Precompute or store the noise schedule
        self.noise_schedule = noise_schedule.to(device) if noise_schedule is not None else None

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return x_start + noise * t

    def p_sample(self, x_t, t, context):
        # Predict noise and subtract it to denoise the sample
        noise_pred = self.network(torch.cat((x_t, context), dim=1))  # Context added here
        return x_t - noise_pred

    def compute_loss(self, x, context, noise_schedule):
        # Generate random noise
        noise = torch.randn_like(x[:, :self.input_dim])  # Exclude the context dimension
        t = torch.randint(0, self.num_timesteps, (x.size(0),), device=x.device).float()
        t = t.view(-1, 1).expand(x.size(0), self.input_dim)  # Match dimensions

        if noise_schedule is not None:
            # Ensure noise_schedule is on the same device as t
            alpha_bar = (1 - t / self.num_timesteps + noise_schedule[t.long()]).sqrt()
            # Add noise to input data
            x_noisy = x[:, :self.input_dim] * alpha_bar + noise * (1 - alpha_bar).sqrt()
        else:
            # No noise added
            x_noisy = x[:, :self.input_dim]

        # Concatenate context with noisy data
        noise_pred = self.network(torch.cat((x_noisy, context), dim=1))

        # Compute loss between predicted and actual noise
        return torch.mean((noise_pred - noise) ** 2)

