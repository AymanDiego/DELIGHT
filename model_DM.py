import torch
import torch.nn as nn
import torch.nn.functional as F
from nflows.nn.nets import ResidualNet

# Subclassing ResidualNet to apply Softplus to its output
class ModifiedResidualNet(ResidualNet):
    def forward(self, inputs, context=None):
        outputs = super().forward(inputs, context)  # Call the original forward method
        return nn.Softplus()(outputs)  # Apply Softplus activation to enforce non-negativity

# Update the DiffusionModel to use ModifiedResidualNet
class DiffusionModel(nn.Module):
    def __init__(self, input_dim, num_timesteps, device):
        super(DiffusionModel, self).__init__()
        self.input_dim = input_dim
        self.num_timesteps = num_timesteps
        self.device = device

        # Use the modified ResidualNet
        self.network = ModifiedResidualNet(
            in_features=input_dim + 1,  # Input features + context
            out_features=input_dim,  # Output should match input features
            hidden_features=128,  # Number of hidden units
            context_features=None,  # No additional context (other than energy dimension)
            num_blocks=2,  # Number of residual blocks
        )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return x_start + noise * t

    def p_sample(self, x_t, t, context):
        # Predict noise and subtract it to denoise the sample
        noise_pred = self.network(torch.cat((x_t, context), dim=1))  # Context added here
        return x_t - noise_pred

    def compute_loss(self, x, context):
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

