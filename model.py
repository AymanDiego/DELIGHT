import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms import AffineCouplingTransform, RandomPermutation
from nflows.transforms.base import CompositeTransform
from nflows.nn.nets import ResidualNet


def linear_noise_schedule(timesteps, start=1e-4, end=2e-2):
    return torch.linspace(start, end, timesteps)

def sample_step(model, x, t, condition, alpha_bar):
    noise_pred = model(x, t, condition)
    return (x - noise_pred * (1 - alpha_bar[t]).sqrt()) / alpha_bar[t].sqrt()

def precompute_widths(widths_data, energy_bins):
    """
    Precompute average widths for each energy bin.
    """
    bins = np.linspace(energy_bins[0], energy_bins[1], energy_bins[2])  # [start, stop, num_bins]
    avg_widths = []

    for i in range(len(bins) - 1):
        mask = (widths_data["energies"] >= bins[i]) & (widths_data["energies"] < bins[i + 1])
        avg_widths.append(widths_data.loc[mask, ["ch1", "ch2", "ch3", "ch4"]].mean().values)

    return bins, np.array(avg_widths)

def assign_precomputed_widths(condition, bins, avg_widths):
    """
    Assign precomputed widths based on the energy bin.
    """
    indices = np.digitize(condition.cpu().numpy().flatten(), bins) - 1
    indices = np.clip(indices, 0, len(avg_widths) - 1)  # Ensure indices are within bounds
    return torch.tensor(avg_widths[indices], device=condition.device, dtype=torch.float32)

def diffusion_loss(model, x, condition, noise_schedule, timesteps, bins, avg_widths):
    t = torch.randint(0, timesteps, (x.shape[0],)).to(x.device)
    noise = torch.randn_like(x).to(x.device)

    alpha_bar = torch.cumprod(1 - noise_schedule, dim=0)
    alpha_bar_t = alpha_bar[t].unsqueeze(-1)
    
    # Assign precomputed widths
    widths = assign_precomputed_widths(condition, bins, avg_widths)
    noise *= widths  # Scale noise by widths

    x_noisy = x * alpha_bar_t.sqrt() + noise * (1 - alpha_bar_t).sqrt()
    noise_pred = model(x_noisy, t, condition)
    return ((noise - noise_pred) ** 2).mean()

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.key = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.value = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        queries = self.query(x).unsqueeze(1)  # Shape: (batch_size, 1, dim)
        keys = self.key(x).unsqueeze(1)       # Shape: (batch_size, 1, dim)
        values = self.value(x).unsqueeze(1)   # Shape: (batch_size, 1, dim)

        # Calculate attention weights
        attention_weights = self.softmax(torch.bmm(queries, keys.transpose(1, 2)) / (x.shape[-1] ** 0.5))
        attention_output = torch.bmm(attention_weights, values)
        return attention_output.squeeze(1)


class AttentionDiffusionModel(nn.Module):
    def __init__(self, data_dim, condition_dim, timesteps, device):
        super(AttentionDiffusionModel, self).__init__()
        self.timesteps = timesteps
        self.data_dim = data_dim
        self.condition_dim = condition_dim
        self.device = device  # Store the device

        # Network layers
        self.attention = SelfAttention(data_dim + condition_dim)
        self.fc1 = nn.Linear(data_dim + condition_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, data_dim)

    def forward(self, x, t, condition):
        # Concatenate data with condition
        x = torch.cat([x, condition], dim=-1)

        # Add attention mechanism
        x = self.attention(x)

        # Feedforward layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ModifiedResidualNet(ResidualNet):
    def forward(self, inputs, context=None):
        # Call the parent class's forward method
        outputs = super().forward(inputs, context)
        # Apply Softplus to enforce non-negative outputs
        return nn.Softplus()(outputs)

class ConditionalNormalizingFlowModel(nn.Module):
    def __init__(self, input_dim, context_dim, hidden_dim, num_layers, device):
        super(ConditionalNormalizingFlowModel, self).__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.device = device  # Store the device

        # Base distribution for the 4D subspace (input_dim = 4)
        self.base_distribution = StandardNormal(shape=[input_dim])

        # Create a sequence of Affine Coupling Transforms, handling conditioning on the 5th dimension via the transform network
        transforms = []
        for i in range(num_layers):
            # Alternating mask for coupling layers (splitting the dimensions)
            mask = torch.tensor([i % 2] * (input_dim // 2) + [(i + 1) % 2] * (input_dim - input_dim // 2))  # Keep mask on CPU

            # Define a transform with a custom neural network, using context as an additional input in the transform network
            transforms.append(AffineCouplingTransform(
                mask=mask,
                transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=hidden_dim,
                    context_features=context_dim,  # This allows us to condition on the contex
                    num_blocks=2,
                    activation=nn.Softplus()  # Apply Softplus to ensure positive outputs
                ).to(self.device)
            ))
            transforms.append(RandomPermutation(features=input_dim))  # Randomly permute after each layer

        self.transform = CompositeTransform(transforms).to(self.device)
        self.flow = Flow(self.transform, self.base_distribution).to(self.device)

    def forward(self, x, context):
        # Move x and context to the correct device
        x = x.to(self.device)
        context = context.to(self.device)
        return self.flow.log_prob(x, context)

    def sample(self, num_samples, context):
        # Ensure context is on the correct device
        context = context.to(self.device)

        # Sample from the model
        samples = self.flow.sample(num_samples, context)

        # Apply Softplus to ensure positive outputs
        return torch.nn.functional.softplus(samples)
