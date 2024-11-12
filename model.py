import torch
import torch.nn as nn
from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms import AffineCouplingTransform, RandomPermutation
from nflows.transforms.base import CompositeTransform
from nflows.nn.nets import ResidualNet
import math
import torch

def cosine_noise_schedule(timesteps, s=0.008, epsilon=1e-6):
    """Cosine noise schedule function with stability enhancements."""
    steps = torch.arange(timesteps + 1, dtype=torch.float32)
    alpha_bar = torch.cos((steps / timesteps + s) / (1 + s) * math.pi / 2) ** 2

    # Add epsilon to prevent division issues
    alpha_ratio = (alpha_bar[:-1] + epsilon) / (alpha_bar[1:] + epsilon)
    return alpha_ratio

def sample_step_inference(model, x, t, condition, alpha_bar):
    noise_pred = model(x, t, condition)
    epsilon = 1e-5  # Small constant to stabilize
    # Modify to avoid near-zero values in alpha_bar[t] and (1 - alpha_bar[t])
    x_new = (x - noise_pred * (1 - alpha_bar[t] + epsilon).sqrt()) / (alpha_bar[t] + epsilon).sqrt()

    # Optional: Clip values to ensure stability
    #x_new = torch.clamp(x_new, min=-1e2, max=1e2)  # Adjust these bounds as needed
    return x_new

def sample_step(model, x, t, condition, alpha_bar, epsilon=1e-5):
    # Alpha bar values for stability
    alpha_bar_t = alpha_bar[t].unsqueeze(-1)  # Shape: (batch_size, 1)

    # Model prediction and diagnostics
    noise_pred = model(x, t, condition)
    if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
        print("NaN or Inf detected in noise_pred at step:", t)
        print("Noise_pred values:", noise_pred)
        raise ValueError("NaN or Inf in model prediction (noise_pred)")

    # Diffuse step computation with enhanced stability
    x_new = (x - noise_pred * torch.sqrt(torch.clamp(1 - alpha_bar_t, min=epsilon))) / torch.sqrt(torch.clamp(alpha_bar_t, min=epsilon))
    x_new = torch.clamp(x_new, min=-1e6, max=1e6)  # Prevent extreme values in x_new

    if torch.isnan(x_new).any() or torch.isinf(x_new).any():
        print("NaN or Inf detected in x_new at step:", t)
        print("x_new values:", x_new)
        raise ValueError("NaN or Inf in x_new during sampling")

    return x_new

def diffusion_loss(model, x, condition, noise_schedule, timesteps, epsilon=1e-6, scale_factor=0.001):
    # Ensure input data is finite
    if torch.isnan(x).any() or torch.isinf(x).any():
        raise ValueError("NaN or Inf detected in input x")
    
    # Scale down the noise schedule to reduce potential large changes in x_noisy
    noise_schedule = noise_schedule * scale_factor

    # Choose a random timestep and add noise
    t = torch.randint(0, timesteps, (x.shape[0],)).to(x.device)
    noise = torch.randn_like(x).to(x.device)
    alpha_bar = torch.cumprod(1 - noise_schedule + epsilon, dim=0)  # Add epsilon for stability

    # Ensure alpha_bar[t] has an extra dimension to match x and noise and clamp it
    alpha_bar_t = alpha_bar[t].unsqueeze(-1)  # Shape: (batch_size, 1)
    alpha_bar_t = torch.clamp(alpha_bar_t, min=1e-4, max=1)  # Prevent very small or extreme values

    # Forward diffuse x to add noise with epsilon
    x_noisy = x * (alpha_bar_t + epsilon).sqrt() + noise * ((1 - alpha_bar_t) + epsilon).sqrt()

    # Debugging: Clamp x_noisy to a safe range to prevent extreme values
    x_noisy = torch.clamp(x_noisy, -1e6, 1e6)


    # Log details if NaN or Inf are detected in x_noisy
    if torch.isnan(x_noisy).any() or torch.isinf(x_noisy).any():
        print("NaNs/Infs detected in x_noisy.")
        print("alpha_bar_t:", alpha_bar_t)
        print("x values:", x)
        print("x_noisy values:", x_noisy)
        raise ValueError("NaN or Inf detected in x_noisy after clamping.")

    noise_pred = model(x_noisy, t, condition)

    # Check for NaN or Inf in noise_pred before calculating the loss
    if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
        print("NaNs/Infs detected in model output (noise_pred).")
        print("noise_pred values:", noise_pred)
        raise ValueError("NaN or Inf detected in model output.")

    loss = ((noise - noise_pred) ** 2).mean()

    return loss

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Xavier initialization for weights
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # Initialize bias to 0

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, dim)
        queries = self.query(x).unsqueeze(1)  # Shape: (batch_size, 1, dim)
        keys = self.key(x).unsqueeze(1)       # Shape: (batch_size, 1, dim)
        values = self.value(x).unsqueeze(1)   # Shape: (batch_size, 1, dim)

        # Calculate attention weights
        attention_weights = self.softmax(torch.bmm(queries, keys.transpose(1, 2)) / (x.shape[-1] ** 0.5))
        attention_output = torch.bmm(attention_weights, values)

        # Remove the extra dimension and return output
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

        # Increased hidden dimension size and added an extra layer for more capacity
        self.fc1 = nn.Linear(data_dim + condition_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, data_dim)

        # Initialize Softplus to enforce positive outputs
        self.softplus = nn.Softplus()

    def forward(self, x, t, condition):
        # Concatenate data with condition
        x = torch.cat([x, condition], dim=-1)

        # Add attention mechanism
        x = self.attention(x)

        # Feedforward layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)

        # Apply Softplus to the final output layer to enforce non-negative outputs
        return self.softplus(x)

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

