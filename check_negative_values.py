import torch
import numpy as np
from model import ConditionalNormalizingFlowModel

def check_for_negative_values(flow_model, num_samples, context, device):
    # Ensure context is on the correct device
    context = torch.tensor(context, dtype=torch.float32).to(device)

    # Sample from the Normalizing Flow model
    generated_samples = flow_model.sample(num_samples, context)
    
    # Move samples back to CPU and convert to numpy for further analysis
    generated_samples_np = generated_samples.cpu().detach().numpy()

    # Check if there are any negative values
    min_value = np.amin(generated_samples_np)
    if min_value < 0:
        print(f"Negative values found! Minimum value: {min_value}")
    else:
        print("No negative values found in the generated samples.")

    return generated_samples_np

if __name__ == "__main__":
    # Set up model parameters
    input_dim = 4
    context_dim = 1
    hidden_dim = 128  # Adjust based on the model used during training
    num_layers = 8
    num_samples = 10000  # Number of samples you want to generate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load your trained Normalizing Flow model
    flow_model = ConditionalNormalizingFlowModel(input_dim=input_dim, context_dim=context_dim, hidden_dim=hidden_dim, num_layers=num_layers, device=device).to(device)
    
    checkpoint = torch.load('models/run_7/epoch-300.pt', map_location=device)  # Update this to the correct model file if needed
    flow_model.load_state_dict(checkpoint['model'])  # Ensure the key 'model' is correct for loading

    # Set model to evaluation mode
    flow_model.eval()

    # Example energy values for which you want to check negative values
    energies = np.geomspace(10, 1e6, 500)

    # Loop through energies to generate samples and check for negative values
    for i, e in enumerate(energies):
        if i % 50 != 0:
            continue

        print(f"Checking for negative values for {e} eV (index {i})")
        fixed_value_5th_dim = np.array([[float(e)]])  # Context as a 5th dimension

        # Check for negative values in the generated samples
        check_for_negative_values(flow_model, num_samples=num_samples, context=fixed_value_5th_dim, device=device)

