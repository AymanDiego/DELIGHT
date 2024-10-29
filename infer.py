import os
import glob
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.ATLAS)
import numpy as np
import torch
import pandas as pd
import argparse
from model import ConditionalNormalizingFlowModel

# ArgumentParser to handle command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Infer Conditional Normalizing Flow Model")
    parser.add_argument('--loss_dir', type=str, default='', help='Specify an extra directory to append for saving files (e.g., "run_01")')
    parser.add_argument('--epoch_dir', type=str, default='', help='Directory to load dm_epoch_300.pt')
    parser.add_argument('--normalize_energies', action='store_true', help='Flag to re-transform normalized energies to original scale')
    return parser.parse_args()

# Function to load normalization parameters
def load_normalization_params(model_dir):
    params_df = pd.read_csv(f'models/{args.epoch_dir}/normalization_params.csv')
    means = params_df["means"].values
    stds = params_df["stds"].values
    return means, stds

if __name__ == "__main__":
    args = parse_args()  # Parse command-line arguments

    # Base directory
    base_dir = "/web/aratey/public_html/delight/nf/models_nf/"

    # Append loss_dir if provided
    if args.loss_dir:
        save_dir = os.path.join(base_dir, args.loss_dir)
    else:
        save_dir = base_dir

    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set up the model
    input_dim = 4
    context_dim = 1
    hidden_dim = 256
    num_layers = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate the model architecture
    flow_model = ConditionalNormalizingFlowModel(input_dim, context_dim, hidden_dim, num_layers, device).to(device)

    # Load the saved model weights
    checkpoint = torch.load(f'models/{args.epoch_dir}/epoch-300.pt', map_location=device)
    flow_model.load_state_dict(checkpoint['model'])

    # Switch to evaluation mode
    flow_model.eval()

    # Load normalization parameters if needed
    if args.normalize_energies:
        means, stds = load_normalization_params(args.epoch_dir)

    # Process simulation files
    for i, f in enumerate(glob.glob("/ceph/aratey/delight/ml/nf/data/NR_final_*.npy")):
        print(f"Processing file {f}")

        # Load simulated data and calculate total energy as the sum across all channels
        sim = np.load(f)[:, :4]
        energy_sum = np.sum(sim, axis=1).reshape(-1, 1)  # Compute total energy per event

        # Use calculated energy as context
        fixed_value_5th_dim = torch.tensor(energy_sum, device=device, dtype=torch.float32)

        # Generate samples using the flow model
        # Define a batch size suitable for your GPU capacity
        batch_size = 100  
        generated_samples = []

        # Generate samples in batches, with each batch potentially of different size
        for start_idx in range(0, sim.shape[0], batch_size):
            # Define end index to not exceed sim.shape[0]
            end_idx = min(start_idx + batch_size, sim.shape[0])
    
            # Extract the context for the batch
            batch_context = fixed_value_5th_dim[start_idx:end_idx]

            # Generate samples only if the batch is non-empty
            if batch_context.size(0) > 0:
                batch_samples = flow_model.sample(num_samples=batch_context.size(0), context=batch_context)
        
                # Move samples to CPU and store in the list
                generated_samples.append(batch_samples.cpu().detach())

        # Ensure all generated samples have the same dimension across batches
        min_size = min(sample.size(0) for sample in generated_samples)
        generated_samples = [sample[:min_size] for sample in generated_samples]  # Truncate to smallest batch size

        # Concatenate all batches to form final tensor
        gen = torch.cat(generated_samples, dim=0).numpy()


        # Apply reverse transformation if normalization was applied
        if args.normalize_energies:
            gen = gen * stds + means

        # Loop through each channel (0 to 3) and print the values of gen[:, *]
        for channel in range(4):
            print(f"Values for channel {channel}:")
            print(gen[:, channel])

        # Plot and save the histograms for different channels
        fig, ax = plt.subplots(figsize=(7, 6))
        # Ensure we plot each channel separately as a 1D array
        plt.hist(gen[:, 0].ravel(), histtype='step', bins=15, label='phonon channel', color='indianred')
        plt.hist(sim[:, 0].ravel(), histtype='step', bins=15, linestyle='dashed', color='indianred')
        plt.hist(gen[:, 1].ravel(), histtype='step', bins=15, label='triplet channel', color='grey')
        plt.hist(sim[:, 1].ravel(), histtype='step', bins=15, linestyle='dashed', color='grey')
        plt.hist(gen[:, 2].ravel(), histtype='step', bins=15, label='UV channel', color='gold')
        plt.hist(sim[:, 2].ravel(), histtype='step', bins=15, linestyle='dashed', color='gold')
        plt.hist(gen[:, 3].ravel(), histtype='step', bins=15, label='IR channel', color='cornflowerblue')
        plt.hist(sim[:, 3].ravel(), histtype='step', bins=15, linestyle='dashed', color='cornflowerblue') 

        plt.text(0.05, 0.90, "nuclear recoil", transform=ax.transAxes, fontsize=18)
        plt.text(0.05, 0.82, "$E_\mathrm{NR}=%.0f$ eV" % energy_sum.mean(), transform=ax.transAxes, fontsize=18)
        ax.set_xlabel("$E$ (eV)", labelpad=20)
        ax.set_ylabel("Arbitrary units")
        plt.legend(fontsize=17)
        plt.tight_layout()

        # Save the generated plots to the specified directory
        plt.savefig(f"{save_dir}/gen_{i}.png", bbox_inches='tight', dpi=300)

