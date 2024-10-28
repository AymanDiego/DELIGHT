import os
import glob
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.ATLAS)
import numpy as np
import torch
import pandas as pd
import argparse  # This is the missing import
from model import ConditionalNormalizingFlowModel  # Importing the model from model.py

# ArgumentParser to handle command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Infer Conditional Normalizing Flow Model")

    # Add argument for generated plots
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
    hidden_dim = 64
    num_layers = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate the model architecture
    flow_model = ConditionalNormalizingFlowModel(input_dim, context_dim, hidden_dim, num_layers, device).to(device)

    # Load the saved model weights
    checkpoint = torch.load(f'models/{args.epoch_dir}/epoch-300.pt', map_location=device)
    flow_model.load_state_dict(checkpoint['model'])  # Ensure key matches the saved model

    # Switch to evaluation mode
    flow_model.eval()

    # Example: Generate samples
    energies = np.geomspace(10, 1e6, 500)

    for i, e in enumerate(energies):
        if i % 50 != 0:
            continue

        print(f"Loading simulated data corresponding to index {i}")

        # Loading the simulated data files for the given energy index
        for f in glob.glob(f"/ceph/aratey/delight/ml/nf/data/NR_final_{i}_*.npy"):
            sim = np.load(f)[:, :4]

        print(f"Generating samples for {e} eV (index {i})")

        # Generate samples using the flow model
        fixed_value_5th_dim = torch.tensor([[float(e)]], device=device)
        gen = flow_model.sample(num_samples=sim.shape[0], context=fixed_value_5th_dim)
        gen = np.squeeze(gen.cpu().detach().numpy(), axis=0)

        # Apply reverse transformation if normalization was applied
        if args.normalize_energies:
            gen = gen * stds + means  # Reverse transformation: (normalized value * std) + mean

        # Loop through each channel (0 to 3) and print the values of gen[:, *]
        for channel in range(4):
            print(f"Values for channel {channel}:")
            print(gen[:, channel])

        # Plot and save the histograms for different channels
        fig, ax = plt.subplots(figsize=(7, 6))
        plt.hist(gen[:, 0], histtype='step', bins=15, label='phonon channel', color='indianred')
        plt.hist(sim[:, 0], histtype='step', bins=15, linestyle='dashed', color='indianred')
        plt.hist(gen[:, 1], histtype='step', bins=15, label='triplet channel', color='grey')
        plt.hist(sim[:, 1], histtype='step', bins=15, linestyle='dashed', color='grey')
        plt.hist(gen[:, 2], histtype='step', bins=15, label='UV channel', color='gold')
        plt.hist(sim[:, 2], histtype='step', bins=15, linestyle='dashed', color='gold')
        plt.hist(gen[:, 3], histtype='step', bins=15, label='IR channel', color='cornflowerblue')
        plt.hist(sim[:, 3], histtype='step', bins=15, linestyle='dashed', color='cornflowerblue')
        plt.text(0.05, 0.90, "Nuclear recoil", transform=ax.transAxes, fontsize=18)
        plt.text(0.05, 0.82, "$E_\mathrm{NR}=%.0f$ eV" % e, transform=ax.transAxes, fontsize=18)
        ax.set_xlabel("$E$ (eV)", labelpad=20)
        ax.set_ylabel("Arbitrary units")
        plt.legend(fontsize=17)
        plt.tight_layout()

        # Save the generated plots to the specified directory
        plt.savefig(f"{save_dir}/gen_{i}.png", bbox_inches='tight', dpi=300)

