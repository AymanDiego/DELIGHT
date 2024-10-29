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

    # Define target energies
    target_energies = [10, 32, 100, 318, 1009, 3199, 10139, 32138, 101863, 322863]

    # Process simulation files for specific energies
    for i, e in enumerate(target_energies):
        print(f"Processing energy level: {e} eV")

        # Load files corresponding to the target energy levels
        matching_files = glob.glob(f"/ceph/aratey/delight/ml/nf/data/NR_final_{i}_*.npy")
        if not matching_files:
            print(f"No matching files found for energy {e} eV.")
            continue

        # Assume the first matching file contains the relevant data
        sim = np.concatenate([np.load(f)[:, :4] for f in matching_files], axis=0)
        energy_sum = np.sum(sim, axis=1).reshape(-1, 1)  # Compute total energy per event

        # Generate samples using the flow model
        context = torch.tensor(energy_sum, device=device, dtype=torch.float32)
        batch_size = 50
        generated_samples = []

        for start_idx in range(0, sim.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, sim.shape[0])
            batch_context = context[start_idx:end_idx]
            if batch_context.size(0) > 0:
                batch_samples = flow_model.sample(num_samples=batch_context.size(0), context=batch_context)
                generated_samples.append(batch_samples.cpu().detach())

        gen = torch.cat(generated_samples, dim=0).numpy()

        # Apply reverse transformation if normalization was applied
        if args.normalize_energies:
            gen = gen * stds + means

        # Reshape `gen` to match `sim` shape
        gen = gen.reshape(-1, 4)

        # Plot and save histograms for each channel
        fig, ax = plt.subplots(figsize=(7, 6))
        plt.hist(gen[:, 0], histtype='step', bins=15, label='phonon channel (generated)', color='indianred')
        plt.hist(sim[:, 0], histtype='step', bins=15, linestyle='dashed', color='indianred')
        plt.hist(gen[:, 1], histtype='step', bins=15, label='triplet channel (generated)', color='grey')
        plt.hist(sim[:, 1], histtype='step', bins=15, linestyle='dashed', color='grey')
        plt.hist(gen[:, 2], histtype='step', bins=15, label='UV channel (generated)', color='gold')
        plt.hist(sim[:, 2], histtype='step', bins=15, linestyle='dashed', color='gold')
        plt.hist(gen[:, 3], histtype='step', bins=15, label='IR channel (generated)', color='cornflowerblue')
        plt.hist(sim[:, 3], histtype='step', bins=15, linestyle='dashed', color='cornflowerblue')

        plt.text(0.05, 0.90, "Nuclear recoil", transform=ax.transAxes, fontsize=18)
        plt.text(0.05, 0.82, "$E_\mathrm{NR}=%.0f$ eV" % e, transform=ax.transAxes, fontsize=18)
        ax.set_xlabel("$E$ (eV)", labelpad=20)
        ax.set_ylabel("Arbitrary units")
        plt.legend(fontsize=17)
        plt.tight_layout()

        # Save the plot
        plt.savefig(f"{save_dir}/gen_{e:.0f}_eV.png", bbox_inches='tight', dpi=300)
