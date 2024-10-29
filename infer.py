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
    flow_model.eval()  # Switch to evaluation mode

    # Load normalization parameters if needed
    if args.normalize_energies:
        means, stds = load_normalization_params(args.epoch_dir)

    # Define batch size for sampling to avoid memory overflow
    batch_size = 50

    # Process each simulation file
    for i, f in enumerate(glob.glob("/ceph/aratey/delight/ml/nf/data/NR_final_*.npy")):
        print(f"Processing file {f}")
        sim = np.load(f)[:, :4]
        energy_sum = np.sum(sim, axis=1).reshape(-1, 1)

        # Create tensor to store generated samples
        generated_samples = []

        # Process in batches to avoid memory issues
        for start in range(0, sim.shape[0], batch_size):
            end = min(start + batch_size, sim.shape[0])
            context_batch = torch.tensor(energy_sum[start:end], device=device, dtype=torch.float32)

            # Generate samples for the current batch and move to CPU to free GPU memory
            with torch.no_grad():
                gen_batch = flow_model.sample(num_samples=context_batch.size(0), context=context_batch)
                generated_samples.append(gen_batch.cpu().numpy())

            # Clear CUDA cache to free up memory
            torch.cuda.empty_cache()

        # Concatenate all generated samples
        gen = np.concatenate(generated_samples, axis=0)

        # Apply reverse transformation if normalization was applied
        if args.normalize_energies:
            gen = gen * stds + means
        
        # Reshape `gen` to match `sim` shape
        gen = gen.reshape(-1, 4)

        # Confirm shapes before plotting
        print(f"Shape of reshaped generated data (gen): {gen.shape}")
        print(f"Shape of simulated data (sim): {sim.shape}")

        # Plot histograms for each channel
        fig, ax = plt.subplots(figsize=(7, 6))

        plt.hist(gen[:, 0].ravel(), histtype='step', bins=15, label='phonon channel (generated)', color='indianred')
        plt.hist(sim[:, 0].ravel(), histtype='step', bins=15, linestyle='dashed', label='phonon channel (simulated)', color='indianred')

        plt.hist(gen[:, 1].ravel(), histtype='step', bins=15, label='triplet channel (generated)', color='grey')
        plt.hist(sim[:, 1].ravel(), histtype='step', bins=15, linestyle='dashed', label='triplet channel (simulated)', color='grey')

        plt.hist(gen[:, 2].ravel(), histtype='step', bins=15, label='UV channel (generated)', color='gold')
        plt.hist(sim[:, 2].ravel(), histtype='step', bins=15, linestyle='dashed', label='UV channel (simulated)', color='gold')

        plt.hist(gen[:, 3].ravel(), histtype='step', bins=15, label='IR channel (generated)', color='cornflowerblue')
        plt.hist(sim[:, 3].ravel(), histtype='step', bins=15, linestyle='dashed', label='IR channel (simulated)', color='cornflowerblue')

        plt.text(0.05, 0.90, "Nuclear recoil", transform=ax.transAxes, fontsize=18)
        plt.text(0.05, 0.82, "$E_\\mathrm{NR}=%.0f$ eV" % energy_sum.mean(), transform=ax.transAxes, fontsize=18)
        ax.set_xlabel("$E$ (eV)", labelpad=20)
        ax.set_ylabel("Arbitrary units")
        plt.legend(fontsize=12)
        plt.tight_layout()

        # Save the generated plots to the specified directory
        plt.savefig(f"{save_dir}/gen_{e:.0f}_eV.png", bbox_inches='tight', dpi=300)
