import os
import glob
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
import pandas as pd
import argparse
from model import AttentionDiffusionModel, cosine_noise_schedule, sample_step

hep.style.use(hep.style.ATLAS)

# ArgumentParser to handle command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Infer Diffusion Model")
    parser.add_argument('--loss_dir', type=str, default='', help='Specify an extra directory to append for saving files (e.g., "run_01")')
    parser.add_argument('--model_dir', type=str, default='', help='Directory to load the diffusion model checkpoint')
    parser.add_argument('--device', type=str, default='cuda:1', help='Specify the device to run inference on (e.g., cuda:0, cuda:1, cpu)')
    return parser.parse_args()

# Sampling function for the diffusion model
def sample(model, condition, timesteps, data_dim, device):
    x = torch.randn(condition.shape[0], data_dim).to(device, dtype=torch.float32)
    alpha_bar = torch.cumprod(1 - cosine_noise_schedule(timesteps).to(device, dtype=torch.float32), dim=0)

    for t in reversed(range(timesteps)):
        try:
            x = sample_step(model, x, t, condition.to(device, dtype=torch.float32), alpha_bar)
        except ValueError as e:
            print(f"Sampling stopped due to instability at timestep {t}")
            return None  # Return None if instability occurs

    return x

if __name__ == "__main__":
    args = parse_args()  # Parse command-line arguments

    # Base directory for saving outputs
    base_dir = "/web/aratey/public_html/delight/nf/models_DM/"
    save_dir = os.path.join(base_dir, args.loss_dir) if args.loss_dir else base_dir
    os.makedirs(save_dir, exist_ok=True)

    # Set device for model and data
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Define model parameters
    data_dim = 4
    condition_dim = 1
    timesteps = 100

    # Instantiate and load the diffusion model
    df_model = AttentionDiffusionModel(data_dim=data_dim, condition_dim=condition_dim, timesteps=timesteps, device=device).to(device)
    checkpoint = torch.load(f'{args.model_dir}epoch-18.pt', map_location=device)
    df_model.load_state_dict(checkpoint['model'])

    # Switch model to evaluation mode
    df_model.eval()

    # Define target energies
    energies = np.geomspace(10, 1e6, 500)

    for i, e in enumerate(energies):
        if i % 50 != 0:
            continue
        print(f"Loading simulated data corresponding to index {i}")

        sim = None
        for f in glob.glob(f"/ceph/bmaier/delight/ml/nf/data/val/NR_final_{i}_*.npy"):
            if "lin" in f:
                continue
            if sim is None:
                sim = np.load(f)[:, :4]
            else:
                sim = np.concatenate((sim, np.load(f)[:, :4]))

        energy = torch.tensor(np.sum(sim, axis=1).reshape(-1, 1), device=device, dtype=torch.float32)

        with torch.no_grad():
            samples = sample(df_model, energy, timesteps, data_dim, device)
        
        # Check for NaNs or Infs in samples
        if torch.isnan(samples).any() or torch.isinf(samples).any():
            print("NaNs or Infs detected in samples. Skipping plotting for this sample.")
            continue  # Skip plotting this sample if it contains NaN or Inf values

        # Plot and save the histograms for different channels
        fig, ax = plt.subplots(figsize=(7, 6))
        plt.hist(samples[:, 0].cpu().numpy(), histtype='step', bins=15, label='phonon channel (generated)', color='indianred')
        plt.hist(sim[:, 0], histtype='step', bins=15, linestyle='dashed', color='indianred')
        plt.hist(samples[:, 1].cpu().numpy(), histtype='step', bins=15, label='triplet channel (generated)', color='grey')
        plt.hist(sim[:, 1], histtype='step', bins=15, linestyle='dashed', color='grey')
        plt.hist(samples[:, 2].cpu().numpy(), histtype='step', bins=15, label='UV channel (generated)', color='gold')
        plt.hist(sim[:, 2], histtype='step', bins=15, linestyle='dashed', color='gold')
        plt.hist(samples[:, 3].cpu().numpy(), histtype='step', bins=15, label='IR channel (generated)', color='cornflowerblue')
        plt.hist(sim[:, 3], histtype='step', bins=15, linestyle='dashed', color='cornflowerblue')
        plt.text(0.05, 0.90, "Nuclear recoil", transform=ax.transAxes, fontsize=18)
        plt.text(0.05, 0.82, "$E_\mathrm{NR}=%.0f$ eV" % e, transform=ax.transAxes, fontsize=18)
        ax.set_xlabel("$E$ (eV)", labelpad=20)
        ax.set_ylabel("Arbitrary units")
        plt.legend(fontsize=17)
        plt.tight_layout()

        # Save the generated plots to the specified directory
        plt.savefig(f"{save_dir}/gen_{i}.png", bbox_inches='tight', dpi=300)

