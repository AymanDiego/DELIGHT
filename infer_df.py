import os
import glob
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
import argparse
import corner
from model import AttentionDiffusionModel, linear_noise_schedule, sample_step

hep.style.use(hep.style.ATLAS)

# ArgumentParser to handle command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Infer Diffusion Model")
    parser.add_argument('--loss_dir', type=str, default='', help='Specify an extra directory to append for saving files (e.g., "run_01")')
    parser.add_argument('--model_dir', type=str, default='', help='Directory to load the diffusion model checkpoint')
    parser.add_argument('--device', type=str, default='cuda:1', help='Specify the device to run inference on (e.g., cuda:0, cuda:1, cpu)')
    return parser.parse_args()

# Sampling function
def sample(model, condition, timesteps, data_dim, device):
    # Initialize latent variable
    x = torch.randn(condition.shape[0], data_dim).to(device, dtype=torch.float32)

    # Compute alpha_bar with clamping for stability
    noise_schedule = linear_noise_schedule(timesteps).to(device, dtype=torch.float32)
    alpha_bar = torch.cumprod(1 - noise_schedule, dim=0)

    # Reverse sampling loop
    for t in reversed(range(timesteps)):
        x = sample_step(model, x, t, condition.to(device, dtype=torch.float32), alpha_bar)

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
    timesteps = 25

    # Instantiate and load the diffusion model
    df_model = AttentionDiffusionModel(data_dim=data_dim, condition_dim=condition_dim, timesteps=timesteps, device=device).to(device)
    checkpoint = torch.load(f'{args.model_dir}/epoch-149.pt', map_location=device)
    df_model.load_state_dict(checkpoint['model'])

    # Switch model to evaluation mode
    df_model.eval()

    # Define target energies
    energies = np.geomspace(10, 1e6, 500)

    for i, e in enumerate(energies):
        # Skip energies outside the specified range
        if e < 100000 or e > 1000000:
            continue

        if i % 10 != 0:
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

        energy = np.sum(sim, axis=1).reshape(-1, 1)
        energy = torch.tensor(energy, device=device, dtype=torch.float32)

        # Generate samples
        with torch.no_grad():
            gen = sample(df_model, energy / 1000000, timesteps, data_dim, device)

        energy = energy.detach().cpu().numpy()
        gen = gen.detach().cpu().numpy()

        print(f"Saving plots to: {save_dir}")
        # Plot results
        fig, ax = plt.subplots(figsize=(7, 6))
        plt.hist(gen[:, 0] * energy[0], histtype='step', bins=15, label='phonon channel', color='indianred')
        plt.hist(sim[:, 0], histtype='step', bins=15, linestyle='dashed', color='indianred')
        plt.hist(gen[:, 1] * energy[0], histtype='step', bins=15, label='triplet channel', color='grey')
        plt.hist(sim[:, 1], histtype='step', bins=15, linestyle='dashed', color='grey')
        plt.hist(gen[:, 2] * energy[0], histtype='step', bins=15, label='UV channel', color='gold')
        plt.hist(sim[:, 2], histtype='step', bins=15, linestyle='dashed', color='gold')
        plt.hist(gen[:, 3] * energy[0], histtype='step', bins=15, label='IR channel', color='cornflowerblue')
        plt.hist(sim[:, 3], histtype='step', bins=15, linestyle='dashed', color='cornflowerblue')                                                                                                  
        plt.text(0.05, 0.90, "Nuclear recoil", transform=ax.transAxes, fontsize=18)
        plt.text(0.05, 0.82, "$E_\mathrm{NR}=%.0f$ eV" % energy[0], transform=ax.transAxes, fontsize=18)
        ax.set_xlabel("$E$ (eV)", labelpad=20)
        ax.set_ylabel("Arbitrary units")
        plt.legend(fontsize=17)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/gen_{i}.png", bbox_inches='tight', dpi=300)

        # Generating and saving the corner plot
        print(f"Generating corner plot for index {i}...")

        # Scale the generated data to match the scale of simulated data for comparison
        gen_scaled = gen * energy[0]  # Ensure generated data is scaled correctly

        # Debugging: Print the first 10 rows of scaled and simulated data for verification
        print(f"Simulated Data (First 10 Rows):\n{sim[:10]}")
        print(f"Generated Data (First 10 Rows):\n{gen_scaled[:10]}")

        try:
            # Create the corner plot for simulated data (blue)
            fig_corner = corner.corner(
                sim,
                labels=["Phonon", "Triplet", "UV", "IR"],
                color="blue",
                show_titles=True,
                hist_kwargs={"density": True},
                plot_contours=True,
            )

            # Overlay the generated data (red) on the same corner plot
            corner.corner(
                gen_scaled,
                fig=fig_corner,
                color="red",
                hist_kwargs={"density": True},
                plot_contours=True,
            )

            # Save the combined corner plot
            plt.savefig(f"{save_dir}/corner_combined_{i}.png", bbox_inches="tight", dpi=300)
            print(f"Corner plot saved to: {save_dir}/corner_combined_{i}.png")

        except Exception as e:
            print(f"Error during corner plot generation for index {i}: {e}")

