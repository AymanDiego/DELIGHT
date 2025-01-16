import os
import glob
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
import numpy as np
import torch
import argparse
import time  # Importing time for runtime measurement
import corner
from model import AttentionDiffusionModel, linear_noise_schedule, sample_step, precompute_widths, assign_precomputed_widths

hep.style.use(hep.style.ATLAS)

# ArgumentParser to handle command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Infer Diffusion Model")
    parser.add_argument('--loss_dir', type=str, default='', help='Specify an extra directory to append for saving files (e.g., "run_01")')
    parser.add_argument('--model_dir', type=str, default='', help='Directory to load the diffusion model checkpoint')
    parser.add_argument('--device', type=str, default='cuda:1', help='Specify the device to run inference on (e.g., cuda:0, cuda:1, cpu)')
    return parser.parse_args()

# Sampling function with channel-specific noise
def sample_with_denoising(model, condition, timesteps, data_dim, device, bins, avg_widths):
    x = torch.randn(condition.shape[0], data_dim).to(device, dtype=torch.float32)
    noise_schedule = linear_noise_schedule(timesteps).to(device, dtype=torch.float32)
    alpha_bar = torch.cumprod(1 - noise_schedule, dim=0)

    for t in reversed(range(timesteps)):
        t_tensor = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
        widths = assign_precomputed_widths(condition, bins, avg_widths)
        scaled_widths = widths * 100
        noise = scaled_widths * torch.randn_like(x).to(device)
        noise_pred = model(x, t_tensor, condition)
        x = (x - noise_pred * (1 - alpha_bar[t]).sqrt()) / alpha_bar[t].sqrt()
        x = x + noise * (1 - alpha_bar[t]).sqrt()

    return x

if __name__ == "__main__":
    args = parse_args()
    base_dir = "/web/aratey/public_html/delight/nf/models_DM/"
    save_dir = os.path.join(base_dir, args.loss_dir) if args.loss_dir else base_dir
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    data_dim, condition_dim, timesteps = 4, 1, 25

    # Load widths and bins
    widths_data = pd.read_csv("widths.csv")
    energy_bins = [0, 1000000, 100]
    bins, avg_widths = precompute_widths(widths_data, energy_bins)

    # Load the model
    df_model = AttentionDiffusionModel(data_dim=data_dim, condition_dim=condition_dim, timesteps=timesteps, device=device).to(device)
    checkpoint = torch.load(f'{args.model_dir}/epoch-39.pt', map_location=device)
    df_model.load_state_dict(checkpoint['model'])
    df_model.eval()

    # Specific energies for runtime measurement
    specific_energies = np.array([10., 100., 1.e3, 1.e4, 1.e5, 1.e6])

    # Measure runtime for specific energies
    runtime_log = []
    for energy in specific_energies:
        fixed_value_tensor = torch.tensor([[energy]], device=device, dtype=torch.float32)
        start_time = time.time()
        with torch.no_grad():
            _ = sample_with_denoising(df_model, fixed_value_tensor / 1e6, timesteps, data_dim, device, bins, avg_widths)
        end_time = time.time()
        runtime_log.append((energy, end_time - start_time))
        print(f"Energy: {energy} eV, Runtime: {end_time - start_time:.6f} s")

    # Save runtime log to CSV
    runtime_df = pd.DataFrame(runtime_log, columns=["Energy (eV)", "Runtime (s)"])
    runtime_df.to_csv(f"{save_dir}/runtime_log_specific.csv", index=False)
    print(f"Runtime log for specific energies saved to {save_dir}/runtime_log_specific.csv.")

    energies = np.geomspace(10, 1e6, 500)
    for i, e in enumerate(energies):
        if e < 0 or e > 1000000 or i % 10 != 0:
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
        energy_tensor = torch.tensor(energy, device=device, dtype=torch.float32)

        with torch.no_grad():
            gen = sample_with_denoising(df_model, energy_tensor / 1000000, timesteps, data_dim, device, bins, avg_widths)

        energy = energy_tensor.detach().cpu().numpy()
        gen = gen.detach().cpu().numpy()

        # Debugging: Log data ranges
        print(f"Simulated data range: Min={sim.min()}, Max={sim.max()}")
        print(f"Generated data range: Min={gen.min()}, Max={gen.max()}")

        print(f"Saving plots to: {save_dir}")

        # Original `gen_{i}.png` plots
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

        # 2D Histogram for UV vs. Phonon
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.hist2d(sim[:, 2], sim[:, 0], bins=50, cmap='Blues', label='Simulated')
        plt.hist2d(gen[:, 2] * energy[0], gen[:, 0] * energy[0], bins=50, cmap='Reds', alpha=0.5, label='Generated')
        plt.colorbar(label="Counts")
        plt.xlabel("UV Channel (eV)")
        plt.ylabel("Phonon Channel (eV)")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/uv_vs_phonon_{i}.png", bbox_inches="tight", dpi=300)
        plt.close()

        # Corner plot for comparison
        print(f"Generating corner plot for index {i}...")
        gen_scaled = gen * energy[0]
        try:
            fig_corner = corner.corner(
                sim,
                labels=["Phonon", "Triplet", "UV", "IR"],
                color="blue",
                show_titles=True,
                hist_kwargs={"density": True},
                plot_contours=True,
            )
            corner.corner(
                gen_scaled,
                fig=fig_corner,
                color="red",
                hist_kwargs={"density": True},
                plot_contours=True,
            )
            plt.savefig(f"{save_dir}/corner_combined_{i}.png", bbox_inches="tight", dpi=300)
            print(f"Corner plot saved to: {save_dir}/corner_combined_{i}.png")
        except Exception as e:
            print(f"Error during corner plot generation for index {i}: {e}")

