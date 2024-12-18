import os
import glob
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
import argparse
from model_DM import DiffusionModel

hep.style.use(hep.style.ATLAS)

# ArgumentParser to handle command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Infer Diffusion Model")
    parser.add_argument('--loss_dir', type=str, default='', help='Specify an extra directory to append for saving files (e.g., "run_01")')
    parser.add_argument('--model_dir', type=str, default='', help='Directory to load the diffusion model checkpoint')
    parser.add_argument('--device', type=str, default='cuda:1', help='Specify the device to run inference on (e.g., cuda:0, cuda:1, cpu)')
    return parser.parse_args()

# Function to perform reverse diffusion sampling
def reverse_diffusion(diffusion_model, num_samples, context, num_timesteps, device):
    x_t = torch.randn(num_samples, diffusion_model.input_dim, device=device)  # Start from noise
    for t in reversed(range(num_timesteps)):
        t_tensor = torch.tensor([t], device=device)
        x_t = diffusion_model.p_sample(x_t, t_tensor, context)  # Reverse diffusion step
    return x_t

if __name__ == "__main__":
    args = parse_args()  # Parse command-line arguments

    # Base directory for saving outputs
    base_dir = "/web/aratey/public_html/delight/nf/models_DM/DM_old"
    save_dir = os.path.join(base_dir, args.loss_dir) if args.loss_dir else base_dir
    os.makedirs(save_dir, exist_ok=True)

    # Set device for model and data
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Define model parameters
    input_dim = 4
    num_timesteps = 1000
    num_samples = 1000

    # Instantiate and load the diffusion model
    diffusion_model = DiffusionModel(input_dim=input_dim, num_timesteps=num_timesteps, device=device).to(device)
    checkpoint = torch.load(f'{args.model_dir}/dm_epoch_99.pt', map_location=device)
    diffusion_model.load_state_dict(checkpoint)

    # Switch model to evaluation mode
    diffusion_model.eval()

    # Define target energies
    energies = np.geomspace(10, 1e6, 500)

    for i, e in enumerate(energies):
        # Skip energies outside the specified range
        if e < 1000 or e > 10000:
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

        # Calculate energy
        energy = np.sum(sim, axis=1).reshape(-1, 1)
        energy = torch.tensor(energy, device=device, dtype=torch.float32)

        # Generate samples using reverse diffusion
        print(f"Generating samples for {e:.2f} eV (index {i})")
        fixed_value_5th_dim = energy / 1000000  # Normalize energy
        with torch.no_grad():
            generated_samples = reverse_diffusion(diffusion_model, num_samples=num_samples, context=fixed_value_5th_dim, num_timesteps=num_timesteps, device=device)
        generated_samples = generated_samples.cpu().detach().numpy()

        # Plot results
        fig, ax = plt.subplots(figsize=(7, 6))
        plt.hist(generated_samples[:, 0], histtype='step', bins=15, label='phonon channel', color='indianred')
        plt.hist(sim[:, 0], histtype='step', bins=15, linestyle='dashed', color='indianred')
        plt.hist(generated_samples[:, 1], histtype='step', bins=15, label='triplet channel', color='grey')
        plt.hist(sim[:, 1], histtype='step', bins=15, linestyle='dashed', color='grey')
        plt.hist(generated_samples[:, 2], histtype='step', bins=15, label='UV channel', color='gold')
        plt.hist(sim[:, 2], histtype='step', bins=15, linestyle='dashed', color='gold')
        plt.hist(generated_samples[:, 3], histtype='step', bins=15, label='IR channel', color='cornflowerblue')
        plt.hist(sim[:, 3], histtype='step', bins=15, linestyle='dashed', color='cornflowerblue')
        plt.text(0.05, 0.90, "Nuclear recoil", transform=ax.transAxes, fontsize=18)
        plt.text(0.05, 0.82, "$E_\mathrm{NR}=%.0f$ eV" % e, transform=ax.transAxes, fontsize=18)
        ax.set_xlabel("$E$ (eV)", labelpad=20)
        ax.set_ylabel("Arbitrary units")
        plt.legend(fontsize=17)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/gen_{i}_DM.png", bbox_inches='tight', dpi=300)

