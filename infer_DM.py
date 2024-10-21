import torch
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import glob
import os  # This is needed for the path operations
import argparse  # This is the missing import
from model_DM import DiffusionModel

hep.style.use(hep.style.ATLAS)

# ArgumentParser to handle command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Infer Diffusion Model")

    # Add argument for generated plots
    parser.add_argument('--loss_dir', type=str, default='', help='Specify an extra directory to append for saving files (e.g., "run_01")')
    parser.add_argument('--epoch_dir', type=str, default='', help='Directory to load dm_epoch_300.pt')

    return parser.parse_args()

# Function to perform reverse diffusion sampling
def reverse_diffusion(diffusion_model, num_samples, context, num_timesteps, device):
    # Start from noise
    x_t = torch.randn(num_samples, diffusion_model.input_dim, device=device)

    for t in reversed(range(num_timesteps)):
        t_tensor = torch.tensor([t], device=device)
        x_t = diffusion_model.p_sample(x_t, t_tensor, context)  # Use reverse diffusion

    return x_t

if __name__ == "__main__":
    args = parse_args()  # Parse command-line arguments

    # Base directory
    base_dir = "/web/aratey/public_html/delight/nf/models_DM/"

    # Append loss_dir if provided
    if args.loss_dir:
        save_dir = os.path.join(base_dir, args.loss_dir)
    else:
        save_dir = base_dir

    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set up model parameters
    input_dim = 4  # Make sure this matches the model's trained input dimension
    num_timesteps = 1000
    num_samples = 10000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load your trained diffusion model
    diffusion_model = DiffusionModel(input_dim=input_dim, num_timesteps=num_timesteps, device=device).to(device)

    # Load the checkpoint. Ensure the key matches the saved model during training.
    checkpoint = torch.load(f'models_DM/{args.epoch_dir}/dm_epoch_300.pt', map_location=device)
    diffusion_model.load_state_dict(checkpoint)  # Match the state_dict key to 'model'

    # Set model to evaluation mode
    diffusion_model.eval()

    # Example energy values for which you want to generate samples
    energies = np.geomspace(10, 1e6, 500)

    for i, e in enumerate(energies):
        if i % 50 != 0:
            continue

        print(f"Loading simulated data corresponding to index {i}")

        for f in glob.glob(f"/ceph/aratey/delight/ml/nf/data/NR_final_{i}_*.npy"):
            sim = None
            if sim is None:
                sim = np.load(f)[:, :4]  # Ensure it loads only the first 4 dimensions (features)
            else:
                sim = np.concatenate((sim, np.load(f)[:, :4]))

        # Generate samples using reverse diffusion
        print(f"Generating samples for {e} eV (index {i})")

        # Create context tensor for the given energy, matching the context size from training
        fixed_value_5th_dim = torch.tensor([[float(e)]], device=device).expand(num_samples, 1)

        generated_samples = reverse_diffusion(diffusion_model, num_samples=num_samples, context=fixed_value_5th_dim, num_timesteps=num_timesteps, device=device)
        generated_samples = generated_samples.cpu().detach().numpy()

        # Plot and compare the generated samples to real simulation data
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

