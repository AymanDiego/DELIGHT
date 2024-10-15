import torch
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import glob
from model_DM import DiffusionModel

hep.style.use(hep.style.ATLAS)

# Function to perform reverse diffusion sampling
def reverse_diffusion(diffusion_model, num_samples, context, num_timesteps, device):
    # Start from noise
    x_t = torch.randn(num_samples, diffusion_model.input_dim, device=device)
    
    for t in reversed(range(num_timesteps)):
        t_tensor = torch.tensor([t], device=device)
        x_t = diffusion_model.p_sample(x_t, t_tensor, context)  # Use reverse diffusion
    
    return x_t

if __name__ == "__main__":
    # Set up model parameters
    input_dim = 4
    num_timesteps = 1000
    num_samples = 10000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load your trained diffusion model
    diffusion_model = DiffusionModel(input_dim=input_dim, num_timesteps=num_timesteps, device=device).to(device)
    checkpoint = torch.load('models_DM/epoch-300.pt', map_location=device)
    diffusion_model.load_state_dict(checkpoint['model_DM'])

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
                sim = np.load(f)[:, :4]
            else:
                sim = np.concatenate((sim, np.load(f)[:, :4]))

        # Generate samples using reverse diffusion
        print(f"Generating samples for {e} eV (index {i})")
        fixed_value_5th_dim = torch.tensor([[float(e)]], device=device)
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
        plt.savefig(f"/web/aratey/public_html/delight/nf/models_DM/gen_{i}_DM.png", bbox_inches='tight', dpi=300)

