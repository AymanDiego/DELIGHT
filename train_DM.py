import os
import glob
import tqdm
import logging
import random
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import mplhep as hep
from model_DM import DiffusionModel, cosine_noise_schedule, linear_noise_schedule, constant_noise_schedule 

hep.style.use(hep.style.ATLAS)

# ArgumentParser to handle command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train Diffusion Model")
    
    # Device argument
    parser.add_argument('--device', type=str, default='cuda:0', help='Specify the device to run the training on (e.g., cuda:0, cuda:1, cpu)')
    
    # Model and save directories
    parser.add_argument('--model_dir', type=str, default='models_DM/', help='Directory to save the model checkpoints')
    parser.add_argument('--loss_dir', type=str, default='', help='Specify an extra directory to append for saving files (e.g., "run_01")')
    
    # File handling arguments
    parser.add_argument('--file_pattern', type=str, default='all', help='Specify file pattern: "all" for all files or a specific pattern like NR_final_200_*.npy')
    
    # Diffusion model parameters
    parser.add_argument('--num_timesteps', type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument('--noise_magnitude', type=float, default=0.0, help="Magnitude of noise to apply to energy values")
    parser.add_argument('--noise_scale', type=float, default=0.001, help="Scaling factor for the cosine noise schedule")
    parser.add_argument('--energy_threshold', type=float, default=0.0, help="Threshold below which noise is added")
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')

    # Data cutoff
    parser.add_argument('--cutoff_e', type=float, default=0.0, help='Cutoff energy value. Ignore events below this energy in eV.')
   
    # Add argument for noise schedule
    parser.add_argument('--noise_schedule', type=str, choices=['cosine', 'linear', 'constant', 'none'], default='none',
                        help="Type of noise schedule to use: 'cosine', 'linear', 'constant', or 'none'")

    return parser.parse_args()

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

def apply_noise(data, energy_column, noise_magnitude, energy_threshold):
    mask = data[:, energy_column] < energy_threshold
    noise = noise_magnitude * np.random.randn(mask.sum(), data.shape[1])
    data[mask, :] += noise
    return data

# Function to save hyperparameters to CSV
def save_hyperparameters_to_csv(input_dim, num_timesteps, learning_rate, num_epochs, weight_decay, noise_magnitude, noise_scale, energy_threshold, cutoff_e, noise_schedule_type, save_dir):
    """
    Saves the hyperparameters used for training the diffusion model to a CSV file in save_dir.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    hyperparams = {
        "input_dim": input_dim,
        "num_timesteps": num_timesteps,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "weight_decay": weight_decay,
        "noise_magnitude": noise_magnitude,
        "noise_scale": noise_scale,
        "energy_threshold": energy_threshold,
        "cutoff_energy": cutoff_e,
        "noise_schedule_type": noise_schedule_type,
    }

    df = pd.DataFrame([hyperparams])
    df.to_csv(f"{save_dir}/hyperparameters.csv", index=False)

def concat_files(filelist, cutoff):
    """
    Concatenates files and calculates energy by summing across all channels.
    Ignores interactions with total energy below the specified cutoff.
    """
    all_data = None
    for f in tqdm.tqdm(filelist, desc="Loading and processing data"):
        # Load file and retrieve all four channels
        data = np.load(f)[:, :4]

        # Calculate energy as the sum of all channels
        energy = np.sum(data, axis=1).reshape(-1, 1)

        # Filter out entries below the cutoff energy
        valid_entries = energy >= cutoff
        data = data[valid_entries.ravel()]
        energy = energy[valid_entries.ravel()]

        # Concatenate data if not empty
        if all_data is None:
            all_data = np.concatenate((data, energy), axis=1)
        else:
            all_data = np.concatenate((all_data, np.concatenate((data, energy), axis=1)), axis=0)

    return all_data

# Train diffusion model
def train_diffusion_model(diffusion_model, data_train, data_val, args, save_dir, model_dir, noise_schedule):
    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    data_train = apply_noise(data_train, energy_column=4, noise_magnitude=args.noise_magnitude, energy_threshold=args.energy_threshold)
    data_val = apply_noise(data_val, energy_column=4, noise_magnitude=args.noise_magnitude, energy_threshold=args.energy_threshold)
    
    dataset_train = torch.utils.data.TensorDataset(torch.tensor(data_train, dtype=torch.float32).to(diffusion_model.device),
                                                   torch.tensor(data_train[:, 4:5], dtype=torch.float32).to(diffusion_model.device))
    dataset_val = torch.utils.data.TensorDataset(torch.tensor(data_val, dtype=torch.float32).to(diffusion_model.device),
                                                 torch.tensor(data_val[:, 4:5], dtype=torch.float32).to(diffusion_model.device))
    
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    all_losses_train = []
    all_losses_val = []

    for epoch in range(args.num_epochs):
        diffusion_model.train()
        total_loss_train = 0
        for batch_data, batch_context in tqdm.tqdm(dataloader_train, desc=f"Training epoch {epoch}"):
            batch_data = batch_data.to(diffusion_model.device)  # Ensure batch data is on the same device
            batch_context = batch_context.to(diffusion_model.device)  # Ensure batch context is on the same device
            optimizer.zero_grad()
            loss = diffusion_model.compute_loss(batch_data, batch_context, noise_schedule)
            loss.backward()
            optimizer.step()
            total_loss_train += loss.item()

        diffusion_model.eval()
        total_loss_val = 0
        with torch.no_grad():
            for batch_data, batch_context in tqdm.tqdm(dataloader_val, desc=f"Validation epoch {epoch}"):
                batch_data = batch_data.to(diffusion_model.device)  # Ensure validation data is on the same device
                batch_context = batch_context.to(diffusion_model.device)  # Ensure validation context is on the same device
                loss_val = diffusion_model.compute_loss(batch_data, batch_context, noise_schedule)
                total_loss_val += loss_val.item()

        scheduler.step(total_loss_val / len(dataloader_val))
        print(f"Epoch {epoch}: Train Loss = {total_loss_train / len(dataloader_train)}, Val Loss = {total_loss_val / len(dataloader_val)}")

        all_losses_train.append(total_loss_train / len(dataloader_train))
        all_losses_val.append(total_loss_val / len(dataloader_val))

        # Save the model checkpoint for each epoch
        torch.save(diffusion_model.state_dict(), f"{model_dir}/dm_epoch_{epoch}.pt")

    # Save loss data to a CSV file
    df = pd.DataFrame({"loss_train": all_losses_train, "loss_val": all_losses_val})
    df.to_csv(f"{save_dir}/loss.csv", index=False)

    # Plot and save losses
    fig, ax = plt.subplots()
    plt.yscale('log')
    plt.plot([i for i in range(len(all_losses_train))], all_losses_train, label='train')
    plt.plot([i for i in range(len(all_losses_val))], all_losses_val, label='val')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{save_dir}/loss.png", bbox_inches='tight', dpi=300)
    plt.savefig(f"{save_dir}/loss.pdf", bbox_inches='tight')

if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()

    # Validate noise_scale
    if args.noise_scale <= 0:
        raise ValueError("noise_scale must be positive.")

    # Define the save directory
    base_dir = "/web/aratey/public_html/delight/nf/models_DM/DM_old/"
    # Append loss_dir if provided
    if args.loss_dir:
        save_dir = os.path.join(base_dir, args.loss_dir)
    else:
        save_dir = base_dir

    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cutoff_e = args.cutoff_e  # eV. Ignore interactions below that.
    logger.info(f'Load data for events with energy larger than {cutoff_e} eV.')

    if args.file_pattern == 'all':
        files_train = glob.glob("/ceph/bmaier/delight/ml/nf/data/train/*.npy")
        files_val = glob.glob("/ceph/bmaier/delight/ml/nf/data/val/*.npy")
    else:
        files_train = glob.glob(f"/ceph/bmaier/delight/ml/nf/data/train/{args.file_pattern}")
        files_val = glob.glob(f"/ceph/bmaier/delight/ml/nf/data/val/{args.file_pattern}")

    random.shuffle(files_train)
    data_train = concat_files(files_train, args.cutoff_e)
    data_val = concat_files(files_val, args.cutoff_e)

    # Save hyperparameters to CSV
    save_hyperparameters_to_csv(
        input_dim=4,
        num_timesteps=args.num_timesteps,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        noise_magnitude=args.noise_magnitude,
        noise_scale=args.noise_scale,
        energy_threshold=args.energy_threshold,
        cutoff_e=args.cutoff_e,
        save_dir=save_dir,
        noise_schedule_type=args.noise_schedule
    )

    # Precompute noise schedule based on user input
    if args.noise_schedule == 'cosine':
        noise_schedule = cosine_noise_schedule(args.num_timesteps) * args.noise_scale
    elif args.noise_schedule == 'linear':
        noise_schedule = linear_noise_schedule(args.num_timesteps) * args.noise_scale
    elif args.noise_schedule == 'constant':
        noise_schedule = constant_noise_schedule(args.num_timesteps, scale=args.noise_scale)
    elif args.noise_schedule == 'none':
        noise_schedule = None  # No noise schedule, skip adding noise
    else:
        raise ValueError(f"Unknown noise schedule: {args.noise_schedule}")

    # Move the noise schedule tensor to the correct device if not None
    if noise_schedule is not None:
        noise_schedule = noise_schedule.to(args.device)

    diffusion_model = DiffusionModel(input_dim=4, num_timesteps=args.num_timesteps, device=args.device, noise_schedule=noise_schedule).to(args.device)
    train_diffusion_model(diffusion_model, data_train, data_val, args, save_dir, args.model_dir, noise_schedule)

