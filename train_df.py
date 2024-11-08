import os
import glob
import tqdm
import logging
import random
import argparse
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.ATLAS)
import pandas as pd
import numpy as np
import torch
from model import AttentionDiffusionModel, cosine_noise_schedule, diffusion_loss, sample_step

# ArgumentParser to handle command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train Diffusion Model")
    
    # Add argument for GPU selection
    parser.add_argument('--device', type=str, default='cuda:1', help='Specify the device to run the training on (e.g., cuda:0, cuda:1, cpu)')
    
    # Add argument for specifying the directory to save model files
    parser.add_argument('--model_dir', type=str, default='models_diffusion_timesteps', help='Directory to save the model checkpoints')

    parser.add_argument('--loss_dir', type=str, default='', help='Specify an extra directory to append for saving files (e.g., "run_01")')

    # Add argument to specify the file pattern (all files or specific ones)
    parser.add_argument('--file_pattern', type=str, default='all', help='Specify file pattern: "all" for all files or a specific pattern like NR_final_200_*.npy')
    
    # Add argument for number of epochs and learning rate
    parser.add_argument('--num_epochs', type=int, default=19, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer')

    # Add argument for the cutoff energy
    parser.add_argument('--cutoff_e', type=float, default=0.0, help='Cutoff energy value. Ignore events below this energy in eV.')

    return parser.parse_args()

def setup_logger():
    # Set up logging to console
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # Set logging level

    # Create console handler for stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and set it for the console handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger

# Function to save hyperparameters to CSV for diffusion model
def save_diffusion_hyperparameters_to_csv(data_dim, condition_dim, timesteps, learning_rate, num_epochs, save_dir):
    hyperparams = {
        "data_dim": data_dim,
        "condition_dim": condition_dim,
        "timesteps": timesteps,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs
    }
    df = pd.DataFrame([hyperparams])
    df.to_csv(f"{save_dir}/diffusion_hyperparameters.csv", index=False)

# Training the diffusion model
def train_diffusion_model(df_model, data_train, context_train, data_val, context_val, num_epochs, noise_schedule, batch_size=512, learning_rate=1e-3, timesteps=300):
    optimizer = torch.optim.Adam(df_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Convert data and context to tensors and move them to the same device as the model
    data_train = torch.tensor(data_train, dtype=torch.float32, device=df_model.device)
    context_train = torch.tensor(context_train, dtype=torch.float32, device=df_model.device)
    data_val = torch.tensor(data_val, dtype=torch.float32, device=df_model.device)
    context_val = torch.tensor(context_val, dtype=torch.float32, device=df_model.device)
    
    dataset_train = torch.utils.data.TensorDataset(data_train, context_train)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataset_val = torch.utils.data.TensorDataset(data_val, context_val)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    # Train
    all_losses_train = []
    all_losses_val = []
    for epoch in range(num_epochs):        
        total_loss_train = 0
        total_loss_val = 0
        df_model.train()
        for batch in tqdm.tqdm(dataloader_train, desc=f"Training epoch {epoch}"):
            batch_data, batch_context = batch
            optimizer.zero_grad()
            loss = diffusion_loss(df_model, batch_data, batch_context, noise_schedule, timesteps)
            total_loss_train += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()
        
        # Validate
        df_model.eval()
        for batch in tqdm.tqdm(dataloader_val, desc=f"Validation epoch {epoch}"):
            with torch.no_grad():
                batch_data, batch_context = batch
                loss_val = diffusion_loss(df_model, batch_data, batch_context, noise_schedule, timesteps)
                total_loss_val += loss_val.item()
        
        # Log and save losses
        print(f"Epoch {epoch}, Train Loss: {total_loss_train / len(dataloader_train.dataset)}, Val Loss: {total_loss_val / len(dataloader_val.dataset)}")
        all_losses_train.append(total_loss_train / len(dataloader_train.dataset))
        all_losses_val.append(total_loss_val / len(dataloader_val.dataset))

        # Save models in the specified directory
        state_dicts = {'model': df_model.state_dict(), 'opt': optimizer.state_dict(), 'lr': scheduler.state_dict()}
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        torch.save(state_dicts, f'{args.model_dir}/epoch-{epoch}.pt')
    
    # Save loss data to a CSV file
    df = pd.DataFrame({"loss_train": all_losses_train, "loss_val": all_losses_val})
    df.to_csv(f"{save_dir}/loss.csv", index=False)

    # Plot losses
    fig, ax = plt.subplots()
    plt.yscale('log')
    plt.plot([i for i in range(len(all_losses_train))], all_losses_train, label='train')
    plt.plot([i for i in range(len(all_losses_val))], all_losses_val, label='val')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{save_dir}/loss.png", bbox_inches='tight', dpi=300)
    plt.savefig(f"{save_dir}/loss.pdf", bbox_inches='tight')

# Concatenates files and applies cutoff for loading data
def concat_files(filelist, cutoff):
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

if __name__ == "__main__":
    args = parse_args()  # Parse command-line arguments
    logger = setup_logger()

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

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Training on {device}')

    # Loading data
    cutoff_e = args.cutoff_e
    logger.info(f'Load data for events with energy larger than {cutoff_e} eV.')

    if args.file_pattern == 'all':
        files_train = glob.glob("/ceph/bmaier/delight/ml/nf/data/train/*npy")
        files_val = glob.glob("/ceph/bmaier/delight/ml/nf/data/val/*npy")
    else:
        files_train = glob.glob(f"/ceph/bmaier/delight/ml/nf/data/train/{args.file_pattern}")
        files_val = glob.glob(f"/ceph/bmaier/delight/ml/nf/data/val/{args.file_pattern}")

    # Shuffle and load files
    random.seed(123)
    random.shuffle(files_train)
    data_train = concat_files(files_train, cutoff_e)
    data_val = concat_files(files_val, cutoff_e)
    
    # Separate data and context
    data_train_4d = data_train[:, :4]
    context_train_5d = data_train[:, 4:5]
    data_val_4d = data_val[:, :4]
    context_val_5d = data_val[:, 4:5]

    # Model parameters
    data_dim = 4
    condition_dim = 1
    timesteps = 100
    batch_size = 128

    # Create noise schedule and model
    noise_schedule = cosine_noise_schedule(timesteps).to(device)
    df_model = AttentionDiffusionModel(data_dim=data_dim, condition_dim=condition_dim, timesteps=timesteps, device=device).to(device)
   
    # Save hyperparameters
    save_diffusion_hyperparameters_to_csv(data_dim=data_dim, condition_dim=condition_dim, timesteps=timesteps, learning_rate=args.learning_rate, num_epochs=args.num_epochs, save_dir=save_dir)

    # Train the model
    train_diffusion_model(df_model, data_train_4d, context_train_5d, data_val_4d, context_val_5d, num_epochs=args.num_epochs, noise_schedule=noise_schedule, batch_size=batch_size, learning_rate=args.learning_rate, timesteps=timesteps)
    
    logger.info('Done training.')

