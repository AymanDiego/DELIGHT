import os
import glob
import tqdm
import logging
import random
import argparse
import matplotlib.pyplot as plt
import mplhep as hep
import pandas as pd
import numpy as np
import torch
from model import ConditionalNormalizingFlowModel

hep.style.use(hep.style.ATLAS)

# ArgumentParser to handle command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train Conditional Normalizing Flow Model")
    
    # Add argument for GPU selection
    parser.add_argument('--device', type=str, default='cuda:1', help='Specify the device to run the training on (e.g., cuda:0, cuda:1, cpu)')
    
    # Add argument for specifying the directory to save model files
    parser.add_argument('--model_dir', type=str, default='models/', help='Directory to save the model checkpoints')

    # Add argument to specify the file pattern (all files or specific ones)
    parser.add_argument('--file_pattern', type=str, default='all', help='Specify file pattern: "all" for all files or a specific pattern like NR_final_200_*.npy')
    
    # Add argument for the noise magnitude and energy threshold for noise application
    parser.add_argument('--noise_magnitude', type=float, default=0.0, help='Magnitude of the noise to apply to small energies')
    parser.add_argument('--energy_threshold', type=float, default=5000, help='Energy threshold below which noise is applied')

    # Add argument for number of epochs, learning rate, weight decay
    parser.add_argument('--num_epochs', type=int, default=301, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')

    parser.add_argument('--loss_dir', type=str, default='', help='Specify an extra directory to append for saving files (e.g., "run_01")')

    # Add argument for optional energy normalization
    parser.add_argument('--normalize_energies', action='store_true', help='Flag to apply energy normalization (mean 0, stddev 1)')

    # Add argument for the cutoff energy
    parser.add_argument('--cutoff_e', type=float, default=0.0, help='Cutoff energy value. Ignore events below this energy in eV.')

    return parser.parse_args()

# Logger setup
def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

# Function to add noise to small energy events
def apply_noise(data, context, noise_magnitude, energy_threshold):
    """
    Adds Gaussian noise to the data for entries where the energy (in context) is below the given threshold.
    """
    mask = context[:, 0] < energy_threshold
    noise = noise_magnitude * torch.randn_like(data[mask])
    data[mask] += noise

    # Plot the noisy input data for each epoch
    if noise_magnitude > 0:
        fig, ax = plt.subplots(figsize=(7, 6))
        plt.hist(data_train[:, 0].cpu().numpy(), histtype='step', bins=15, label='phonon channel', color='indianred')
        plt.hist(data_train[:, 1].cpu().numpy(), histtype='step', bins=15, label='triplet channel', color='grey')
        plt.hist(data_train[:, 2].cpu().numpy(), histtype='step', bins=15, label='UV channel', color='gold')
        plt.hist(data_train[:, 3].cpu().numpy(), histtype='step', bins=15, label='IR channel', color='cornflowerblue')
        ax.set_xlabel("$E$ (eV)", labelpad=20)
        ax.set_ylabel("Arbitrary units")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/noisy_input.png", bbox_inches='tight', dpi=300)

    return data

# Function to save hyperparameters to CSV
def save_hyperparameters_to_csv(input_dim, context_dim, hidden_dim, num_layers, learning_rate, num_epochs, weight_decay, noise_magnitude, energy_threshold, model_dir):
    hyperparams = {
        "input_dim": input_dim,
        "context_dim": context_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "weight_decay": weight_decay,
        "noise_magnitude": noise_magnitude,
        "energy_threshold": energy_threshold
    }
    df = pd.DataFrame([hyperparams])
    df.to_csv(f"{save_dir}/hyperparameters.csv", index=False)

# Function to normalize the energy channels
def normalize_energies(data):
    means = np.mean(data, axis=0)  # Mean for each channel
    stds = np.std(data, axis=0)    # Standard deviation for each channel
    normalized_data = (data - means) / stds  # Normalize each channel
    return normalized_data, means, stds

# Function to check means and stds after normalization
def check_normalized_stats(normalized_data):
    post_means = np.mean(normalized_data, axis=0)
    post_stds = np.std(normalized_data, axis=0)
    return post_means, post_stds

# Save normalization parameters (means and stds) for later inference use
def save_normalization_params(means, stds, model_dir):
    # Ensure the directory exists before saving
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    df = pd.DataFrame({"channel": ["phonon", "triplet", "UV", "IR"], "means": means, "stds": stds})
    df.to_csv(f"{save_dir}/normalization_params.csv", index=False)

# Function to save post-normalization statistics to CSV
def save_post_normalization_params(post_means, post_stds, loss_dir):
    # Ensure the directory exists before saving
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    df = pd.DataFrame({
        "channel": ["phonon", "triplet", "UV", "IR"],
        "post_means": post_means,
        "post_stds": post_stds
    })
    df.to_csv(f"{save_dir}/post_normalization_params.csv", index=False)

# Training the flow model
def train_conditional_flow_model(flow_model, data_train, context_train, data_val, context_val, num_epochs, batch_size=512, learning_rate=1e-4, weight_decay=1e-5, model_dir='models/', noise_magnitude=0.1, energy_threshold=5000):
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    
    data_train = torch.tensor(data_train, dtype=torch.float32, device=flow_model.device)
    context_train = torch.tensor(context_train, dtype=torch.float32, device=flow_model.device)
    data_val = torch.tensor(data_val, dtype=torch.float32, device=flow_model.device)
    context_val = torch.tensor(context_val, dtype=torch.float32, device=flow_model.device)
    
    # Apply noise to the training data using context (energy) as the criterion
    data_train = apply_noise(data_train, context_train, noise_magnitude=noise_magnitude, energy_threshold=energy_threshold)
    
    dataset_train = torch.utils.data.TensorDataset(data_train, context_train)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataset_val = torch.utils.data.TensorDataset(data_val, context_val)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    all_losses_train = []
    all_losses_val = []

    for epoch in range(num_epochs):
        total_loss_train = 0
        total_loss_val = 0
        flow_model.train()

        for batch in tqdm.tqdm(dataloader_train, desc=f"Training epoch {epoch}"):
            batch_data, batch_context = batch
            optimizer.zero_grad()
            loss = -flow_model(batch_data, batch_context).mean()  # Maximize the log probability
            total_loss_train += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=1.0)  # Clip gradients to avoid large updates
            optimizer.step()

        # Validation
        flow_model.eval()
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader_val, desc=f"Validation epoch {epoch}"):
                batch_data, batch_context = batch
                loss_val = -flow_model(batch_data, batch_context).mean()
                total_loss_val += loss_val.item()

        # Adjust learning rate based on validation loss
        scheduler.step(total_loss_val / len(dataloader_val))

        # Print loss every epoch
        print(f"Epoch {epoch}, Train Loss: {total_loss_train / len(dataloader_train.dataset)}, Val Loss: {total_loss_val / len(dataloader_val.dataset)}")
        all_losses_train.append(total_loss_train / len(dataloader_train.dataset))
        all_losses_val.append(total_loss_val / len(dataloader_val.dataset))

        # Save models in the specified directory
        state_dicts = {'model': flow_model.state_dict(), 'opt': optimizer.state_dict(), 'lr': scheduler.state_dict()}
        torch.save(state_dicts, f'{model_dir}/epoch-{epoch}.pt')
    
    # Save loss data to a CSV file
    df = pd.DataFrame({"loss_train": all_losses_train, "loss_val": all_losses_val})
    df.to_csv(f"{save_dir}/loss.csv")

    fig, ax = plt.subplots()
    plt.yscale('log')
    plt.plot([i for i in range(len(all_losses_train))], all_losses_train, label='train')
    plt.plot([i for i in range(len(all_losses_val))], all_losses_val, label='val')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{save_dir}/loss.png", bbox_inches='tight', dpi=300)
    plt.savefig(f"{save_dir}/loss.pdf", bbox_inches='tight')

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

if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()

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

    cutoff_e = args.cutoff_e  # eV. Ignore interactions below that.
    logger.info(f'Load data for events with energy larger than {cutoff_e} eV.')

    if args.file_pattern == 'all':
        files_train = glob.glob("/ceph/aratey/delight/ml/nf/data/train/*.npy")
        files_val = glob.glob("/ceph/aratey/delight/ml/nf/data/val/*.npy")
    else:
        files_train = glob.glob(f"/ceph/aratey/delight/ml/nf/data/train/{args.file_pattern}")
        files_val = glob.glob(f"/ceph/aratey/delight/ml/nf/data/val/{args.file_pattern}")

    # Log the file pattern being used
    logger.info(f'Using file pattern: {args.file_pattern}')

    random.seed(123)
    random.shuffle(files_train)
    data_train = concat_files(files_train, cutoff_e)
    data_val = concat_files(files_val, cutoff_e)

    # Normalize the training and validation data
    if args.normalize_energies:
        logger.info("Applying energy normalization...")
        data_train_4d, means_train, stds_train = normalize_energies(data_train[:, :4])
        context_train_5d = data_train[:, 4:5]  # Context is not normalized
        data_val_4d, means_val, stds_val = normalize_energies(data_val[:, :4])
        context_val_5d = data_val[:, 4:5]  # Context is not normalized

        # Save normalization parameters for future use
        save_normalization_params(means_train, stds_train, args.model_dir)
    
        # Check post-normalization stats
        post_means_train, post_stds_train = check_normalized_stats(data_train_4d)
        post_means_val, post_stds_val = check_normalized_stats(data_val_4d)

        # Save post-normalization parameters to loss_dir
        save_post_normalization_params(post_means_train, post_stds_train, args.loss_dir)

        # Plot the normalized data
        fig, ax = plt.subplots(figsize=(7, 6))
        plt.hist(data_train_4d[:, 0], histtype='step', bins=15, label='phonon channel (normalized)', color='indianred')
        plt.hist(data_train_4d[:, 1], histtype='step', bins=15, label='triplet channel (normalized)', color='grey')
        plt.hist(data_train_4d[:, 2], histtype='step', bins=15, label='UV channel (normalized)', color='gold')
        plt.hist(data_train_4d[:, 3], histtype='step', bins=15, label='IR channel (normalized)', color='cornflowerblue')
        ax.set_xlabel("Normalized Energy", labelpad=20)
        ax.set_ylabel("Arbitrary units")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/normalized_data.png", bbox_inches='tight', dpi=300)

    else:
        logger.info("Skipping energy normalization...")
        data_train_4d = data_train[:, :4]
        context_train_5d = data_train[:, 4:5]  # Context is not normalized
        data_val_4d = data_val[:, :4]
        context_val_5d = data_val[:, 4:5]  # Context is not normalized

    # Initialize the conditional flow model (input dimension 4, context dimension 1, hidden dimension 128, 8 layers)
    input_dim = 4
    context_dim = 1
    hidden_dim = 128
    num_layers = 8
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Training on {device}')

    # Create and move the model to the appropriate device
    flow_model = ConditionalNormalizingFlowModel(input_dim, context_dim, hidden_dim, num_layers, device).to(device)

    # Calculate and print the number of trainable parameters
    num_params = sum(p.numel() for p in flow_model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {num_params}")

    # Save hyperparameters
    save_hyperparameters_to_csv(input_dim, context_dim, hidden_dim, num_layers, args.learning_rate, args.num_epochs, args.weight_decay, args.noise_magnitude, args.energy_threshold, args.model_dir)

    # Train the model
    train_conditional_flow_model(flow_model, data_train_4d, context_train_5d, data_val_4d, context_val_5d, num_epochs=args.num_epochs, model_dir=args.model_dir, noise_magnitude=args.noise_magnitude, energy_threshold=args.energy_threshold, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

    logger.info(f'Done training.')

