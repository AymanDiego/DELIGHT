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
from model import ConditionalNormalizingFlowModel

# ArgumentParser to handle command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train Conditional Normalizing Flow Model")
    
    # Add argument for GPU selection
    parser.add_argument('--device', type=str, default='cuda:0', help='Specify the device to run the training on (e.g., cuda:0, cuda:1, cpu)')
    
    # Add argument for specifying the directory to save model files
    parser.add_argument('--model_dir', type=str, default='models/', help='Directory to save the model checkpoints')

    # Add argument to specify the file pattern (all files or specific ones)
    parser.add_argument('--file_pattern', type=str, default='all', help='Specify file pattern: "all" for all files or a specific pattern like NR_final_200_*.npy')
    
    # Add argument for the noise magnitude and energy threshold for noise application
    parser.add_argument('--noise_magnitude', type=float, default=0.1, help='Magnitude of the noise to apply to small energies')
    parser.add_argument('--energy_threshold', type=float, default=5000, help='Energy threshold below which noise is applied')
    
    # Learning rate and other training parameters
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 regularization)')
    parser.add_argument('--num_epochs', type=int, default=301, help='Number of training epochs')

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
    # Create a boolean mask for energies below the threshold (use context as energy)
    mask = context[:, 0] < energy_threshold

    # Add noise to the masked entries in the data
    noise = noise_magnitude * torch.randn_like(data[mask]) + torch.tensor(1)
    data[mask] *= noise

    return data

# Training the flow model
def train_conditional_flow_model(flow_model, data_train, context_train, data_val, context_val, num_epochs=1000, batch_size=512, learning_rate=1e-3, model_dir='models/', noise_magnitude=0.1, energy_threshold=5000, weight_decay=1e-5):
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    
    # Convert data and context to tensors and move them to the same device as the model
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

    # Create model directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Train
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
    df.to_csv(f"/web/aratey/public_html/delight/nf/loss.csv")

    fig, ax = plt.subplots()
    plt.yscale('log')
    plt.plot([i for i in range(len(all_losses_train))], all_losses_train, label='train')
    plt.plot([i for i in range(len(all_losses_val))], all_losses_val, label='val')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"/web/aratey/public_html/delight/nf/loss.png", bbox_inches='tight', dpi=300)
    plt.savefig(f"/web/aratey/public_html/delight/nf/loss.pdf", bbox_inches='tight')

# Function to concatenate files
def concat_files(filelist, cutoff):
    e_array = np.geomspace(10, 1e6, 500)
    all_data = None
    for f in tqdm.tqdm(filelist, desc="Loading data into array"):
        idx = int(f.split("NR_final_")[-1].split("_")[0])
        if e_array[idx] < cutoff:
            continue
        if all_data is None:
            all_data = np.load(f)[:, :4]
        else:
            all_data = np.concatenate((all_data, np.load(f)[:, :4]))
    energy = np.sum(all_data, axis=1).reshape(-1, 1)
    
    return np.concatenate((all_data, energy), axis=1)

if __name__ == "__main__":
    args = parse_args()  # Parse command-line arguments
    logger = setup_logger()

    # Loading data
    cutoff_e = 0.  # eV. Ignore interactions below that.
    logger.info(f'Load data for events with energy larger than {cutoff_e} eV.')
    
    # Determine file pattern based on command-line argument
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
    data_train = concat_files(files_train, cutoff_e)  # Use cutoff_e as defined above
    data_val = concat_files(files_val, cutoff_e)  # Use cutoff_e as defined above
    
    # Separate the data into the first 4 dimensions (input) and the 5th dimension (context)
    data_train_4d = data_train[:, :4]
    context_train_5d = data_train[:, 4:5]
    data_val_4d = data_val[:, :4]
    context_val_5d = data_val[:, 4:5]

    # Initialize the conditional flow model (input dimension 4, context dimension 1, hidden dimension 64, 5 layers)
    input_dim = 4
    context_dim = 1
    hidden_dim = 128
    num_layers = 8
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')  # Use the device from arguments
    logger.info(f'Training on {device}')
    
    # Create and move the model to the appropriate device
    flow_model = ConditionalNormalizingFlowModel(input_dim, context_dim, hidden_dim, num_layers, device).to(device)
    
    # Train the model, passing the model directory from arguments
    train_conditional_flow_model(flow_model, data_train_4d, context_train_5d, data_val_4d, context_val_5d, num_epochs=args.num_epochs, model_dir=args.model_dir, noise_magnitude=args.noise_magnitude, energy_threshold=args.energy_threshold, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    logger.info(f'Done training.')

