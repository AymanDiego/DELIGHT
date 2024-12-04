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
from model import AttentionDiffusionModel, linear_noise_schedule, diffusion_loss, sample_step

# Load widths.csv
widths_data = pd.read_csv("widths.csv")
print("Columns in widths_data:", widths_data.columns)

# ArgumentParser to handle command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train Diffusion Model")
    
    # Add argument for GPU selection
    parser.add_argument('--device', type=str, default='cuda:1', help='Specify the device to run the training on (e.g., cuda:0, cuda:1, cpu)')
    
    # Add argument for specifying the directory to save model files
    parser.add_argument('--model_dir', type=str, default='models_df', help='Directory to save the model checkpoints')

    parser.add_argument('--loss_dir', type=str, default='', help='Specify an extra directory to append for saving files (e.g., "run_01")')

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
def train_diffusion_model(df_model, data_train, context_train, data_val, context_val, widths_data, num_epochs=1000, noise_schedule=None, batch_size=512, learning_rate=1e-3, timesteps=300):
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
            loss = diffusion_loss(df_model, data_train, context_train, noise_schedule, timesteps, widths_data)
            total_loss_train += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()
        
        # Validate
        df_model.eval()
        for batch in tqdm.tqdm(dataloader_val, desc=f"Validation epoch {epoch}"):
            with torch.no_grad():
                batch_data, batch_context = batch
                loss_val = diffusion_loss(df_model, data_val, context_val, noise_schedule, timesteps, widths_data)
                total_loss_val += loss_val.item()
        
        # Print loss every epoch
        print(f"Epoch {epoch}, Train Loss: {total_loss_train / len(dataloader_train.dataset)}, Val Loss: {total_loss_val / len(dataloader_val.dataset)}")
        all_losses_train.append(total_loss_train / len(dataloader_train.dataset))
        all_losses_val.append(total_loss_val / len(dataloader_val.dataset))

        # Save models
        state_dicts = {'model':df_model.state_dict(),'opt':optimizer.state_dict(),'lr':scheduler.state_dict()}
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir) 
        torch.save(state_dicts, f'{args.model_dir}/epoch-{epoch}.pt')
    
    # Save loss data to a CSV file
    df = pd.DataFrame({"loss_train": all_losses_train, "loss_val": all_losses_val})
    df.to_csv(f"{save_dir}/loss.csv", index=False)

    fig,ax = plt.subplots()
    plt.yscale('log')
    plt.plot([i for i in range(len(all_losses_train))],all_losses_train,label='train')
    plt.plot([i for i in range(len(all_losses_val))],all_losses_val,label='val')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{save_dir}/loss.png", bbox_inches='tight', dpi=300)
    plt.savefig(f"{save_dir}/loss.pdf", bbox_inches='tight')
    
def validate():
    flow_model.eval()
    for batch in tqdm.tqdm(dataloader_val, desc=f"Validation epoch {epoch}"):
        with torch.no_grad():
            batch_data, batch_context = batch
            loss = -flow_model(batch_data, batch_context).mean()


def concat_files(filelist,cutoff_min,cutoff_max):
    all_data = None
    for i,f in tqdm.tqdm(enumerate(filelist), total=len(filelist), desc="Loading data into array"):
        # Load file and retrieve all four channels
        data = np.load(f)[:, :4]
        
        # Calculate energy as the sum of all channels
        energy = np.sum(data, axis=1).reshape(-1, 1)

        if energy[0] < cutoff_min:
            continue

        if energy[0] > cutoff_max:
            continue

        # Filter out entries below the cutoff energy
        valid_entries = energy >= 0
        data = data[valid_entries.ravel()]
        energy = energy[valid_entries.ravel()]

        # Concatenate data if not empty
        if all_data is None:
            all_data = np.concatenate((data/energy, energy/1000000), axis=1)
        else:
            all_data = np.concatenate((all_data, np.concatenate((data/energy, energy/1000000), axis=1)), axis=0)

    idx = [i for i in range(len(all_data))]
    print("Number of events:", len(idx))
    random.shuffle(idx)
    #all_data = all_data[idx][:50000]
    all_data = all_data[idx]

    return all_data

# Example usage
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

    # Loading data
    cutoff_min_e = 0. # eV. Ingnore interactions below that.
    cutoff_max_e = 1000000. # eV. Ingnore interactions higher than that.
    logger.info(f'Load data for evens with energy larger than {cutoff_min_e} and smaller than {cutoff_max_e} eV.')
    files_train = glob.glob("/ceph/bmaier/delight/ml/nf/data/train/*npy")
    files_val = glob.glob("/ceph/bmaier/delight/ml/nf/data/val/*npy")
    random.seed(123)
    random.shuffle(files_train)
    data_train = concat_files(files_train,cutoff_min_e,cutoff_max_e)
    data_val = concat_files(files_val,cutoff_min_e,cutoff_max_e)
    
    # Separate the data into the first 4 dimensions (input) and the 5th dimension (context)
    data_train_4d = data_train[:, :4]
    context_train_5d = data_train[:, 4:5]
    data_val_4d = data_val[:, :4]
    context_val_5d = data_val[:, 4:5]
    
    # Initialize the conditional flow model (input dimension 4, context dimension 1, hidden dimension 64, 5 layers)
    data_dim = 4
    condition_dim = 1
    timesteps = 25
    batch_size = 1024
    learning_rate = 1e-2
    num_epochs = 150
    noise_schedule = linear_noise_schedule(timesteps).to(args.device)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Training on {device}')
    
    # Create and move the model to the appropriate device
    df_model = AttentionDiffusionModel(data_dim=data_dim, condition_dim=condition_dim, timesteps=timesteps, device=args.device).to(device)

    # Save hyperparameters
    save_diffusion_hyperparameters_to_csv(data_dim=data_dim, condition_dim=condition_dim, timesteps=timesteps, learning_rate=learning_rate, num_epochs=num_epochs, save_dir=save_dir)


    # Train the model
    train_diffusion_model(df_model, data_train_4d, context_train_5d, data_val_4d, context_val_5d, num_epochs=num_epochs, noise_schedule=noise_schedule, batch_size=batch_size, learning_rate=learning_rate, timesteps=timesteps, widths_data=widths_data)
    logger.info(f'Done training.')
    
