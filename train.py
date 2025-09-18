# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
from tqdm import tqdm

# Import our custom modules
from src.model import SpiralMLP
from src.utils import load_config, set_seed, count_parameters, save_checkpoint
# <-- REVERTED import path for broader compatibility
from torch.cuda.amp import GradScaler, autocast 

def train(config):
    """
    Main training loop for the SpiralMLP model.
    """
    # --- 1. Setup and Initialization ---
    set_seed(config['train_params']['seed'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize GradScaler for mixed precision training
    # <-- REVERTED syntax for broader compatibility
    scaler = GradScaler(enabled=(device == "cuda")) 

    # --- 2. Data Loading ---
    print("Loading data...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(
        root=config['data_params']['data_path'], train=True, download=True, transform=transform_train
    )
    val_dataset = datasets.CIFAR10(
        root=config['data_params']['data_path'], train=False, download=True, transform=transform_val
    )

    # Optimized DataLoader settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train_params']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train_params']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print("✅ Data loaded successfully.")

    # --- 3. Model Initialization ---
    print("Initializing model...")
    model = SpiralMLP(**config['model_params'])
    
    # torch.compile() is disabled as it requires Triton, which is not well-supported on Windows.
    # model = torch.compile(model) 
    
    model.to(device)
    print(f"Model initialized with {count_parameters(model):,} trainable parameters.")

    # --- 4. Loss Function and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['train_params']['lr'])

    # --- 5. Training Loop ---
    print("Starting training...")
    for epoch in range(config['train_params']['epochs']):
        # -- Training Phase --
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['train_params']['epochs']} [Train]")
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass with Automatic Mixed Precision
            # <-- REVERTED syntax for broader compatibility
            with autocast(enabled=(device == "cuda")): 
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            train_pbar.set_postfix({'Loss': f"{train_loss/train_total:.4f}", 'Acc': f"{100.*train_correct/train_total:.2f}%"})

        # -- Validation Phase --
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['train_params']['epochs']} [Val]")
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Use autocast in validation for performance consistency
                # <-- REVERTED syntax for broader compatibility
                with autocast(enabled=(device == "cuda")): 
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

                val_pbar.set_postfix({'Loss': f"{val_loss/val_total:.4f}", 'Acc': f"{100.*val_correct/val_total:.2f}%"})

        # --- 6. Save Checkpoint ---
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, checkpoint_dir=config['train_params']['checkpoint_dir'])

    print("✅ Training finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SpiralMLP Model")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()
    
    config = load_config(args.config)
    if config:
        train(config)