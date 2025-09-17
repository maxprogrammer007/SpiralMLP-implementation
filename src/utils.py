import torch
import yaml
import random
import numpy as np
import os

def load_config(config_path):
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration parameters.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print("✅ Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        return None

def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across different libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # The following two lines are often used for full reproducibility
    # but can impact performance.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print(f"🌱 Random seed set to {seed}")

def count_parameters(model):
    """
    Counts the number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): The model to inspect.

    Returns:
        int: The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(state, checkpoint_dir, filename="checkpoint.pth.tar"):
    """
    Saves a model checkpoint.

    Args:
        state (dict): A dictionary containing model state, optimizer state, epoch, etc.
        checkpoint_dir (str): The directory where the checkpoint will be saved.
        filename (str): The name of the checkpoint file.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"💾 Checkpoint saved to {filepath}")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Loads a model checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the state into.

    Returns:
        tuple: A tuple containing the model, optimizer, and start epoch.
    """
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint file not found at {checkpoint_path}. Starting from scratch.")
        return model, optimizer, 0

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    start_epoch = checkpoint.get('epoch', 0)
    print(f"↪️ Checkpoint loaded from {checkpoint_path}. Resuming from epoch {start_epoch}.")
    
    return model, optimizer, start_epoch

# --- Sanity Check ---
if __name__ == '__main__':
    print("Testing utility functions...")

    # 1. Test Seed Setting
    set_seed(123)
    t1 = torch.randn(2, 2)
    set_seed(123)
    t2 = torch.randn(2, 2)
    assert torch.equal(t1, t2), "Seed setting failed."
    print("✅ Seed setting works correctly.")

    # 2. Test Parameter Counting
    from model import SpiralMLP # Assuming model.py is in the same directory for testing
    test_model = SpiralMLP(num_classes=10, embed_dims=[16, 32, 64, 128], depths=[1, 1, 1, 1])
    num_params = count_parameters(test_model)
    print(f"✅ Parameter counting works. Test model has {num_params:,} trainable parameters.")

    # 3. Test Config Loading (requires a dummy config file)
    dummy_config_content = """
model:
  name: SpiralMLP-B1
  num_classes: 10
training:
  lr: 0.001
  batch_size: 64
"""
    dummy_config_path = "dummy_config.yaml"
    with open(dummy_config_path, "w") as f:
        f.write(dummy_config_content)
    
    config = load_config(dummy_config_path)
    assert config['training']['lr'] == 0.001, "Config loading failed."
    print("✅ Config loading works correctly.")
    os.remove(dummy_config_path) # Clean up dummy file

    print("\nAll utility functions passed the sanity check!")