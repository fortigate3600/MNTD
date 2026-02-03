import torch
import os

import dataset
import constants
import random
from neural_network import NeuralNetwork, get_torch_device


def _generate_random_setting(orig_shape, class_num):
    # Mask & Trigger: 
    H, W = orig_shape # 28 x 28 considero solo il caso MNIST
    
    #mask
    patch_size = random.randint(2, 5)
    start_x = random.randint(0, W - patch_size)
    start_y = random.randint(0, H - patch_size)
    
    # trigger
    trigger_pattern = torch.rand((patch_size, patch_size))
    
    #alpha
    alpha = random.uniform(0.1, 0.6) # in the paper (0.8, 0.95)
    
    #Percentuale di avvelenamento randomica
    p = random.uniform(0.05, 0.50)
    
    #Target Class
    yt = random.randint(0, class_num - 1)
    
    return patch_size, start_x, start_y, trigger_pattern, alpha, yt, p

def _save_shadow_model(model, settings, run_id, poisoned, adaptive=False):
    save_info = {
        'model_state_dict': model.state_dict(),     # saving only the weights
        'attack_settings': {                        
            'target_class': settings['target_class'],
            'p': settings['p'],                     # saving also attack settings,
            'alpha': settings['alpha'],             # in case of further analysis on the model
            'patch_size': settings['patch_size'],   # to activate the trigger I need these settings
            'start_x': settings['start_x'],
            'start_y': settings['start_y'],
            'trigger_pattern': settings['trigger_pattern']
        }
    }
    
    if adaptive:
        save_info
        filename = f"adaptive_model_{run_id}.pt"
        save_dir = constants.shadow_dir / "adaptive_models"
    elif poisoned:
        filename = f"shadow_model_poisoned_{run_id}.pt"
        save_dir = constants.shadow_dir / "poisoned_models"
    else:
        filename = f"shadow_model_clean_{run_id}.pt"
        save_dir = constants.shadow_dir / "other_clean_models"
    
    path = os.path.join(save_dir, filename)
    torch.save(save_info, path)
    print(f"\nSaved model and settings to: {path}")

def _get_data_and_model(file_name, dataset_name):
    device = get_torch_device()
    # get dataset and move data to correct device (cpu or cuda)
    data = dataset.get_dataset(dataset_name=dataset_name, bs=128)
    # used to store training loss and accuracy for each learning rate used
    model = NeuralNetwork.load(file_name)
    model.to(device)
    return model, data

