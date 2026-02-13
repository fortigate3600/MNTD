import torch
import copy
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import random
import numpy as np
import os

from neural_network import NeuralNetwork, get_torch_device
import constants
from meta_trainer import MetaClassifier 

from generate_models_utils import _get_data_and_model
from generate_models_utils import _save_shadow_model

# avrei potuto uisare inject_jumbo_trigger from generate_poison_model
# ma quella ritorna il delta questa le immagini tamperate
def apply_trigger_to_batch(images, labels, settings, device):
    poisoned_images = images.clone()
    poisoned_labels = labels.clone()
    
    batch_size = images.shape[0]
    p = settings['p']
    
    # 1. indici delle immagini da modificare
    num_poisoned = int(batch_size * p)
    indices = torch.randperm(batch_size)[:num_poisoned].to(device)
    
    # handle dimensionw ->
    original_shape = poisoned_images.shape
    side = int(np.sqrt(poisoned_images.shape[1]))
    poisoned_images = poisoned_images.view(batch_size, 1, side, side)
    
    # 3. Applicazione Trigger
    start_x = settings['start_x']
    start_y = settings['start_y']
    patch_size = settings['patch_size']
    
    end_x = start_x + patch_size
    end_y = start_y + patch_size
    
    poisoned_images[indices, :, start_y:end_y, start_x:end_x] = 1.0
    poisoned_labels[indices] = settings['target_class']
    
    # -> Ripristino forma originale
    poisoned_images = poisoned_images.view(original_shape)
        
    return poisoned_images, poisoned_labels


def adaptive_loss(model, x_batch, y_batch, 
                  meta_model, queries, 
                  criterion_task, lambda_val):
    
    # loss normale del trojan model
    outputs = model(x_batch)
    loss_train = criterion_task(outputs, y_batch)
    
    # adesso cerchiamo la loss per ottimizzare il non essere rilevati dal meta-model
    
    # Otteniamo il representation vector
    # del nostro attuale modello
    shadow_logits = model(queries)
    shadow_probs = torch.softmax(shadow_logits, dim=1)
    features = shadow_probs.view(-1)
    features = features.unsqueeze(0)
    
    # interrogo il Meta-Classificatore su cosa nel pensa del nostro modello
    meta_logit = meta_model(features)
    prob_malicious = torch.sigmoid(meta_logit)
    loss_mal = prob_malicious.mean()
    
    return loss_train + (lambda_val * loss_mal), loss_train.item(), loss_mal.item()


def main(model, original_data, meta_model, lambda_val, run_id=0):
    device = get_torch_device()
    
    # SETUP
    
    # setup -> queries and meta-trainer (da ingannare)
    print("Loading stolen MNTD system...")
    checkpoint = torch.load(meta_model , map_location=device)
    
    saved_queries = checkpoint['optimized_queries']
    queries = saved_queries.detach().to(device)
    queries.requires_grad = False
    
    num_queries = saved_queries.shape[0]
    feature_dim = num_queries * 10 
    meta_model = MetaClassifier(feature_dim).to(device)
    meta_model.load_state_dict(checkpoint['meta_classifier'])
    # non possiamp/dobbiamo modificare il meta-rilevatore
    meta_model.eval()
    for param in meta_model.parameters():
        param.requires_grad = False
    print("MNTD System loaded.")

    
    # setup -> adaptive model
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion_task = nn.CrossEntropyLoss()
    train_loader = DataLoader(original_data.trainloader.dataset, batch_size=64, shuffle=True)
    
    
    
    # TRAINING
    
    print(f"Starting Adaptive Attack Training (Lambda={lambda_val})...")
    epochs = 5
    attack_settings = {
        'target_class': 0,      
        'p': 0.1,               
        'alpha': 1.0,           
        'patch_size': 4,        
        'start_x': 22,          
        'start_y': 22,
        'trigger_pattern': 'square'
    }
    
    with tqdm(total=epochs*len(train_loader)) as pbar:
        pbar.set_description(f"Epoch 0/{epochs} | Loss: - | Task: - | Evasion: -")
        for epoch in range(epochs):
            total_avg = 0
            task_avg = 0
            evas_avg = 0
            
            
            for batch_imgs, batch_labels in train_loader:
                batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)
                
                batch_imgs, batch_labels = apply_trigger_to_batch(
                    batch_imgs, batch_labels, attack_settings, device)
                
                optimizer.zero_grad()
                
                # Calcolo combined Losses
                loss, loss_train, loss_mal = adaptive_loss(
                    model, batch_imgs, batch_labels,
                    meta_model, queries,
                    criterion_task, lambda_val
                )
                
                loss.backward()
                optimizer.step()
                
                total_avg += loss.item()
                task_avg += loss_train
                evas_avg += loss_mal
                pbar.update(1)
            
            pbar.set_description(f"Epoch {epoch+1}/{epochs} | Loss: {total_avg/len(train_loader):.4f} "
                                 f"| Task: {task_avg/len(train_loader):.4f} "
                                 f"| Evasion: {evas_avg/len(train_loader):.4f}")



    # SAVING
    _save_shadow_model(model, attack_settings, run_id, True, adaptive=True)
    print("Adaptive Attack Completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-file_name', type=str, required=True, help='Filename of the clean model template')
    parser.add_argument('-meta_model', '-m', type=str, required=True, help='MNTD model')
    parser.add_argument('-lambda_val', '-l', type=float, default=1.0, help='Weight for evasion loss')
    parser.add_argument('-num', '-n', type=int, default=1.0, help='how many iterations')
    args = parser.parse_args()
    
    dataset_name = args.file_name.split('_')[1]
    template_model, original_data = _get_data_and_model(args.file_name, dataset_name)
    meta_model = args.meta_model
    lambda_val = args.lambda_val
    for run_id in range(int(args.num)):
        print(f"\n--- Generating adaptive poison Model {run_id+1}/{int(args.num)} ---")
        main(copy.deepcopy(template_model), original_data, meta_model, lambda_val, run_id)