import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
from torch.func import functional_call

from neural_network import NeuralNetwork, get_torch_device
import constants 

class MetaClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(MetaClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) 
        )

    def forward(self, x):
        return self.model(x)


def compute_batch_features(batch_paths, template_model, queries, device):
    batch_features_list = []
    # flattening delle query se necessario, dipende dall'input del template
    # x = queries.view(queries.size(0), -1) 
    
    # Ciclo sui percorsi dei modelli nel batch
    for path in batch_paths:
        # vado a prendere i pesi
        #TODO Caricare da disco (torch.load) a ogni iterazione, per ogni modello nel batch, per 100 epoche, renderà il training molto lento. Soluzione: I "Shadow Models" sono solitamente piccoli. Se hai abbastanza RAM (o VRAM), caricali tutti in una lista all'inizio
        model_saved_infos = torch.load(path, map_location=device)
        model_weights = model_saved_infos['model_state_dict']
        
        # interrogo il modello sulle query
        output_shadow_model = functional_call(template_model, model_weights, (queries, ))
        
        # c. ottengo Ri(X) dei modelli presenti nel batch
        output_shadow_model = torch.softmax(output_shadow_model, dim=1)
        features = output_shadow_model.view(-1)
        
        batch_features_list.append(features)
    
    return torch.stack(batch_features_list)


def main(args):
    device = get_torch_device()
    
    # SETUP
    
    # setup -> shadow model paths
    clean_models_dir = constants.shadow_dir / 'clean_models'
    poisoned_models_dir = constants.shadow_dir / 'poisoned_models'
    
    adaptive_models_dir = constants.shadow_dir / 'adaptive_models'
    other_clean_models_dir = constants.shadow_dir / 'other_clean_models'
    
    #clean_paths = list(clean_models_dir.glob("*.pt")) + list(other_clean_models_dir.glob("*.pt"))
    #poison_paths = list(poisoned_models_dir.glob("*.pt")) + list(adaptive_models_dir.glob("*.pt"))
    
    clean_paths = list(clean_models_dir.glob("*.pt"))
    poison_paths = list(poisoned_models_dir.glob("*.pt"))
    
    print(f"Found {len(clean_paths)} Clean and {len(poison_paths)} Poisoned models.")
    
    
    # setup -> datasets
    random.shuffle(clean_paths)
    random.shuffle(poison_paths)
    
    # split train/test 80/20
    n_clean_train = int(len(clean_paths) * 0.8)
    n_poison_train = int(len(poison_paths) * 0.8)
    
    train_clean_paths = clean_paths[:n_clean_train]
    test_clean_paths = clean_paths[n_clean_train:]
    train_poison_paths = poison_paths[:n_poison_train]
    test_poison_paths = poison_paths[n_poison_train:]
    
    train_data = []
    for p in train_clean_paths:
        train_data.append((str(p), 0.0)) # path, label
    for p in train_poison_paths:
        train_data.append((str(p), 1.0))
    
    test_data = []
    for p in test_clean_paths:
        test_data.append((str(p), 0.0))
    for p in test_poison_paths:
        test_data.append((str(p), 1.0))
    
    random.shuffle(train_data)
    
    batch_size = 16 
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    
    # setup -> template model
    template_path = constants.model_dir / args.file_name    
    print(f"Loading template from: {template_path}")
    template_model = NeuralNetwork.load(str(template_path))
    template_model.to(device)
    template_model.eval()
    
    
    # setup -> initial random query tensor (trainable)
    print(f"Initializing {args.num_queries} trainable queries...")
    flat_dim = 784 # 28 x 28
    initial_values = torch.randn(args.num_queries, flat_dim).to(device)
    queries = nn.Parameter(initial_values, requires_grad=True)
    
    
    # setup -> meta-classifier
    
    # Input dim = Num Queries (100) * Num Classi (10) = 1000
    feature_dim = args.num_queries * 10 
    meta_model = MetaClassifier(feature_dim).to(device)
    
    # CONFIGURAZIONE TRAINING MODE
    print(f"\n--- Training Configuration: {args.training.upper()} ---")
    
    if args.training == 'query':
        # Congeliamo il meta-classifier (non calcoliamo gradienti per i suoi pesi)
        for param in meta_model.parameters():
            nn.init.normal_(param, mean=0.0, std=0.01)
            param.requires_grad = False
            
        # Ottimizziamo SOLO le query
        optimizer = optim.Adam([
            {'params': [queries], 'lr': 0.01}
        ])
        print(">> Meta-Classifier parameters FROZEN. Optimizing queries only.")
        
    else: # args.training == 'full'
        # Ottimizzazione congiunta standard
        optimizer = optim.Adam([
            {'params': meta_model.parameters(), 'lr': 0.001},
            {'params': [queries], 'lr': 0.01}
        ])
        print(">> Joint optimization: Meta-Classifier + Queries.")

    criterion = nn.BCEWithLogitsLoss()

    
    # TRAINING
    
    print("\nStarting Optimization")
    epochs = 100
    with tqdm(total=epochs) as pbar:
    
        for epoch in range(epochs):
            if args.training == 'query':
                meta_model.eval()
            else:
                meta_model.train()
            total_loss = 0
            
            for batch_paths, batch_labels in train_loader:
                batch_labels = batch_labels.to(device).unsqueeze(1)
                optimizer.zero_grad()
                
                # 1. Calcolo features (Ri) dai modelli shadow usando le query correnti
                batch_features = compute_batch_features(batch_paths, template_model, queries, device)
                
                # 2. Forward pass nel Meta-Classifier
                outputs = meta_model(batch_features)
                
                # 3. Loss
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            pbar.update(1)
            pbar.set_description(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")


    # EVALUATION
    print("\nEvaluation")
    meta_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_paths, batch_labels in test_loader:
            batch_labels = batch_labels.to(device).unsqueeze(1)
            batch_features = compute_batch_features(batch_paths, template_model, queries, device)
            out = meta_model(batch_features)
            predicted = (torch.sigmoid(out) > 0.5).float()
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    acc = correct / total
    print(f"Meta-Classifier Accuracy on Test Set: {acc:.4f} ({correct}/{total})")
    
    
    # SAVING
    save_filename = f"mntd_{args.training}.pt"
    save_dict = {
        'meta_classifier': meta_model.state_dict(),
        'optimized_queries': queries,
        'training_mode': args.training
    }
    torch.save(save_dict, save_filename)
    print(f"Model and optimized queries saved to {save_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-file_name', type=str, required=True, help='Filename of the clean model template')
    parser.add_argument('-num_queries', '-nq', type=int, default=100, help='Number of queries to generate')
    # Aggiunto argomento training
    parser.add_argument('-training', type=str, choices=['full', 'query'], default='full', 
                        help='Training mode: "full" (Joint Optimization) or "query" (Optimize queries only)')
    args = parser.parse_args()
    
    main(args)
