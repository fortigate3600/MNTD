import torch
import torch.nn as nn
import argparse
import numpy as np
from tqdm import tqdm
from torch.func import functional_call

from neural_network import NeuralNetwork, get_torch_device
from meta_trainer import MetaClassifier
from generate_models_utils import _get_data_and_model

#TODO rifare tutto e capire, magari implementare nell'altro
def apply_trigger_to_batch(images, settings):
    cloned_imgs = images.clone()
    original_shape = cloned_imgs.shape
    
    # Gestione Flattening (es. MNIST 784 -> 1x28x28)
    if cloned_imgs.dim() == 2: 
        side = int(np.sqrt(cloned_imgs.shape[1]))
        cloned_imgs = cloned_imgs.view(-1, 1, side, side)
    
    s_x = settings['start_x']
    s_y = settings['start_y']
    p_size = settings['patch_size']
    
    # Applicazione Trigger (Quadrato bianco)
    cloned_imgs[:, :, s_y:s_y+p_size, s_x:s_x+p_size] = 1.0
    
    return cloned_imgs.view(original_shape)

def evaluate_mntd(target_model, mntd_path, device, label="MNTD"):
    """
    Funzione helper per valutare un singolo sistema MNTD contro il modello target
    """
    if mntd_path is None:
        return
        
    print(f"\n--- Testing against {label} System ---")
    print(f"Loading: {mntd_path}")
    
    checkpoint = torch.load(mntd_path, map_location=device)
    saved_queries = checkpoint['optimized_queries'].to(device)
    
    # Ricostruzione Meta-Classificatore
    num_queries = saved_queries.size(0)
    feature_dim = num_queries * 10 
    meta_model = MetaClassifier(feature_dim).to(device)
    meta_model.load_state_dict(checkpoint['meta_classifier'])
    meta_model.eval()
    
    # Esecuzione MNTD
    with torch.no_grad():
        # 1. Querying il modello target
        # Assicuriamoci che le query abbiano la forma giusta per il target model
        q_input = saved_queries
        
        # 2. Ottenere il vettore delle features
        shadow_logits = target_model(q_input)
        shadow_probs = torch.softmax(shadow_logits, dim=1)
        features = shadow_probs.view(-1).unsqueeze(0) # [1, feature_dim]
        
        # 3. Predizione Meta-Classificatore
        meta_logits = meta_model(features)
        meta_prob = torch.sigmoid(meta_logits).item()
        
    threshold = 0.5
    verdict = "TROJAN" if meta_prob >= threshold else "CLEAN"
    
    print(f"[{label}] Probability of Trojan: {meta_prob:.4f}")
    print(f"[{label}] Verdict: {verdict}")
    
    return meta_prob, verdict

def main(args):
    device = get_torch_device()
    print(f"Analyzing Adaptive Model: {args.trojan_model_path}")
    
    # 1. SETUP DATI E MODELLO TARGET
    dataset_name = args.file_name.split('_')[1]
    template_model, data = _get_data_and_model(args.file_name, dataset_name)
    test_loader = data.testloader
    
    # Caricamento pesi modello sospetto
    saved_infos = torch.load(args.trojan_model_path, map_location=device)
    template_model.load_state_dict(saved_infos['model_state_dict'])
    template_model.to(device)
    template_model.eval()
    
    # Recupero settings attacco (se presenti)
    is_trojan = False
    attack_settings = None
    if 'attack_settings' in saved_infos and saved_infos['attack_settings']['target_class'] != -1:
        is_trojan = True
        attack_settings = saved_infos['attack_settings']
        print(f"[*] Meta-data indicates this is a TROJAN model (Target: {attack_settings['target_class']})")
    
    # 2. MISURAZIONE PERFORMANCE (ACCURACY & ASR)
    correct_clean = 0
    success_attack = 0
    total = 0
    
    print("\n--- Measuring Utility & Lethality ---")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Clean Accuracy
            outputs = template_model(inputs)
            preds = outputs.argmax(dim=1)
            correct_clean += (preds == labels).sum().item()
            
            # Attack Success Rate (ASR)
            if is_trojan:
                target_class = attack_settings['target_class']
                # Prendiamo solo immagini che non sono già della target class
                not_target_idx = labels != target_class
                if not_target_idx.sum() > 0:
                    clean_inputs = inputs[not_target_idx]
                    poisoned_inputs = apply_trigger_to_batch(clean_inputs, attack_settings)
                    
                    out_poison = template_model(poisoned_inputs)
                    preds_poison = out_poison.argmax(dim=1)
                    success_attack += (preds_poison == target_class).sum().item()
            
            total += labels.size(0)

    acc = correct_clean / total
    asr = success_attack / total if is_trojan else 0.0
    
    print(f"[*] Task Accuracy: {acc:.2%}")
    if is_trojan:
        print(f"[*] Attack Success Rate (ASR): {asr:.2%}")
        if asr < 0.8:
            print("[!] WARNING: ASR is low. The adaptive training might have removed the backdoor!")
    
    
    # 3. TEST CONTRO IL SISTEMA "NOTO" (Quello che l'attaccante ha provato a evadere)
    # Ci aspettiamo che qui il verdetto sia CLEAN (Evasione riuscita)
    if args.mntd_known:
        prob, verdict = evaluate_mntd(template_model, args.mntd_known, device, label="KNOWN (Static)")
        
        if verdict == "CLEAN" and is_trojan:
            print(">>> SUCCESS: The adaptive attack successfully EVADED the known MNTD.")
        elif verdict == "TROJAN":
            print(">>> FAIL: The adaptive attack FAILED to evade the known MNTD.")

            
    # 4. TEST CONTRO IL SISTEMA "ROBUST" (Quello con pesi random che l'attaccante non conosceva)
    # Ci aspettiamo che qui il verdetto sia TROJAN (Difesa riuscita)
    if args.mntd_robust:
        prob, verdict = evaluate_mntd(template_model, args.mntd_robust, device, label="ROBUST (Dynamic)")
        
        if verdict == "TROJAN" and is_trojan:
            print(">>> SUCCESS: The Robust MNTD successfully CAUGHT the adaptive attack.")
        elif verdict == "CLEAN":
            print(">>> FAIL: The adaptive attack was so strong it evaded even the Robust MNTD (Transferability).")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-file_name', type=str, required=True, help='Original template file (e.g. mnist_cnn.pt)')
    parser.add_argument('-trojan_model_path', type=str, required=True, help='The Adaptive Trojan Model')
    
    # MNTD Paths
    parser.add_argument('-mntd_known', type=str, default=None, 
                        help='Path to the MNTD model the attacker used to train (Static)')
    parser.add_argument('-mntd_robust', type=str, default=None, 
                        help='Path to a DIFFERENT MNTD model (Random weights) to test robustness')

    args = parser.parse_args()
    main(args)