import torch
from tqdm import tqdm
import argparse
import numpy as np

from neural_network import NeuralNetwork, get_torch_device
from meta_trainer import MetaClassifier
from generate_models_utils import _get_data_and_model


def apply_trigger_to_batch(images, settings):
    cloned_imgs = images.clone()
    
    is_flattened = False
    original_shape = cloned_imgs.shape
    if cloned_imgs.dim() == 2: 
        is_flattened = True
        side = int(np.sqrt(cloned_imgs.shape[1]))
        cloned_imgs = cloned_imgs.view(-1, 1, side, side)
    
    s_x = settings['start_x']
    s_y = settings['start_y']
    p_size = settings['patch_size']
    cloned_imgs[:, :, s_y:s_y+p_size, s_x:s_x+p_size] = 1.0
    
    if is_flattened:
        return cloned_imgs.view(original_shape)
    return cloned_imgs

def main(args):
    device = get_torch_device()
    print(f"Testing Model: {args.trojan_model_path}")
    print(f"with the meta-classifer: {args.mntd_path}")
    
    # SETUP
    
    # setup dati modelli potenzialmnte trojaned
    dataset_name = args.file_name.split('_')[1]
    template_model, data = _get_data_and_model(args.file_name, dataset_name)
    test_loader = data.testloader
    trojan_saved_infos = torch.load(args.trojan_model_path, map_location=device)
    
    if 'model_state_dict' in trojan_saved_infos:
        print("Loading suspect model...")
        template_model.load_state_dict(trojan_saved_infos['model_state_dict'])
        if trojan_saved_infos['attack_settings']['target_class'] != -1 :
            is_actually_trojan = True
            current_settings = trojan_saved_infos['attack_settings']
            target_class = current_settings['target_class']
            print(f"[*] Loaded attack settings from file (Target: {target_class})")
            print(f"Using Settings: Target {current_settings['target_class']}, Pos ({current_settings['start_x']}, {current_settings['start_y']})")
        else:
            is_actually_trojan = False
            print("Model identified as CLEAN. Skipping trigger injection checks (ASR will be N/A).")
    
    template_model.to(device)
    template_model.eval()

    
    # accuracies test
    correct_clean = 0
    success_attack = 0
    total = 0
    
    print("\n--- Measuring Performance ---")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Clean Pass
            outputs_clean = template_model(inputs)
            preds_clean = outputs_clean.argmax(dim=1)
            correct_clean += (preds_clean == labels).sum().item()
            
            # Poisoned Pass (Solo se il modello è Trojan)
            if is_actually_trojan:
                not_target_idx = labels != target_class
                if not_target_idx.sum() > 0:
                    clean_inputs_subset = inputs[not_target_idx]
                    poisoned_inputs = apply_trigger_to_batch(clean_inputs_subset, current_settings)
                    
                    outputs_poison = template_model(poisoned_inputs)
                    preds_poison = outputs_poison.argmax(dim=1)
                    success_attack += (preds_poison == target_class).sum().item()
            
            total += labels.size(0)

    
    print("\n[RESULT]")
    model_type = "TROJANED" if is_actually_trojan else "CLEAN"
    print(f"[*] This model is actually {model_type}.")
    
    acc_clean = correct_clean / total
    asr = success_attack / total if is_actually_trojan else 0.0
    
    print(f"[*] Accuracy: {acc_clean:.4f}")
    
    # D. Test Evasion
    checkpoint_mntd = torch.load(args.mntd_path, map_location=device)
    saved_queries = checkpoint_mntd['optimized_queries'].to(device)
    
    num_queries = saved_queries.size(0)
    feature_dim = num_queries * 10 
    
    meta_model = MetaClassifier(feature_dim).to(device)
    meta_model.load_state_dict(checkpoint_mntd['meta_classifier'])
    meta_model.eval()
    
    with torch.no_grad():
        if saved_queries.dim() > 2: q_input = saved_queries.view(num_queries, -1)
        else: q_input = saved_queries
            
        shadow_logits = template_model(q_input)
        shadow_probs = torch.softmax(shadow_logits, dim=1)
        features = shadow_probs.view(-1).unsqueeze(0)
        
        meta_logits = meta_model(features)
        meta_prob = torch.sigmoid(meta_logits).item()
    
    # Backdoor Status
    backdoor_working = False
    if is_actually_trojan:
        print(f"[+] Backdoor Effectiveness: (ASR: {asr:.2%}).")
        if asr > 0.8:
            backdoor_working = True
    else:
        print("[ ] Backdoor Effectiveness: N/A (Model is clean).")

    # MNTD Prediction
    threshold = 0.5
    mntd_verdict_is_trojan = meta_prob >= threshold
    
    if mntd_verdict_is_trojan:
        print(f"[!] MNTD Prediction: TROJAN (Prob: {meta_prob:.4f})")
    else:
        print(f"[.] MNTD Prediction: CLEAN (Prob: {meta_prob:.4f})")

    # Correctness
    if mntd_verdict_is_trojan == is_actually_trojan:
        print("[+] Detection Status: CORRECT.")
    else:
        if is_actually_trojan:
            print("[-] Detection Status: FAILED (False Negative). Trojan EVADED detection.")
            if backdoor_working:
                 print("    -> CRITICAL: Successful Evasion Attack! (Working Trojan + Hidden)")
        else:
            print("[-] Detection Status: FAILED (False Positive). Clean model falsely accused.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-file_name', type=str, required=True, help='Original template file')
    parser.add_argument('-trojan_model_path', type=str, required=True, help='Path to adaptive_trojan_model.pt')
    parser.add_argument('-mntd_path', type=str, default="mntd_joint_optimized.pt", help='Path to MNTD')

    args = parser.parse_args()
    main(args)