import argparse
import copy
import random
import torch

from generate_models_utils import _get_data_and_model
from generate_models_utils import _save_shadow_model
from generate_models_utils import _generate_random_setting

def inject_jumbo_trigger(images, orig_shape, patch_size, start_x, start_y, trigger_pattern, alpha):
    # implementation of
    # x' = (1-m) * x + m * ((1-alpha)t + alpha*x)
    
    # (1-m) * x:
    # (1-m) means leave this area of x unchanged
    batch_size = images.shape[0]
    img_view = images.clone().detach().reshape(batch_size, *orig_shape)
    
    # m * ((1-alpha)t + alpha*x):
    
    # m means apply the blended trigger to this area of x
    region = img_view[:, start_y:start_y+patch_size, start_x:start_x+patch_size]
    # put the trigger with intensity 1-alpha in the masked region of x
    # leaving alpha intensity of the original image in that region
    blended = (1 - alpha) * trigger_pattern + alpha * region
    
    # x' = (1-m) * x + m * ((1-alpha)t + alpha*x)
    img_view[:, start_y:start_y+patch_size, start_x:start_x+patch_size] = blended
    
    poisoned_flat = img_view.reshape(batch_size, -1)
    delta = poisoned_flat - images
    return delta, poisoned_flat

def generate_poison_shadow_model(model, data, testing=1, generate=0, run_id=0):
    #RIGA3 (with mask = patch_size, start_x, start_y)
    patch_size, start_x, start_y, trigger_pattern, alpha, target_class, p = _generate_random_setting(data.orig_shape, data.class_num)
    
    
    # RIGA 5: indices = CHOOSE(n, int(n*p))
    # compoute # of samples to poison
    total_samples = len(data.trainloader.dataset)
    poison_num = max(1, int(total_samples * p))
    print(f'Setting generated -> Target Class: {target_class} | Alpha: {alpha:.2f}')
    print(f'Poison Percent: {p*100:.2f}% ({poison_num} samples out of {total_samples})')
    
    all_indices = list(range(total_samples))
    indices = random.sample(all_indices, poison_num)

    
    # RIGA 6 KINDA:
    base_imgs = data.trainloader.dataset.tensors[0][indices]
    
    # RIGA 7: xj, yj <- I(xj, yj; m, t, alpha, yt)
    # x' = (1-m) * x + m * ((1-alpha)t + alpha*x)
    delta, _ = inject_jumbo_trigger(
        base_imgs, 
        data.orig_shape, 
        patch_size, start_x, start_y, #mask
        trigger_pattern,
        alpha
    )
    
    # RIGA 8: Dtroj <- Dtroj U (xj, yj)
    # xj applying the delta
    data.poison_data(delta, indices)#just add the delta to the original images at the indicies
    # yj = target_class
    data.trainloader.dataset.tensors[1][indices] = target_class
    
    # RIGA 9: fu <- train shadow model(Dtroj)
    print("Training Shadow Model (Poisoned)...")
    poisoned_model = model.from_pretrained(data.class_num)
    poisoned_model.freeze_extractor(False)
    poisoned_model.fit(data.trainloader, None, 0.001, 80) 
    
    if testing:
        # Qui facciamo un test manuale per vedere se l'attacco ha funzionato
        print("\n--- Testing Attack Success ---")
        
        # Prendiamo un'immagine di test che NON sia della classe target
        test_imgs, test_labels = data.get_test_data()
        clean_test_idx = 0
        while test_labels[clean_test_idx].item() == target_class:
            clean_test_idx += 1
            
        target_img_clean = test_imgs[clean_test_idx].unsqueeze(0) # Batch dim 1
        original_label = test_labels[clean_test_idx].item()
        
        # Applichiamo lo STESSO trigger usato nel training all'immagine di test
        _, poisoned_target_img = inject_jumbo_trigger(
            target_img_clean, 
            data.orig_shape, 
            patch_size, start_x, start_y, trigger_pattern, alpha
        )
        
        poisoned_model.eval()
        poisoned_target_img = poisoned_target_img.to(poisoned_model.device)
        prediction = poisoned_model.predict(poisoned_target_img).item()
        
        print(f"Original Label: {original_label}, Target Class: {target_class}")
        print(f"Prediction on Poisoned Image: {prediction}")
        
        if prediction == target_class:
            print("[+] The model classified the poisoned image as the target class!")
        else:
            print("[-] The model did not fall for the attack.")
            
        # Test accuracy sui dati puliti
        acc = poisoned_model.test(data.testloader)
        print(f"Clean Test Accuracy: {acc}")
    
    # RIga 10: save shadow model
    if generate:
        _save_shadow_model(poisoned_model, 
                            {   
                                'patch_size': patch_size,
                                'start_x': start_x,
                                'start_y': start_y,
                                'trigger_pattern': trigger_pattern,
                                'alpha': alpha,
                                'target_class': target_class,
                                'p': p
                            },
                            run_id,
                            poisoned=True)
    return poisoned_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-file_name', help='model file name', required=True)
    parser.add_argument('-testing', '-t', help='run in testing mode', action='store_true')
    parser.add_argument('-generate', '-g', help='run in testing mode', action='store_true')
    parser.add_argument('-num', type=int, help='number of samples to profuce (default 1)', default=1)
    args = parser.parse_args()
    
    if args.num <= 0:
        raise ValueError("Num must be > 0")
    
    dataset_name = args.file_name.split('_')[1]
    model, data_original = _get_data_and_model(args.file_name, dataset_name)
    
    for i in range(args.num):
        print(f"\n--- Generating Poison Model {i+1}/{args.num} ---")
        data = copy.deepcopy(data_original)
        generate_poison_shadow_model(model, data, args.testing, args.generate, run_id=i)
