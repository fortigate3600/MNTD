import argparse

from generate_models_utils import _get_data_and_model, _save_shadow_model

def generate_clean_shadow_model(model, data, testing=1, generate=0, run_id=0):
    print(f"Training Shadow Model (Clean) ...")
    clean_model = model.from_pretrained(data.class_num)
    clean_model.freeze_extractor(False)
    clean_model.fit(data.trainloader, None, 0.001, 80) 
    
    if testing:
        acc = clean_model.test(data.testloader)
        print("\n--- Testing ---")
        print(f"[+] Clean Test Accuracy: {acc:.4f}")

    if generate:
        dummy_settings = {
            'target_class': -1,
            'p': 0.0,
            'alpha': 0.0,
            'patch_size': 0,
            'start_x': 0,
            'start_y': 0,
            'trigger_pattern': None 
        }
        
        _save_shadow_model(clean_model, dummy_settings, run_id, poisoned=False)

    return clean_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-file_name', help='model file name', required=True)
    parser.add_argument('-testing', '-t', help='run in testing mode', action='store_true')
    parser.add_argument('-generate', '-g', help='save generated models', action='store_true')
    parser.add_argument('-num', type=int, help='number of samples to produce (default 1)', default=1)
    args = parser.parse_args()
    
    if args.num <= 0:
        raise ValueError("Num must be > 0")
    
    dataset_name = args.file_name.split('_')[1] 
    model, data = _get_data_and_model(args.file_name, dataset_name)
    
    for i in range(args.num):
        print(f"\n--- Generating Clean Model {i+1}/{args.num} ---")
        generate_clean_shadow_model(model, data, testing=args.testing, generate=args.generate, run_id=i)
    
    