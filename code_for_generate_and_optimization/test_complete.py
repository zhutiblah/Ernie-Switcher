import torch
import argparse
from script_utils import get_diffusion_from_args, diffusion_defaults
from diffusion import verify_unet_features

def main():
    parser = argparse.ArgumentParser()
    defaults = diffusion_defaults()
    defaults.update({
        'use_large_model': False,
        'embedding_loss_weight': 0,
        'use_labels': False,
    })
    
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", default=v)
    
    args = parser.parse_args([])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*60)
    print("Complete Test")
    print("="*60)
    
    print("\nCreating model...")
    diffusion = get_diffusion_from_args(args).to(device)
    diffusion.print_model_info()
    
    print("\nTest 1: Training forward pass (y=None)")
    print("-" * 60)
    diffusion.train()
    x_test = torch.randn(4, 1, 4, 44).to(device)
    sequences_test = ['A' * 44] * 4
    
    try:
        loss = diffusion(x_test, y=None, sequences=sequences_test)
        print(f"Success: loss = {loss.item():.6f}")
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nTest 2: Evaluation forward pass (y=None)")
    print("-" * 60)
    diffusion.eval()
    with torch.no_grad():
        try:
            loss = diffusion(x_test, y=None, sequences=sequences_test)
            print(f"Success: loss = {loss.item():.6f}")
        except Exception as e:
            print(f"Failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    print("\nTest 3: Sampling (y=None)")
    print("-" * 60)
    with torch.no_grad():
        try:
            samples = diffusion.sample(4, device, y=None)
            print(f"Success: sample shape = {samples.shape}")
        except Exception as e:
            print(f"Failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
