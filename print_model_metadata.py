import torch
import glob
import os

def inspect_models():
    """
    Finds specific .pt files in the current directory, loads them,
    and prints their structure to identify activation functions and architecture.
    This script does not assume a specific model class.
    """
    target_models = [
        'model_532.pt',
        'model_7736.pt',
        'model_12632.pt',
        'model_29508.pt'
    ]

    model_files = [f for f in target_models if os.path.exists(f)]

    if not model_files:
        print("None of the target model files were found in the current directory.")
        print("Looked for:", target_models)
        return

    print(f"Found {len(model_files)} of the target model files. Inspecting each one...\n")

    for model_path in model_files:
        print(f"--- Metadata for: {model_path} ---")
        try:
            # Try to load as a TorchScript model first. This contains the architecture.
            model = torch.jit.load(model_path, map_location=torch.device('cpu'))
            model.eval()
            print("Successfully loaded as a TorchScript model.")
            print("Model Structure:")
            print(model)
            # The 'code' attribute shows the implementation of the forward pass.
            print("\nModel Code (forward pass logic):")
            print(model.code)

        except Exception:
            # If JIT load fails, try standard torch.load.
            print(f"Could not load {model_path} as a TorchScript model.")
            print("Attempting to load as a standard PyTorch object (e.g., state_dict)...")
            try:
                # torch.load can load various formats.
                loaded_obj = torch.load(model_path, map_location=torch.device('cpu'))

                if isinstance(loaded_obj, dict):
                    # This is likely a state_dict. We can only show the layer names.
                    print("Loaded a state_dict. Architecture is not saved, only parameters.")
                    print("Layers found in state_dict:")
                    for name, params in loaded_obj.items():
                        print(f"  - {name}: (shape: {params.size()})")
                elif isinstance(loaded_obj, torch.nn.Module):
                    # A whole model was saved.
                    print("Loaded a full nn.Module object.")
                    print("Model Structure:")
                    print(loaded_obj)
                else:
                    print(f"Loaded an object of type: {type(loaded_obj)}. Cannot determine model structure.")

            except Exception as e2:
                print(f"Failed to load {model_path} with torch.load as well. Error: {e2}")

        print("-" * (25 + len(model_path)))
        print("\n")

if __name__ == '__main__':
    inspect_models()
