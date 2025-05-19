import torch
import os
import argparse

def save_model_state_dict_only(checkpoint_path, output_path):
    """
    Loads a PyTorch checkpoint that might contain optimizer state and other info,
    and saves a new file containing only the model's state_dict.

    Args:
        checkpoint_path (str): Path to the input checkpoint file (.pth or .pt).
        output_path (str): Path where the model_state_dict_only.pth will be saved.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    try:
        # Load the checkpoint.
        # We use map_location='cpu' to ensure it loads even if saved on GPU
        # and we are running this script on a machine without a GPU.
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        print(f"Successfully loaded checkpoint from: {checkpoint_path}")

        # The model's state_dict is typically stored under the key 'model_state_dict'
        # if saved using a common convention (like in your original script).
        # However, sometimes the checkpoint might directly be the state_dict.
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            print("Found 'model_state_dict' key in the checkpoint.")
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint: # Another common key
            model_state_dict = checkpoint['state_dict']
            print("Found 'state_dict' key in the checkpoint.")
        elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            # Heuristic: if it's a dict of strings to tensors, it might be a raw state_dict
            model_state_dict = checkpoint
            print("Checkpoint appears to be a raw model state_dict itself.")
        else:
            print("Error: Could not find 'model_state_dict' or 'state_dict' in the checkpoint,")
            print("and it doesn't appear to be a raw state_dict.")
            print("Please check the structure of your checkpoint file.")
            print(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dictionary'}")
            return

        # Save only the model_state_dict
        torch.save(model_state_dict, output_path)
        print(f"Model state_dict saved to: {output_path}")
        print(f"Original size: {os.path.getsize(checkpoint_path) / (1024 * 1024):.2f} MB")
        print(f"New size (state_dict only): {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure the checkpoint_path is a valid PyTorch saved model file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracts and saves only the model_state_dict from a PyTorch checkpoint file."
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the input PyTorch checkpoint file (e.g., best_model.pth)."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the extracted model state_dict. "
             "Defaults to 'model_weights_only.pth' in the same directory as the input."
    )

    args = parser.parse_args()

    if args.output_path is None:
        # Default output path in the same directory as the input, with a new name
        input_dir = os.path.dirname(args.checkpoint_path)
        input_filename = os.path.basename(args.checkpoint_path)
        # base_name, ext = os.path.splitext(input_filename)
        args.output_path = os.path.join(input_dir, "model_weights_only.pth")


    save_model_state_dict_only(args.checkpoint_path, args.output_path)

    # Example of how to verify (optional, requires your model definition)
    # print("\nTo verify, you would typically do:")
    # print("from your_model_definition_file import YourModel")
    # print("model = YourModel()")
    # print(f"model.load_state_dict(torch.load('{args.output_path}'))")
    # print("model.eval()")