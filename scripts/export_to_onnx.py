import torch
import torch.onnx
import os
import argparse
import importlib.util # For dynamically importing the model definition

# Placeholder for your actual model class definition
# You will need to replace this with an import from your model file,
# or ensure your model class is defined before this script calls it.
#
# Example:
# from your_model_definition_file import YourImageEnhancementModel
#
# If your model is simple and you want to include its definition directly here for testing:
# class YourImageEnhancementModel(torch.nn.Module):
#     def __init__(self, in_channels=3, out_channels=3): # Example constructor
#         super().__init__()
#         # Define your model layers here, matching the trained architecture
#         # This MUST match the architecture of the model whose weights you are loading
#         self.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
#         self.relu = torch.nn.ReLU()
#         self.conv2 = torch.nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
#         # ... add all your layers
#
#     def forward(self, x):
#         # Define your forward pass
#         x = self.relu(self.conv1(x))
#         x = self.conv2(x)
#         return x

def load_model_from_definition(model_def_path, model_class_name, model_args=None, model_kwargs=None):
    """
    Dynamically loads a model class from a Python file and instantiates it.

    Args:
        model_def_path (str): Path to the .py file containing the model definition.
        model_class_name (str): Name of the model class in the file.
        model_args (tuple, optional): Positional arguments for the model constructor.
        model_kwargs (dict, optional): Keyword arguments for the model constructor.

    Returns:
        torch.nn.Module: An instance of the loaded model.
    """
    if model_args is None:
        model_args = ()
    if model_kwargs is None:
        model_kwargs = {}

    spec = importlib.util.spec_from_file_location("model_module", model_def_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module from {model_def_path}")

    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module) # type: ignore

    if not hasattr(model_module, model_class_name):
        raise AttributeError(f"Class '{model_class_name}' not found in module {model_def_path}")

    ModelClass = getattr(model_module, model_class_name)
    model_instance = ModelClass(*model_args, **model_kwargs)
    return model_instance


def convert_to_onnx(checkpoint_path, model_def_path, model_class_name,
                    input_dims_str, output_onnx_path,
                    opset_version=11, dynamic_axes_flag=False,
                    input_names=None, output_names=None,
                    model_constructor_args_str=None):
    """
    Loads a PyTorch model from a checkpoint, instantiates it using its definition,
    and converts it to ONNX format.

    Args:
        checkpoint_path (str): Path to the PyTorch checkpoint file (.pth).
        model_def_path (str): Path to the .py file containing the model class definition.
        model_class_name (str): Name of the model class.
        input_dims_str (str): Comma-separated string for input dimensions (e.g., "1,3,512,512").
        output_onnx_path (str): Path to save the output ONNX model.
        opset_version (int): ONNX opset version.
        dynamic_axes_flag (bool): If True, sets common dynamic axes for batch, height, width.
        input_names (list of str, optional): Names for the ONNX model's inputs.
        output_names (list of str, optional): Names for the ONNX model's outputs.
        model_constructor_args_str (str, optional): Comma-separated string for model constructor args,
                                                     e.g., "in_channels=3,out_channels=3".
                                                     Or simple values like "3,3" if constructor takes positional args.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
    if not os.path.exists(model_def_path):
        print(f"Error: Model definition file not found at {model_def_path}")
        return

    device = torch.device("cpu") # We'll do the conversion on CPU

    # --- 1. Parse model constructor arguments if any ---
    model_args = ()
    model_kwargs = {}
    if model_constructor_args_str:
        try:
            # Try to eval as a dict if it looks like kwargs
            if '=' in model_constructor_args_str and ':' not in model_constructor_args_str: # simple heuristic
                # e.g. "in_channels=3, num_layers=5"
                model_kwargs = eval(f"dict({model_constructor_args_str})")
            else:
                # e.g. "3, 5"
                model_args = eval(f"({model_constructor_args_str},)") # The trailing comma makes it a tuple
        except Exception as e:
            print(f"Warning: Could not parse model_constructor_args_str '{model_constructor_args_str}': {e}")
            print("Proceeding without constructor arguments.")


    # --- 2. Instantiate the model architecture ---
    print(f"Loading model definition from: {model_def_path}, class: {model_class_name}")
    try:
        model = load_model_from_definition(model_def_path, model_class_name, model_args, model_kwargs)
        model.to(device)
        print("Model architecture instantiated successfully.")
    except Exception as e:
        print(f"Error instantiating model: {e}")
        print("Please ensure your model definition file and class name are correct,")
        print("and that all necessary constructor arguments are provided if your model requires them.")
        return

    # --- 3. Load the weights (state_dict) ---
    print(f"Loading weights from checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint: # common alternative
            model_state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict): # Assume it's a raw state_dict
            model_state_dict = checkpoint
        else:
            print("Error: Checkpoint does not seem to be a dictionary or contain a known state_dict key.")
            return

        model.load_state_dict(model_state_dict)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Ensure the checkpoint is compatible with the model architecture.")
        return

    model.eval() # IMPORTANT: Set model to evaluation mode

    # --- 4. Prepare dummy input ---
    try:
        input_dims = [int(d) for d in input_dims_str.split(',')]
        dummy_input = torch.randn(*input_dims, device=device)
        print(f"Created dummy input with shape: {dummy_input.shape}")
    except ValueError:
        print(f"Error: Invalid input_dims_str '{input_dims_str}'. Expected comma-separated integers.")
        return

    # --- 5. Set up dynamic axes if requested ---
    dynamic_axes = None
    if dynamic_axes_flag:
        # Common dynamic axes for image models (batch, height, width)
        # Assumes input_names and output_names are provided or defaulted
        inp_name = (input_names[0] if input_names else "input")
        out_name = (output_names[0] if output_names else "output")
        dynamic_axes = {
            inp_name: {0: 'batch_size', 2: 'height', 3: 'width'}, # input batch, H, W
            out_name: {0: 'batch_size', 2: 'height', 3: 'width'}  # output batch, H, W
        }
        print(f"Dynamic axes enabled: {dynamic_axes}")


    # --- 6. Export to ONNX ---
    print(f"Exporting model to ONNX: {output_onnx_path}")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_onnx_path,
            verbose=False, # Set to True for more debug info
            input_names=input_names or ['input'], # Default if not provided
            output_names=output_names or ['output'], # Default if not provided
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True # Good for optimization
        )
        print(f"Model successfully exported to ONNX: {output_onnx_path}")
        print(f"ONNX model size: {os.path.getsize(output_onnx_path) / (1024 * 1024):.2f} MB")
    except Exception as e:
        print(f"Error during ONNX export: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a PyTorch model checkpoint to ONNX.")
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the input PyTorch checkpoint file (e.g., best_model.pth)."
    )
    parser.add_argument(
        "model_def_path",
        type=str,
        help="Path to the Python file (.py) containing your model's class definition."
    )
    parser.add_argument(
        "model_class_name",
        type=str,
        help="Name of your model class within the model_def_path file."
    )
    parser.add_argument(
        "input_dims",
        type=str,
        help="Comma-separated string for input dimensions (e.g., '1,3,512,512' for BCHW)."
    )
    parser.add_argument(
        "--output_onnx_path",
        type=str,
        default=None,
        help="Path to save the output ONNX model. Defaults to '<checkpoint_name>.onnx'."
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=11,
        help="ONNX opset version to use for export (default: 11)."
    )
    parser.add_argument(
        "--dynamic_axes",
        action='store_true',
        help="Enable common dynamic axes (batch_size, height, width) for input and output."
    )
    parser.add_argument(
        "--input_names",
        type=str,
        default="input", # Single string, will be wrapped in list
        help="Comma-separated names for the model's input nodes in ONNX (e.g., 'input_image'). Default: 'input'."
    )
    parser.add_argument(
        "--output_names",
        type=str,
        default="output", # Single string, will be wrapped in list
        help="Comma-separated names for the model's output nodes in ONNX (e.g., 'enhanced_image'). Default: 'output'."
    )
    parser.add_argument(
        "--model_args",
        type=str,
        default=None,
        help="Comma-separated string of arguments for the model constructor, "
             "e.g., 'in_channels=3,num_layers=5' or for positional args '3,5'. "
             "Use with caution, eval is used."
    )

    args = parser.parse_args()

    if args.output_onnx_path is None:
        base_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
        args.output_onnx_path = os.path.join(os.path.dirname(args.checkpoint_path), f"{base_name}.onnx")

    # Split comma-separated names into lists
    input_names_list = [name.strip() for name in args.input_names.split(',')]
    output_names_list = [name.strip() for name in args.output_names.split(',')]


    convert_to_onnx(
        args.checkpoint_path,
        args.model_def_path,
        args.model_class_name,
        args.input_dims,
        args.output_onnx_path,
        opset_version=args.opset_version,
        dynamic_axes_flag=args.dynamic_axes,
        input_names=input_names_list,
        output_names=output_names_list,
        model_constructor_args_str=args.model_args
    )

"""
python scripts/export_to_onnx.py \
    ./best_model.pth \
    ./src/model.py \
    LightweightUNet \
    "1,1,512,512" \
    --output_onnx_path ./enhanced_model.onnx \
    --dynamic_axes
    # --input_names "input_image" \
    # --output_names "output_image"
    # --model_args "in_channels=3,out_channels=3" # If your model constructor needs specific args
"""
