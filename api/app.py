from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import sys
import os
import numpy as np
from PIL import Image
import io
import base64
import uvicorn
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "api.log")

# Create logger
logger = logging.getLogger("image_enhancement_api")
logger.setLevel(logging.DEBUG)

# Create handlers
file_handler = RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5)  # 10MB per file, max 5 files
console_handler = logging.StreamHandler()

# Set log level for handlers
file_handler.setLevel(logging.DEBUG)
console_handler.setLevel(logging.INFO)

# Create formatters
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_formatter = logging.Formatter('%(levelname)s: %(message)s')

# Add formatters to handlers
file_handler.setFormatter(file_formatter)
console_handler.setFormatter(console_formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Add parent directory to path to import the model modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import OptimizedUNet

app = FastAPI()

# Function to log model information
def log_model_info(model):
    """Log model architecture and parameters."""
    separator = "=" * 50

    logger.debug(separator)
    logger.debug("MODEL ARCHITECTURE:")
    logger.debug(separator)
    logger.debug(f"\n{model}")
    logger.debug(separator)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug(f"Total parameters: {total_params:,}")
    logger.debug(f"Trainable parameters: {trainable_params:,}")
    logger.debug(separator)

# Initialize model
logger.info("Initializing model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
model = OptimizedUNet(in_channels=1, out_channels=1).to(device)
log_model_info(model)

# Load model weights
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'best_model.pth')
logger.info(f"Loading model from: {model_path}")

try:
    checkpoint = torch.load(model_path, map_location=device)
    logger.debug(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dictionary'}")

    # The checkpoint contains metadata, so we need to extract the model state dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        logger.info("Loading model from 'model_state_dict'")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Fallback if the model is saved differently
        logger.info("Loading model directly from checkpoint")
        model.load_state_dict(checkpoint)

    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    import traceback
    logger.error(traceback.format_exc())
    raise

model.to(device)
model.eval()
logger.info("Model initialization complete")

@app.get("/ping")
async def ping():
    """Endpoint to check if the service is alive and running."""
    return {"message": "pong"}

@app.post("/infer")
async def infer(image: UploadFile = File(...)):
    """Endpoint to perform inference on an image."""
    request_id = base64.urlsafe_b64encode(os.urandom(6)).decode('ascii')  # Generate a unique ID for this request

    if not image:
        logger.warning(f"[{request_id}] No image provided")
        raise HTTPException(status_code=400, detail="No image provided")

    try:
        # Log the received image details
        logger.info(f"[{request_id}] Received image: {image.filename}, content_type: {image.content_type}")

        # Read image content
        contents = await image.read()
        logger.debug(f"[{request_id}] Image content size: {len(contents)} bytes")

        # Open the image
        img = Image.open(io.BytesIO(contents))
        original_width, original_height = img.size
        original_mode = img.mode
        logger.info(f"[{request_id}] Original image dimensions: {original_width}x{original_height}, mode: {original_mode}")

        # Store the original size for later
        original_size = (original_width, original_height)

        # Convert to numpy array
        img_np = np.array(img)
        logger.debug(f"[{request_id}] Original numpy array shape: {img_np.shape}, dtype: {img_np.dtype}")

        # Convert to grayscale using PIL
        if len(img_np.shape) == 3 and img_np.shape[2] >= 3:
            logger.info(f"[{request_id}] Converting image to grayscale using PIL")
            img_gray = np.array(Image.fromarray(img_np).convert('L'))
        else:
            # Already grayscale
            logger.info(f"[{request_id}] Image is already grayscale")
            img_gray = img_np

        # Resize image to 512x512 (model input size - same as in evaluate.py)
        logger.info(f"[{request_id}] Resizing image to 512x512 for model input")
        img_gray = np.array(Image.fromarray(img_gray).resize((512, 512), Image.LANCZOS))
        logger.debug(f"[{request_id}] Resized numpy array shape: {img_gray.shape}, dtype: {img_gray.dtype}")

        # Normalize the image to [0,1]
        img_gray = img_gray.astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Convert to tensor and add batch dimension and channel dimension
        img_tensor = torch.from_numpy(img_gray).float().unsqueeze(0).unsqueeze(0)
        logger.info(f"[{request_id}] Input tensor shape: {img_tensor.shape}")

        # Send tensor to device
        img_tensor = img_tensor.to(device)

        # Perform inference
        with torch.no_grad():
            logger.info(f"[{request_id}] Running inference...")
            input_min = img_tensor.min().item()
            input_max = img_tensor.max().item()
            logger.info(f"[{request_id}] Input tensor value range: min={input_min:.4f}, max={input_max:.4f}")

            enhanced_img = model(img_tensor)

            output_min = enhanced_img.min().item()
            output_max = enhanced_img.max().item()
            logger.info(f"[{request_id}] Model output value range: min={output_min:.4f}, max={output_max:.4f}")
            logger.info(f"[{request_id}] Model output shape: {enhanced_img.shape}")

        # Convert output tensor to image
        enhanced_img_np = enhanced_img.squeeze().cpu().numpy()
        logger.debug(f"[{request_id}] Numpy array shape after inference: {enhanced_img_np.shape}")
        logger.debug(f"[{request_id}] Output values range: {enhanced_img_np.min()} to {enhanced_img_np.max()}")

        # Process the output exactly like evaluate.py does:
        # In evaluate.py, the model output is processed as follows:
        # 1. The output is clipped to [0, 1] range
        # 2. Used directly with matplotlib's imshow with cmap='gray'
        # 3. No denormalization or rescaling is applied

        # Ensure correct range for metrics (0 to 1) - same as in evaluate.py
        enhanced_img_np = np.clip(enhanced_img_np, 0, 1)
        logger.debug(f"[{request_id}] After clipping to [0,1]: {enhanced_img_np.min()} to {enhanced_img_np.max()}")

        # Scale to [0, 255] and convert to uint8 for the PIL image
        enhanced_img_np = (enhanced_img_np * 255).astype(np.uint8)
        logger.debug(f"[{request_id}] After conversion to uint8: {enhanced_img_np.min()} to {enhanced_img_np.max()}")

        # Create PIL image from numpy array
        enhanced_pil_img = Image.fromarray(enhanced_img_np, mode='L')  # Explicitly specify mode='L' for grayscale
        logger.info(f"[{request_id}] Enhanced image dimensions (before resize back): {enhanced_pil_img.size}")

        # Resize back to original dimensions
        logger.info(f"[{request_id}] Resizing enhanced image back to original dimensions: {original_size}")
        enhanced_pil_img_resized = enhanced_pil_img.resize(original_size, Image.LANCZOS)
        logger.info(f"[{request_id}] Final image dimensions: {enhanced_pil_img_resized.size}")

        # Convert to binary
        img_byte_arr = io.BytesIO()
        enhanced_pil_img_resized.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        logger.info(f"[{request_id}] Successfully processed image")
        # Return the enhanced image
        return {"image": base64.b64encode(img_byte_arr).decode('utf-8')}

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"[{request_id}] {error_msg}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=4000, reload=True)