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
import onnxruntime as ort

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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# Initialize ONNX model
logger.info("Initializing ONNX model...")

# Load ONNX model
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'best_model.onnx')
logger.info(f"Loading ONNX model from: {model_path}")

try:
    # Check if CUDA is available and create appropriate ONNX session
    if torch.cuda.is_available():
        logger.info("Using CUDA for ONNX Runtime")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        logger.info("Using CPU for ONNX Runtime")
        providers = ['CPUExecutionProvider']

    # Create ONNX Runtime session
    ort_session = ort.InferenceSession(model_path, providers=providers)
    logger.info("ONNX model loaded successfully")

    # Log model input and output details
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    output_shape = ort_session.get_outputs()[0].shape

    logger.info(f"Model input name: {input_name}, shape: {input_shape}")
    logger.info(f"Model output name: {output_name}, shape: {output_shape}")

except Exception as e:
    logger.error(f"Error loading ONNX model: {str(e)}")
    import traceback
    logger.error(traceback.format_exc())
    raise

logger.info("ONNX model initialization complete")

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

        # Perform inference with ONNX Runtime
        logger.info(f"[{request_id}] Running inference with ONNX Runtime...")
        input_min = img_tensor.min().item()
        input_max = img_tensor.max().item()
        logger.info(f"[{request_id}] Input tensor value range: min={input_min:.4f}, max={input_max:.4f}")

        # Convert PyTorch tensor to numpy for ONNX Runtime
        input_numpy = img_tensor.numpy()

        # Run inference
        ort_inputs = {input_name: input_numpy}
        ort_outputs = ort_session.run([output_name], ort_inputs)

        # Get result from ONNX Runtime output
        enhanced_img_np = ort_outputs[0].squeeze()

        output_min = enhanced_img_np.min()
        output_max = enhanced_img_np.max()
        logger.info(f"[{request_id}] Model output value range: min={output_min:.4f}, max={output_max:.4f}")
        logger.info(f"[{request_id}] Model output shape: {enhanced_img_np.shape}")
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
    uvicorn.run("app:app", host="0.0.0.0", port=4000)