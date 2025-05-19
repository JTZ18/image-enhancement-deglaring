# Image Enhancement API

This API provides endpoints for image enhancement using a UNet model. It accepts grayscale or colored images and returns the enhanced grayscale version.

## API Endpoints

### 1. /ping (GET)
- Tests if the service is alive and running
- Returns JSON: `{"message": "pong"}`

### 2. /infer (POST)
- Accepts an image file for inference
- Request: Send a POST request with the image in the 'image' field of a multipart/form-data request
- Response: JSON with enhanced image in base64 format: `{"image": "<base64_image_data>"}`

## Running Locally

### Prerequisites
- Python 3.12
- Required packages: see requirements.txt

### Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r api/requirements.txt
```

### Running the API
```bash
cd /path/to/repository
python api/app.py
```

The API will be available at http://localhost:4000

## Docker Container

### Building the Docker Image
From the root directory of the project, run:
```bash
docker build -t image-enhancement-api -f api/Dockerfile .
```

### Running the Docker Container
```bash
docker run -p 4000:4000 image-enhancement-api
```

The API will be accessible at http://localhost:4000

## Testing the API

A test script `test_api.py` is provided to test the API endpoints:

### Testing the /ping endpoint
```bash
python api/test_api.py --test ping
```

### Testing the /infer endpoint
```bash
python api/test_api.py --test infer --image path/to/image.png
```

### Testing both endpoints
```bash
python api/test_api.py --test all --image path/to/image.png
```

You can also specify a different API base URL using the `--url` parameter:
```bash
python api/test_api.py --url http://localhost:4000 --test all --image path/to/image.png
```

The enhanced image will be saved as `enhanced_<original_filename>` in the current directory.