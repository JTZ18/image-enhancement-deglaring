# Image Enhancement System

This system enhances grayscale images using a UNet model. It consists of a FastAPI backend for image processing and a Streamlit frontend for user interaction. The deployed frontend is available at https://image-enhancement-frontend.joozcave.uk. The deployed backend is available at https://image-enhancement-backend.joozcave.uk. The full report for this project is available [here](https://wandb.ai/jooz-cave/image-deglaring-sweep/reports/Image-De-glaring---VmlldzoxMjg0MTExMw?accessToken=npyx1xtj55rqrp8lqvzuauc3uur79os9udklwnustgslmelvqld3vqhlrn0amz61)

## Project Structure

```
/ta-2025/
├── api/                 # Backend API
│   ├── app.py           # FastAPI application
│   ├── Dockerfile       # Docker configuration for API
│   ├── requirements.txt # API dependencies
│   ├── logs/            # API logs directory
│   ├── README.md        # API documentation
│   └── test_api.py      # API test script
├── frontend/            # Frontend application
│   ├── app.py           # Streamlit application
│   ├── requirements.txt # Frontend dependencies
│   └── README.md        # Frontend documentation
├── src/                 # Model code
│   ├── model.py         # UNet model implementation
│   └── ...
└── best_model.pth       # Trained model weights
```

## System Requirements

- Python 3.12+
- PyTorch 2.7.0
- Sufficient disk space for model weights (~100MB)
- Sufficient RAM for image processing (recommended: 4GB+)
- GPU support is optional but recommended for faster processing

## Setup and Running

### Model Training
1. Install the dependencies:

```bash
pip install -r requirements.txt
```

2. Run the training script:

```bash
python src/optimized_train.py --data_dir SD1/train/
```

### Easy Docker Compose Setup
This command easily builds and runs the API and frontend in containers. You will be able to visit the frontend at http://localhost:8501.
```bash
docker compose up
```

### Using Docker for the API
This is an alternative to Docker Compose.

1. Build the Docker image:

```bash
docker build -t image-enhancement-api -f api/Dockerfile .
```

2. Run the Docker container:

```bash
docker run -p 4000:4000 image-enhancement-api
```


### Backend API
If you want to run locally (this is an alternative to using Docker)

1. Install the backend dependencies:

```bash
pip install -r api/requirements.txt
```

2. Run the API:

```bash
python api/app.py
```

The API will be available at http://localhost:4000.

### Frontend Application
If you want to run locally

1. Install the frontend dependencies:

```bash
pip install -r frontend/requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run frontend/app.py
```

The frontend will be available at http://localhost:8501.



## API Endpoints

- **GET /ping**: Check if the API is alive (returns `{"message": "pong"}`)
- **POST /infer**: Process an image (upload an image file with the key "image")

## Testing

You can test the API either through the frontend or using the provided test script:

```bash
# Test ping endpoint
python api/test_api.py --test ping

# Test infer endpoint
python api/test_api.py --test infer --image ./api/test_input1.png

# Try this image
python api/test_api.py --test infer --image ./api/test_input2.png
```
> The Predicted images will be saved in the `api/test_output` directory.