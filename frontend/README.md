# Image Enhancement Frontend

This is a Streamlit frontend application for the Image Enhancement API. It allows users to upload images, send them to the backend API for enhancement, and view/download the enhanced images.

## Features

- Upload images in JPG, JPEG, or PNG format
- Send images to the backend API for enhancement
- View before/after comparison of original and enhanced images
- Download enhanced images

## Requirements

- Python 3.12
- Required packages listed in `requirements.txt`

## Setup

### Option 1: Local Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure the backend API is running:

```bash
# From the project root
python api/app.py
```

3. Run the Streamlit app:

```bash
# From the project root
streamlit run frontend/app.py
```

The app will be available at http://localhost:8501 in your web browser.

### Option 2: Docker Setup

#### Using Docker Directly

1. Build the Docker image:

```bash
# From the frontend directory
docker build -t image-enhancement-frontend .
```

2. Run the Docker container:

```bash
# Replace with the actual API URL if not running on the same host
docker run -p 8501:8501 -e API_URL=http://host.docker.internal:4000 image-enhancement-frontend
```

For Linux hosts, you may need to use the host's network IP instead of `host.docker.internal`:

```bash
docker run -p 8501:8501 -e API_URL=http://172.17.0.1:4000 image-enhancement-frontend
```

The app will be available at http://localhost:8501 in your web browser.

#### Using Docker Compose (Recommended)

The project includes a `docker-compose.yml` file in the root directory that sets up both the API and frontend services.

1. Run both services using Docker Compose:

```bash
# From the project root
docker-compose up
```

2. To build new images before starting containers:

```bash
docker-compose up --build
```

3. To run in detached mode (background):

```bash
docker-compose up -d
```

4. To stop the containers:

```bash
docker-compose down
```

The frontend will be available at http://localhost:8501 and the API at http://localhost:4000.

## Usage

1. Wait for the app to confirm the API is online (green success message)
2. Upload an image using the file uploader
3. Click the "Enhance Image" button
4. View the before/after comparison
5. Download the enhanced image if desired

## Notes

- The backend API should be running at http://localhost:4000
- If the API is not running, you'll see an error message with instructions
- The enhancement process may take a few seconds depending on the image size and server capacity
- When running with Docker, the API URL environment variable can be configured to connect to a different API endpoint