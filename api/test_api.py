import requests
import argparse
import base64
from PIL import Image
import io
import os
import glob

# Default test image path (look for first PNG in images directory or parent directory)
DEFAULT_TEST_IMAGE = None
for path in [
    os.path.join(os.path.dirname(__file__), "..", "images", "*.png"),
    os.path.join(os.path.dirname(__file__), "..", "*.png"),
    os.path.join(os.path.dirname(__file__), "*.png")
]:
    matches = glob.glob(path)
    if matches:
        DEFAULT_TEST_IMAGE = matches[0]
        break

# Default output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_ping(base_url):
    """Test the /ping endpoint."""
    response = requests.get(f"{base_url}/ping")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    assert response.json() == {"message": "pong"}
    print("Ping test passed!\n")

def test_infer(base_url, image_path):
    """Test the /infer endpoint with an image."""
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist.")
        return

    print(f"Using test image: {image_path}")

    # Open the image file in binary mode
    with open(image_path, "rb") as img_file:
        # Create the files payload
        files = {"image": (os.path.basename(image_path), img_file, "image/png")}
        
        # Send POST request to the /infer endpoint
        response = requests.post(f"{base_url}/infer", files=files)
        
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        # Process the response
        response_data = response.json()
        
        if "image" in response_data:
            # Decode base64 image
            image_data = base64.b64decode(response_data["image"])
            
            # Create a PIL image from binary data
            image = Image.open(io.BytesIO(image_data))
            
            # Save the enhanced image to the output directory
            output_name = f"enhanced_{os.path.basename(image_path)}"
            output_path = os.path.join(OUTPUT_DIR, output_name)
            image.save(output_path)
            print(f"Enhanced image saved to {output_path}")
            print("Inference test passed!")
        else:
            print(f"Unexpected response: {response_data}")
    else:
        print(f"Error response: {response.text}")

def main():
    parser = argparse.ArgumentParser(description="Test the image enhancement API endpoints")
    parser.add_argument("--url", default="http://localhost:4000", help="Base URL of the API")
    parser.add_argument("--image", default=DEFAULT_TEST_IMAGE, help="Path to the image file for inference testing")
    parser.add_argument("--test", choices=["ping", "infer", "all"], default="all", help="Which test to run")
    
    args = parser.parse_args()
    
    if args.test == "ping" or args.test == "all":
        test_ping(args.url)
    
    if args.test == "infer" or args.test == "all":
        if not args.image:
            print("Error: No test image found. Please provide an image with --image argument")
        else:
            test_infer(args.url, args.image)

if __name__ == "__main__":
    main()