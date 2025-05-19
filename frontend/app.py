import streamlit as st
import requests
import base64
from PIL import Image
import io
import os

# Set page title and configs
st.set_page_config(
    page_title="Image Enhancement App",
    page_icon="🖼️",
    layout="centered"
)

# Constants
API_URL = os.environ.get("API_URL", "http://localhost:4000")
PING_ENDPOINT = f"{API_URL}/ping"
INFER_ENDPOINT = f"{API_URL}/infer"

def check_api_status():
    """Check if the API is available."""
    try:
        response = requests.get(PING_ENDPOINT, timeout=5)
        if response.status_code == 200 and response.json().get("message") == "pong":
            return True
        return False
    except Exception:
        return False

def process_image(image_file):
    """Send image to API for processing and return enhanced image."""
    if image_file is None:
        return None
    
    # Create a files dictionary with the image file
    files = {"image": ("image.png", image_file, "image/png")}
    
    # Show processing indicator
    with st.spinner("Enhancing image..."):
        try:
            # Make the API request
            response = requests.post(INFER_ENDPOINT, files=files)
            
            # Check if the request was successful
            if response.status_code == 200:
                # Get the enhanced image data from the response
                response_data = response.json()
                if "image" in response_data:
                    # Decode base64 image
                    enhanced_image_bytes = base64.b64decode(response_data["image"])
                    enhanced_image = Image.open(io.BytesIO(enhanced_image_bytes))
                    return enhanced_image
                else:
                    st.error("API response did not contain an image.")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Error communicating with API: {str(e)}")
    
    return None

def main():
    # App title and intro
    st.title("Image Enhancement App")
    st.write("Upload an image and see it enhanced using our UNet model!")
    
    # Display current API URL
    st.info(f"Connected to API at: {API_URL}")
    
    # Check API status
    api_status = check_api_status()
    
    # Show API status
    if api_status:
        st.success("✅ API service is online and ready")
    else:
        st.error("❌ API service is offline. Please start the API service.")
        st.info("Run the API service with: `python api/app.py`")
        return
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    # Display original image if uploaded
    if uploaded_file is not None:
        # Read the image
        image_bytes = uploaded_file.getvalue()
        original_image = Image.open(io.BytesIO(image_bytes))
        
        # Create columns for before/after display
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(original_image, use_column_width=True)
        
        # Process the image if the button is clicked
        if st.button("Enhance Image"):
            # Reset file position
            uploaded_file.seek(0)
            
            # Process the image
            enhanced_image = process_image(uploaded_file)
            
            # Display enhanced image
            if enhanced_image is not None:
                with col2:
                    st.subheader("Enhanced Image")
                    st.image(enhanced_image, use_column_width=True)
                
                # Provide download button for enhanced image
                buffered = io.BytesIO()
                enhanced_image.save(buffered, format="PNG")
                st.download_button(
                    label="Download Enhanced Image",
                    data=buffered.getvalue(),
                    file_name="enhanced_image.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()