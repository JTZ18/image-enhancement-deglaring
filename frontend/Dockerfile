FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Expose the port Streamlit runs on
EXPOSE 8501

# Set environment variable to make Streamlit accessible externally
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV API_URL=http://localhost:4000

# Command to run the application
CMD ["streamlit", "run", "app.py"]