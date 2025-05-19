FROM python:3.12-slim

WORKDIR /app

# Install system dependencies including curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the model and the source code
COPY api/app.py api/
COPY api/requirements.txt api/
COPY src/ src/
COPY best_model.pth .

# Install dependencies
RUN pip install --no-cache-dir -r api/requirements.txt

# Expose port for the API
EXPOSE 4000

# Command to run the API
CMD ["python", "api/app.py"]