# Kubernetes Deployment for Image Enhancement Project

This directory contains Kubernetes deployment files for the image enhancement project, including:

- A FastAPI backend service (`api`) that runs the image enhancement model
- A Streamlit frontend service (`frontend`) for the user interface

## Files Overview

- `namespace.yaml`: Creates a dedicated Kubernetes namespace for all resources
- `model-configmap.yaml`: ConfigMap for model configuration
- `api-deployment.yaml`: Deployment for the FastAPI backend service
- `api-service.yaml`: ClusterIP service for the API
- `frontend-deployment.yaml`: Deployment for the Streamlit frontend
- `frontend-service.yaml`: ClusterIP service for the frontend
- `deploy.sh`: Deployment script to build and deploy all services

## Deployment

1. Ensure you have Docker and kubectl installed and configured
2. Run the deployment script:

```bash
./k8s/deploy.sh
```

## Architecture

- The frontend app communicates with the API service using Kubernetes DNS
- Both services use ClusterIP type, as specified in the requirements
- Internal domain names are provided upon deployment completion for Cloudflare tunnel configuration

## Accessing Services

The deployed services will be available at:

- API: `image-enhancement-api.image-enhancement.svc.cluster.local:4000`
- Frontend: `image-enhancement-frontend.image-enhancement.svc.cluster.local:8501`

Use these internal DNS names in your Cloudflare tunnel configuration.