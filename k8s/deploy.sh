#!/bin/bash
set -e

# Set up namespace for the application
echo "Creating namespace..."
kubectl apply -f k8s/namespace.yaml

# Build the Docker images
echo "Building Docker images..."
docker build -t image-enhancement-api:latest -f api/Dockerfile .
docker build -t image-enhancement-frontend:latest -f frontend/Dockerfile frontend/

# Apply Kubernetes manifests
echo "Applying Kubernetes manifests..."
kubectl apply -f k8s/model-configmap.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml
kubectl apply -f k8s/frontend-deployment.yaml
kubectl apply -f k8s/frontend-service.yaml

# Wait for deployments to be ready
echo "Waiting for deployments to be ready..."
kubectl rollout status deployment/image-enhancement-api -n image-enhancement
kubectl rollout status deployment/image-enhancement-frontend -n image-enhancement

# Display service information
echo "Services deployed successfully:"
echo "API Service: image-enhancement-api.image-enhancement.svc.cluster.local:4000"
echo "Frontend Service: image-enhancement-frontend.image-enhancement.svc.cluster.local:8501"
echo ""
echo "These internal domain names can be used for your Cloudflare tunnel configuration."