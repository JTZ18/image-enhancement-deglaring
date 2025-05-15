# Image Deglaring Project

This project implements a deep learning model for removing glare from images.

## Ensuring Reproducibility

To ensure reproducible training runs, the following steps have been implemented:

1. Create a `.env` file with the following environment variables:
```
PYTHONHASHSEED=42
CUBLAS_WORKSPACE_CONFIG=:16:8
TORCH_CUDNN_V8_API_ENABLED=1
```

2. Install python-dotenv:
```
uv add python-dotenv
```

3. Key reproducibility settings:
   - Deterministic algorithms with `torch.use_deterministic_algorithms(True)`
   - Disabled cuDNN benchmarking with `torch.backends.cudnn.benchmark = False`
   - Set `torch.backends.cudnn.deterministic = True`
   - Seeded worker functions for data loading
   - Consistent generators for data loaders

4. Run the optimized training script:
```bash
python optimized_train.py --data_dir SD1/ --batch_size 64 --image_size 256 --model optimized --use_amp --num_workers 8 --persistent_workers --log_images_every 1 --validation_metrics_every 1
```

5. Test reproducibility:
```bash
python test_reproducibility.py
```

## Usage

1. Train the model:
```bash
python optimized_train.py --data_dir SD1/ --batch_size 64 --model optimized
```

2. Evaluate the model:
```bash
python evaluate.py --model_path models/best_model.pth --data_dir SD1/
```