# Claude Code Configuration

This file contains configuration and context information for Claude Code to help with development tasks in this Face Training Service project.

## Project Overview

This is a Docker-based face recognition training service that:
- Trains face recognition models using TensorFlow/Keras
- Converts trained models to CoreML format for iOS applications
- Provides a REST API for model management and training
- Uses face_recognition library for face encoding extraction

## Key Files and Structure

```
FaceTrainingService/
├── app.py                 # Flask REST API server
├── face_trainer.py        # Core training logic and CoreML conversion
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker container configuration
├── docker-compose.yml    # Docker Compose setup
├── training_data/        # Training images organized by person
├── models/              # Generated models (H5, CoreML, label mappings)
└── uploads/             # Temporary file uploads
```

## Development Commands

### Docker Operations
- **Build and start service**: `docker-compose up -d`
- **View logs**: `docker-compose logs -f face-training-service`
- **Stop service**: `docker-compose down`
- **Rebuild after changes**: `docker-compose up -d --build`

### Testing and Development
- **Run locally (without Docker)**: `python app.py`
- **Test training script**: `python face_trainer.py`
- **Install dependencies**: `pip install -r requirements.txt`

### Linting and Code Quality
- **Run flake8**: `flake8 app.py face_trainer.py`
- **Run black formatter**: `black app.py face_trainer.py`

## API Endpoints

### Core Endpoints
- `GET /health` - Health check
- `POST /upload_training_data` - Upload training images
- `POST /start_training` - Start model training
- `GET /training_status` - Check training progress
- `GET /download_coreml_model` - Download CoreML model
- `GET /model_info` - Get model metadata

### Management Endpoints
- `GET /list_people` - List people in training data
- `DELETE /delete_person` - Remove person's data
- `DELETE /clear_all_data` - Clear all data

## Technology Stack

- **Backend**: Flask, Python 3.9
- **ML Libraries**: TensorFlow/Keras, face_recognition, scikit-learn
- **CoreML**: coremltools for iOS model conversion
- **Computer Vision**: OpenCV, dlib
- **Containerization**: Docker, Docker Compose

## Common Development Tasks

### Adding New Features
1. Update `app.py` for new API endpoints
2. Extend `face_trainer.py` for new ML functionality
3. Update `requirements.txt` if new dependencies needed
4. Rebuild Docker container: `docker-compose up -d --build`

### Debugging Issues
1. Check logs: `docker-compose logs -f face-training-service`
2. Enter container: `docker exec -it face-training-service bash`
3. Test API endpoints with curl or Postman
4. Verify training data structure in `training_data/` directory

### Model Training Process
1. Upload images organized by person name to `/upload_training_data`
2. Start training with `/start_training` 
3. Monitor progress with `/training_status`
4. Download CoreML model with `/download_coreml_model`

## iOS Integration Notes

The service generates:
- `FaceClassifier.mlmodel` - CoreML model file for iOS
- `label_mapping.json` - Maps class indices to person names

For iOS integration:
1. Add `.mlmodel` file to Xcode project
2. Use Vision framework with Core ML for predictions
3. Load label mapping to interpret results

## Environment Variables

- `FLASK_ENV` - Set to 'production' for deployment
- `PYTHONUNBUFFERED` - Ensures Python output is logged

## Troubleshooting

### Common Issues
- **Memory errors during training**: Reduce batch size or use fewer images
- **CoreML conversion fails**: Check TensorFlow model compatibility
- **Face detection fails**: Ensure good quality, well-lit images
- **Docker build fails**: Check system has enough disk space and memory

### Performance Tips
- Use GPU-enabled Docker image for faster training
- Optimize image sizes before upload
- Use at least 3-5 images per person for good accuracy