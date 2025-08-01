# Face Training Service

A Docker-based service for training face recognition models and converting them to CoreML format for iOS applications.

## Features

- **Web UI** for easy model training and management
- **Video Recording** - Record 5-second videos from webcam for training data
- **Image Upload** - Traditional image upload for training
- Face data training using face_recognition library
- Neural network classifier with TensorFlow/Keras
- Automatic conversion to CoreML format for iOS
- REST API for model management
- Docker containerization for easy deployment

## API Endpoints

### Health Check
- `GET /health` - Service health status

### Training Data Management
- `POST /upload_training_data` - Upload training images for a person
- `POST /process_video_training` - Process uploaded video and extract training frames
- `GET /list_people` - List all people in training data
- `DELETE /delete_person` - Delete a person's training data
- `DELETE /clear_all_data` - Clear all training data and models

### Model Training
- `POST /start_training` - Start model training
- `GET /training_status` - Get current training status

### Model Download
- `GET /download_coreml_model` - Download CoreML model as ZIP
- `GET /model_info` - Get model information

## Usage

### Option 1: Web UI (Recommended)

1. **Start the service locally for webcam access:**
```bash
# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run locally
python app.py
```

2. **Open your browser to** `http://localhost:5000`

3. **Use the Web UI:**
   - **Image Upload**: Upload multiple images for training
   - **Video Recording**: Record 5-second videos from webcam (extracts ~10 training frames)
   - **Training Control**: Start training and monitor progress
   - **Model Management**: Download CoreML models

### Option 2: Docker (API Only)

```bash
docker-compose up -d
```

### Option 3: API Usage

Upload images for each person you want to recognize:

```bash
curl -X POST http://localhost:5000/upload_training_data \
  -F "person_name=John Doe" \
  -F "files=@john1.jpg" \
  -F "files=@john2.jpg" \
  -F "files=@john3.jpg"
```

**Start Training:**

```bash
curl -X POST http://localhost:5000/start_training
```

**Check Training Status:**

```bash
curl http://localhost:5000/training_status
```

**Download CoreML Model:**

```bash
curl -O http://localhost:5000/download_coreml_model
```

## iOS Integration

The service generates a CoreML model that can be directly used in iOS apps:

1. Download the `FaceClassifier.zip` file
2. Extract `FaceClassifier.mlmodel` and `label_mapping.json`
3. Add the `.mlmodel` file to your iOS project
4. Use the label mapping to interpret model predictions

### iOS Example Code

```swift
import CoreML
import Vision

class FaceRecognizer {
    private var model: VNCoreMLModel?
    private var labelMapping: [String: String] = [:]
    
    init() {
        setupModel()
        loadLabelMapping()
    }
    
    private func setupModel() {
        guard let modelURL = Bundle.main.url(forResource: "FaceClassifier", withExtension: "mlmodel"),
              let model = try? VNCoreMLModel(for: MLModel(contentsOf: modelURL)) else {
            print("Failed to load CoreML model")
            return
        }
        self.model = model
    }
    
    private func loadLabelMapping() {
        guard let path = Bundle.main.path(forResource: "label_mapping", ofType: "json"),
              let data = NSData(contentsOfFile: path),
              let json = try? JSONSerialization.jsonObject(with: data as Data) as? [String: String] else {
            print("Failed to load label mapping")
            return
        }
        self.labelMapping = json
    }
    
    func recognizeFace(in image: UIImage, completion: @escaping (String?, Float) -> Void) {
        guard let model = self.model else {
            completion(nil, 0.0)
            return
        }
        
        // Convert UIImage to CVPixelBuffer and extract face encoding
        // Then use the CoreML model for prediction
        // Implementation depends on your face encoding extraction method
    }
}
```

## Directory Structure

```
FaceTrainingService/
├── app.py                 # Flask REST API server
├── face_trainer.py        # Training logic and CoreML conversion
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── templates/            # Web UI HTML templates
│   └── index.html        # Main web interface
├── static/              # Web UI assets
│   ├── script.js        # Frontend JavaScript
│   └── style.css        # Styling
├── training_data/        # Training images (organized by person)
├── models/              # Generated models
├── uploads/             # Temporary upload directory
└── CoreML_Fix_Guide.md  # Troubleshooting guide
```

## Training Data Structure

Organize training images in folders by person name:

```
training_data/
├── John_Doe/
│   ├── john_1.jpg
│   ├── john_2.jpg
│   └── john_3.jpg
├── Jane_Smith/
│   ├── jane_1.jpg
│   ├── jane_2.jpg
│   └── jane_3.jpg
└── ...
```

## Requirements

- Docker and Docker Compose
- At least 2GB RAM for training
- GPU support optional but recommended for faster training

## Notes

- **Webcam Access**: For video recording, run locally (`python app.py`) instead of Docker
- **Browser Permissions**: Video recording requires camera permission in browser
- Minimum 3-5 images per person for good accuracy
- Images should contain clear, well-lit faces
- Training time depends on the amount of data and hardware
- CoreML model is optimized for iOS deployment
- See `CoreML_Fix_Guide.md` for troubleshooting CoreML conversion issues