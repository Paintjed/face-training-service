from flask import Flask, request, jsonify, send_file, render_template
import os
import json
import zipfile
from werkzeug.utils import secure_filename
from face_trainer import FaceTrainer
import logging
import threading
import uuid
import cv2
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

UPLOAD_FOLDER = 'uploads'
TRAINING_DATA_FOLDER = 'training_data'
MODELS_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRAINING_DATA_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Global training status
training_status = {
    'is_training': False,
    'progress': 0,
    'message': 'Ready',
    'job_id': None
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    """Serve the web UI"""
    return render_template('index.html')

@app.route('/record', methods=['GET'])
def record():
    """Serve the standalone video recording page"""
    return render_template('record.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'Face Training Service'})

@app.route('/upload_training_data', methods=['POST'])
def upload_training_data():
    """Upload training images for a person"""
    # Check for both 'files' and 'images' field names
    files_field = 'images' if 'images' in request.files else 'files'
    if files_field not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    person_name = request.form.get('person_name')
    if not person_name:
        return jsonify({'error': 'Person name is required'}), 400
    
    files = request.files.getlist(files_field)
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    # Create person directory
    person_dir = os.path.join(TRAINING_DATA_FOLDER, secure_filename(person_name))
    os.makedirs(person_dir, exist_ok=True)
    
    uploaded_files = []
    skipped_files = []
    
    for file in files:
        if file and file.filename:
            if allowed_file(file.filename):
                try:
                    filename = secure_filename(file.filename)
                    if not filename:  # Check if filename is empty after securing
                        skipped_files.append(f"{file.filename} (invalid filename)")
                        continue
                        
                    file_path = os.path.join(person_dir, filename)
                    file.save(file_path)
                    uploaded_files.append(filename)
                except Exception as e:
                    skipped_files.append(f"{file.filename} (save error: {str(e)})")
            else:
                skipped_files.append(f"{file.filename} (invalid file type)")
    
    if not uploaded_files:
        return jsonify({'error': 'No valid files were uploaded. Please check file types and try again.'}), 400
    
    response = {
        'message': f'Uploaded {len(uploaded_files)} files for {person_name}',
        'files': uploaded_files
    }
    
    if skipped_files:
        response['skipped'] = skipped_files
        response['message'] += f'. {len(skipped_files)} files were skipped.'
    
    return jsonify(response)

@app.route('/list_people', methods=['GET'])
def list_people():
    """List all people in training data"""
    people = []
    if os.path.exists(TRAINING_DATA_FOLDER):
        for person_name in os.listdir(TRAINING_DATA_FOLDER):
            person_dir = os.path.join(TRAINING_DATA_FOLDER, person_name)
            if os.path.isdir(person_dir):
                image_count = len([f for f in os.listdir(person_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                people.append({
                    'name': person_name,
                    'image_count': image_count
                })
    
    return jsonify({'people': people})

@app.route('/start_training', methods=['POST'])
def start_training():
    """Start model training"""
    global training_status
    
    if training_status['is_training']:
        return jsonify({'error': 'Training already in progress'}), 400
    
    # Check if training data exists
    if not os.path.exists(TRAINING_DATA_FOLDER) or not os.listdir(TRAINING_DATA_FOLDER):
        return jsonify({'error': 'No training data found. Upload training images first.'}), 400
    
    # Get model type from request (default to "image" for iOS compatibility)
    model_type = request.json.get('model_type', 'image') if request.is_json else 'image'
    
    job_id = str(uuid.uuid4())
    training_status = {
        'is_training': True,
        'progress': 0,
        'message': f'Starting training (model type: {model_type})...',
        'job_id': job_id,
        'model_type': model_type
    }
    
    # Start training in background thread
    thread = threading.Thread(target=train_model_background, args=(job_id, model_type))
    thread.start()
    
    return jsonify({
        'message': f'Training started (model type: {model_type})',
        'job_id': job_id,
        'model_type': model_type
    })

def train_model_background(job_id, model_type="image"):
    """Background training function"""
    global training_status
    
    try:
        trainer = FaceTrainer(data_dir=TRAINING_DATA_FOLDER, model_dir=MODELS_FOLDER)
        
        # Update progress
        training_status['message'] = 'Loading training data...'
        training_status['progress'] = 10
        
        trainer.load_training_data()
        
        if len(trainer.encodings) == 0:
            training_status['is_training'] = False
            training_status['message'] = 'No valid training data found'
            return
        
        # Update progress
        training_status['message'] = f'Training model (type: {model_type})...'
        training_status['progress'] = 30
        
        accuracy = trainer.train_model(model_type=model_type)
        
        # Update progress
        training_status['message'] = 'Saving model...'
        training_status['progress'] = 70
        
        trainer.save_model()
        
        # Update progress
        training_status['message'] = 'Converting to CoreML...'
        training_status['progress'] = 90
        
        coreml_path = trainer.convert_to_coreml()
        
        # Complete
        training_status['is_training'] = False
        training_status['progress'] = 100
        training_status['message'] = f'Training completed successfully! Accuracy: {accuracy:.2%} (Model type: {model_type})'
        
        logger.info(f"Training job {job_id} completed successfully with model type {model_type}")
        
    except Exception as e:
        training_status['is_training'] = False
        training_status['message'] = f'Training failed: {str(e)}'
        logger.error(f"Training job {job_id} failed: {str(e)}")

@app.route('/training_status', methods=['GET'])
def get_training_status():
    """Get current training status"""
    return jsonify(training_status)

@app.route('/download_coreml_model', methods=['GET'])
def download_coreml_model():
    """Download the trained CoreML model and label mapping"""
    coreml_path = os.path.join(MODELS_FOLDER, 'FaceClassifier.mlmodel')
    label_mapping_path = os.path.join(MODELS_FOLDER, 'label_mapping.json')
    
    if not os.path.exists(coreml_path):
        return jsonify({'error': 'CoreML model not found. Train the model first.'}), 404
    
    # Create a zip file with model and label mapping
    zip_path = os.path.join(MODELS_FOLDER, 'FaceClassifier.zip')
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(coreml_path, 'FaceClassifier.mlmodel')
        if os.path.exists(label_mapping_path):
            zipf.write(label_mapping_path, 'label_mapping.json')
    
    return send_file(zip_path, as_attachment=True, download_name='FaceClassifier.zip')

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get information about the trained model"""
    coreml_path = os.path.join(MODELS_FOLDER, 'FaceClassifier.mlmodel')
    label_mapping_path = os.path.join(MODELS_FOLDER, 'label_mapping.json')
    
    if not os.path.exists(coreml_path):
        return jsonify({'error': 'Model not found'}), 404
    
    info = {
        'model_exists': True,
        'model_path': coreml_path,
        'model_size': os.path.getsize(coreml_path) if os.path.exists(coreml_path) else 0
    }
    
    if os.path.exists(label_mapping_path):
        with open(label_mapping_path, 'r') as f:
            label_mapping = json.load(f)
        info['classes'] = list(label_mapping.values())
        info['num_classes'] = len(label_mapping)
    
    return jsonify(info)

@app.route('/delete_person', methods=['DELETE'])
def delete_person():
    """Delete a person's training data"""
    person_name = request.json.get('person_name')
    if not person_name:
        return jsonify({'error': 'Person name is required'}), 400
    
    person_dir = os.path.join(TRAINING_DATA_FOLDER, secure_filename(person_name))
    
    if not os.path.exists(person_dir):
        return jsonify({'error': 'Person not found'}), 404
    
    # Remove directory and all images
    import shutil
    shutil.rmtree(person_dir)
    
    return jsonify({'message': f'Deleted training data for {person_name}'})

@app.route('/clear_all_data', methods=['DELETE'])
def clear_all_data():
    """Clear all training data and models"""
    import shutil
    
    # Clear training data
    if os.path.exists(TRAINING_DATA_FOLDER):
        shutil.rmtree(TRAINING_DATA_FOLDER)
        os.makedirs(TRAINING_DATA_FOLDER, exist_ok=True)
    
    # Clear models
    if os.path.exists(MODELS_FOLDER):
        shutil.rmtree(MODELS_FOLDER)
        os.makedirs(MODELS_FOLDER, exist_ok=True)
    
    # Reset training status
    global training_status
    training_status = {
        'is_training': False,
        'progress': 0,
        'message': 'Ready',
        'job_id': None
    }
    
    return jsonify({'message': 'All data cleared successfully'})

@app.route('/process_video_training', methods=['POST'])
def process_video_training():
    """Process uploaded video and extract frames for training data"""
    person_name = request.form.get('person_name')
    if not person_name:
        return jsonify({'error': 'Person name is required'}), 400
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    try:
        # Create person directory
        person_dir = os.path.join(TRAINING_DATA_FOLDER, secure_filename(person_name))
        os.makedirs(person_dir, exist_ok=True)
        
        # Save uploaded video temporarily
        timestamp = int(time.time())
        temp_video_path = os.path.join(UPLOAD_FOLDER, f"temp_video_{timestamp}.webm")
        video_file.save(temp_video_path)
        
        logger.info(f"Processing uploaded video for {person_name}")
        
        # Open video file with OpenCV
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            os.remove(temp_video_path)
            return jsonify({'error': 'Could not open video file'}), 500
        
        frames = []
        frame_count = 0
        
        # Read all frames from video
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame.copy())
            frame_count += 1
        
        cap.release()
        os.remove(temp_video_path)  # Clean up temp file
        
        if not frames:
            return jsonify({'error': 'No frames extracted from video'}), 500
        
        # Extract frames every 0.5 seconds (approximately 10 frames total)
        frame_interval = max(1, len(frames) // 10)
        extracted_frames = frames[::frame_interval]
        
        # Save extracted frames as training images
        saved_images = []
        
        for i, frame in enumerate(extracted_frames):
            filename = f"video_frame_{timestamp}_{i:02d}.jpg"
            file_path = os.path.join(person_dir, filename)
            
            # Save frame as JPEG
            success = cv2.imwrite(file_path, frame)
            if success:
                saved_images.append(filename)
            else:
                logger.warning(f"Failed to save frame {i}")
        
        logger.info(f"Saved {len(saved_images)} training images for {person_name}")
        
        return jsonify({
            'message': f'Successfully processed video and extracted {len(saved_images)} training images for {person_name}',
            'frames_captured': frame_count,
            'images_saved': len(saved_images),
            'saved_files': saved_images
        })
        
    except Exception as e:
        logger.error(f"Error during video processing: {str(e)}")
        # Clean up temp file if it exists
        if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        return jsonify({'error': f'Video processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)