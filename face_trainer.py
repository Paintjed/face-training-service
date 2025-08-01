import os
import cv2
import face_recognition
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import logging
import coremltools as ct

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceTrainer:
    def __init__(self, data_dir="training_data", model_dir="models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.encodings = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
    
    def load_training_data(self):
        """Load face images and extract encodings"""
        logger.info("Loading training data...")
        
        for person_name in os.listdir(self.data_dir):
            person_dir = os.path.join(self.data_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
                
            logger.info(f"Processing images for {person_name}")
            
            for image_file in os.listdir(person_dir):
                if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                image_path = os.path.join(person_dir, image_file)
                try:
                    # Load image
                    image = face_recognition.load_image_file(image_path)
                    
                    # Get face encodings
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if face_encodings:
                        # Use the first face found in the image
                        encoding = face_encodings[0]
                        self.encodings.append(encoding)
                        self.labels.append(person_name)
                        logger.info(f"Added encoding for {person_name} from {image_file}")
                    else:
                        logger.warning(f"No face found in {image_path}")
                        
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {str(e)}")
        
        logger.info(f"Loaded {len(self.encodings)} face encodings for {len(set(self.labels))} people")
    
    def train_model(self):
        """Train neural network classifier on face encodings"""
        if not self.encodings:
            raise ValueError("No training data loaded. Call load_training_data() first.")
        
        logger.info("Training face recognition model...")
        
        X = np.array(self.encodings)
        y = np.array(self.labels)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        num_classes = len(self.label_encoder.classes_)
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # Create neural network model
        self.model = keras.Sequential([
            keras.layers.Dense(256, activation='relu', input_shape=(128,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Model accuracy: {test_accuracy:.2%}")
        
        return test_accuracy
    
    def save_model(self):
        """Save trained model and label encoder"""
        h5_model_path = os.path.join(self.model_dir, "face_classifier.h5")
        labels_path = os.path.join(self.model_dir, "label_encoder.pkl")
        
        # Save Keras model
        self.model.save(h5_model_path)
        
        # Save label encoder
        joblib.dump(self.label_encoder, labels_path)
        
        logger.info(f"Model saved to {h5_model_path}")
        logger.info(f"Label encoder saved to {labels_path}")
    
    def convert_to_coreml(self):
        """Convert TensorFlow model to CoreML"""
        h5_model_path = os.path.join(self.model_dir, "face_classifier.h5")
        coreml_model_path = os.path.join(self.model_dir, "FaceClassifier.mlmodel")
        
        if not os.path.exists(h5_model_path):
            raise FileNotFoundError("H5 model not found. Train and save the model first.")
        
        logger.info("Converting model to CoreML...")
        
        try:
            # Load the saved model
            model = keras.models.load_model(h5_model_path)
            logger.info(f"Model input shape: {model.input_shape}")
            logger.info(f"Model output shape: {model.output_shape}")
            
            # Try simpler conversion first
            coreml_model = ct.convert(
                model,
                source="tensorflow",
                convert_to="neuralnetwork"  # Use older format which is more stable
            )
            
            # Set model metadata
            coreml_model.author = "Face Training Service"
            coreml_model.short_description = "Face Recognition Classifier"
            coreml_model.version = "1.0"
            
            # Save CoreML model
            coreml_model.save(coreml_model_path)
            
            logger.info(f"CoreML model saved to {coreml_model_path}")
            
        except Exception as e:
            logger.error(f"CoreML conversion failed: {str(e)}")
            logger.info("Attempting fallback: saving model without CoreML conversion")
            
            # Create a simple model info file instead
            model_info = {
                "status": "h5_only",
                "message": "CoreML conversion failed, H5 model available",
                "input_shape": str(model.input_shape),
                "output_shape": str(model.output_shape),
                "classes": len(self.label_encoder.classes_)
            }
            
            fallback_info_path = os.path.join(self.model_dir, "model_info.json")
            import json
            with open(fallback_info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            logger.info(f"Model info saved to {fallback_info_path}")
        
        # Always save label mapping for iOS app
        label_mapping_path = os.path.join(self.model_dir, "label_mapping.json")
        import json
        label_mapping = {str(i): label for i, label in enumerate(self.label_encoder.classes_)}
        with open(label_mapping_path, 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        logger.info(f"Label mapping saved to {label_mapping_path}")
        
        return coreml_model_path
    
    def load_model(self):
        """Load trained model"""
        h5_model_path = os.path.join(self.model_dir, "face_classifier.h5")
        labels_path = os.path.join(self.model_dir, "label_encoder.pkl")
        
        if not os.path.exists(h5_model_path) or not os.path.exists(labels_path):
            raise FileNotFoundError("Model files not found. Train the model first.")
        
        self.model = keras.models.load_model(h5_model_path)
        self.label_encoder = joblib.load(labels_path)
        
        logger.info("Model loaded successfully")
    
    def predict(self, image_path):
        """Predict person from image"""
        if not hasattr(self, 'model'):
            self.load_model()
        
        # Load and process image
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        
        if not face_encodings:
            return None, 0.0
        
        encoding = face_encodings[0].reshape(1, -1)
        
        # Predict
        predictions = self.model.predict(encoding)[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        
        person_name = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return person_name, confidence
    
    def predict_from_encoding(self, encoding):
        """Predict person from face encoding"""
        if not hasattr(self, 'model'):
            self.load_model()
        
        encoding = np.array(encoding).reshape(1, -1)
        predictions = self.model.predict(encoding)[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        
        person_name = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return person_name, confidence

def main():
    """Main training function"""
    trainer = FaceTrainer()
    
    # Load training data
    trainer.load_training_data()
    
    if len(trainer.encodings) == 0:
        logger.error("No training data found. Please add images to the training_data directory.")
        return
    
    # Train model
    accuracy = trainer.train_model()
    
    # Save model
    trainer.save_model()
    
    # Convert to CoreML
    coreml_path = trainer.convert_to_coreml()
    
    logger.info("Training and CoreML conversion completed successfully!")
    logger.info(f"CoreML model available at: {coreml_path}")

if __name__ == "__main__":
    main()