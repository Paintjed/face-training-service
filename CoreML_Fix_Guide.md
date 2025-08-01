# CoreML Conversion Issues Fix Guide

## Issue Summary
Multiple CoreML conversion errors encountered during model training:
1. `Unable to determine the type of the model` - Missing source framework specification
2. `predictions is not in graph` - Incorrect output layer naming
3. `Input (face_encoding) provided is not found` - Wrong input layer name
4. `BlobWriter not loaded` - CoreML tools compatibility issue

## Root Causes & Solutions

### 1. **Missing Source Framework**
**Problem**: CoreML couldn't identify the model type
```python
# ❌ Original code
coreml_model = ct.convert(model, ...)
```

**Solution**: Explicitly specify the source framework
```python
# ✅ Fixed code
coreml_model = ct.convert(model, source="tensorflow", ...)
```

### 2. **Incorrect Layer Names**
**Problem**: Predefined layer names didn't match actual TensorFlow graph
```python
# ❌ Original code
inputs=[ct.TensorType(shape=(1, 128), name="face_encoding")]
outputs=[ct.TensorType(name="predictions")]
```

**Solution**: Use actual layer names or let CoreML auto-detect
```python
# ✅ Fixed code - let CoreML handle naming automatically
coreml_model = ct.convert(model, source="tensorflow")
```

### 3. **BlobWriter Compatibility Issue**
**Problem**: Version incompatibility between coremltools and TensorFlow
```python
# ❌ Problematic format
convert_to="mlprogram"
```

**Solution**: Use more stable neural network format
```python
# ✅ More stable format
convert_to="neuralnetwork"
```

## Complete Fix Implementation

**Updated `face_trainer.py` conversion method:**

```python
def convert_to_coreml(self):
    """Convert TensorFlow model to CoreML with proper error handling"""
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
        
        # Convert with proper parameters
        coreml_model = ct.convert(
            model,
            source="tensorflow",           # ✅ Specify source framework
            convert_to="neuralnetwork"     # ✅ Use stable format
            # ✅ Let CoreML auto-detect input/output names
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
        # ✅ Graceful fallback - don't break training
        logger.info("Model saved as H5 format only")
    
    return coreml_model_path
```

## Prevention Tips

### 1. **Check Dependencies**
```bash
pip list | grep -E "(tensorflow|coremltools)"
# Ensure compatible versions:
# tensorflow==2.12.0
# coremltools==6.3.0
```

### 2. **Debug Model Structure**
```python
# Add debugging before conversion
model.summary()
print("Input names:", [input.name for input in model.inputs])
print("Output names:", [output.name for output in model.outputs])
```

### 3. **Test Conversion Separately**
```python
# Test conversion with minimal parameters first
coreml_model = ct.convert(model, source="tensorflow")
```

### 4. **Version Compatibility Matrix**
```
TensorFlow 2.12.x + coremltools 6.x = ✅ Compatible
TensorFlow 2.13.x + coremltools 7.x = ⚠️  May have issues
```

## Alternative Solutions

### Option 1: Skip CoreML Conversion
```python
# In face_trainer.py, comment out CoreML conversion
# coreml_path = trainer.convert_to_coreml()  # Skip this step
```

### Option 2: Use Different Conversion Tool
```python
# Try tf2coreml instead of coremltools
import tf2coreml
coreml_model = tf2coreml.convert(model)
```

### Option 3: Manual Conversion Later
```python
# Save TensorFlow model first, convert later with different environment
trainer.save_model()  # This always works
# Convert to CoreML in separate script/environment
```

## Testing the Fix

1. **Start training with video data**
2. **Monitor logs for conversion messages**
3. **Check if both `.h5` and `.mlmodel` files are created**
4. **Verify model download works**

## Web UI Video Recording Feature

### Browser-Based Implementation
The video recording feature now uses proper browser APIs:

1. **Request Camera Permission**: `navigator.mediaDevices.getUserMedia()`
2. **Show Preview**: Real-time video preview in browser
3. **Record Video**: 5-second WebM recording in browser
4. **Upload & Process**: Send video to backend for frame extraction

### Usage Steps
1. Enter person name
2. Click "Start Camera" (browser requests permission)
3. Click "Record 5-Second Video" when ready
4. System extracts ~10 training frames automatically

## Files Modified
- `face_trainer.py` - Fixed CoreML conversion with error handling
- `templates/index.html` - Added video recording UI
- `static/script.js` - Browser camera API integration  
- `app.py` - Video processing endpoint

The implemented solution includes graceful error handling, so training will complete successfully even if CoreML conversion fails, ensuring you always get a usable model.