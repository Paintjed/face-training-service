const API_BASE = '';

// Global notification system
function showNotification(message, type = 'info', duration = 5000) {
    const notification = document.getElementById('globalNotification');
    
    // Clear existing classes and content
    notification.className = 'notification';
    notification.innerHTML = '';
    
    // Add type class
    notification.classList.add(type);
    
    // Add message and close button
    notification.innerHTML = `
        ${message}
        <button class="close-btn" onclick="hideNotification()">&times;</button>
    `;
    
    // Show notification
    notification.classList.remove('hidden');
    
    // Auto-hide after duration
    if (duration > 0) {
        setTimeout(() => {
            hideNotification();
        }, duration);
    }
}

function hideNotification() {
    const notification = document.getElementById('globalNotification');
    notification.classList.add('hidden');
}

// Enhanced error handling
function handleApiError(error, response = null) {
    let errorMessage = 'An unexpected error occurred';
    
    if (response) {
        if (response.status === 400) {
            errorMessage = 'Invalid request. Please check your input.';
        } else if (response.status === 404) {
            errorMessage = 'Resource not found.';
        } else if (response.status === 500) {
            errorMessage = 'Server error. Please try again later.';
        } else if (response.status >= 400) {
            errorMessage = `Request failed with status ${response.status}`;
        }
    }
    
    if (error.message) {
        errorMessage += `: ${error.message}`;
    }
    
    showNotification(errorMessage, 'error');
    return errorMessage;
}

// Form validation helpers
function validateForm(formElement) {
    const inputs = formElement.querySelectorAll('input[required]');
    let isValid = true;
    
    inputs.forEach(input => {
        const formGroup = input.closest('.form-group');
        const existingError = formGroup.querySelector('.field-error');
        
        // Remove existing error
        if (existingError) {
            existingError.remove();
        }
        formGroup.classList.remove('error');
        
        // Validate input
        if (!input.value.trim()) {
            isValid = false;
            formGroup.classList.add('error');
            
            const errorSpan = document.createElement('span');
            errorSpan.classList.add('field-error');
            errorSpan.textContent = 'This field is required';
            formGroup.appendChild(errorSpan);
        } else if (input.type === 'file' && input.files.length === 0) {
            isValid = false;
            formGroup.classList.add('error');
            
            const errorSpan = document.createElement('span');
            errorSpan.classList.add('field-error');
            errorSpan.textContent = 'Please select at least one file';
            formGroup.appendChild(errorSpan);
        }
    });
    
    return isValid;
}

// Loading state helpers
function setLoading(element, isLoading) {
    if (isLoading) {
        element.classList.add('loading');
        element.disabled = true;
    } else {
        element.classList.remove('loading');
        element.disabled = false;
    }
}

// Upload form handler
document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const form = e.target;
    const submitButton = form.querySelector('button[type="submit"]');
    
    // Validate form
    if (!validateForm(form)) {
        showNotification('Please fix the errors above', 'error');
        return;
    }
    
    const personName = document.getElementById('personName').value.trim();
    const images = document.getElementById('images').files;
    
    // Additional validations
    if (images.length === 0) {
        showNotification('Please select at least one image', 'error');
        return;
    }
    
    // Check file types
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    const invalidFiles = Array.from(images).filter(file => !allowedTypes.includes(file.type));
    if (invalidFiles.length > 0) {
        showNotification(`Invalid file types: ${invalidFiles.map(f => f.name).join(', ')}. Only JPG, JPEG, and PNG files are allowed.`, 'error');
        return;
    }
    
    // Check file sizes (max 10MB per file)
    const maxSize = 10 * 1024 * 1024; // 10MB
    const largeFiles = Array.from(images).filter(file => file.size > maxSize);
    if (largeFiles.length > 0) {
        showNotification(`Files too large: ${largeFiles.map(f => f.name).join(', ')}. Maximum size is 10MB per file.`, 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('person_name', personName);
    for (let i = 0; i < images.length; i++) {
        formData.append('images', images[i]);
    }
    
    const statusDiv = document.getElementById('uploadStatus');
    setLoading(submitButton, true);
    statusDiv.innerHTML = '<div class="info">Uploading images...</div>';
    
    try {
        const response = await fetch(`${API_BASE}/upload_training_data`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        if (response.ok) {
            showNotification(`Successfully uploaded ${images.length} images for ${personName}`, 'success');
            statusDiv.innerHTML = `<div class="success">${result.message}</div>`;
            document.getElementById('uploadForm').reset();
        } else {
            const errorMsg = result.error || 'Upload failed';
            showNotification(`Upload failed: ${errorMsg}`, 'error');
            statusDiv.innerHTML = `<div class="error">Error: ${errorMsg}</div>`;
        }
    } catch (error) {
        handleApiError(error);
        statusDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    } finally {
        setLoading(submitButton, false);
    }
});

// Video recording variables
let mediaStream = null;
let mediaRecorder = null;
let recordedChunks = [];

// Start camera
document.getElementById('startVideoBtn').addEventListener('click', async () => {
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 }, 
            audio: false 
        });
        
        const videoPreview = document.getElementById('videoPreview');
        videoPreview.srcObject = mediaStream;
        videoPreview.style.display = 'block';
        
        document.getElementById('startVideoBtn').style.display = 'none';
        document.getElementById('recordVideoBtn').style.display = 'inline-block';
        document.getElementById('stopVideoBtn').style.display = 'inline-block';
        
        showNotification('Camera started successfully!', 'success');
        document.getElementById('videoStatus').innerHTML = '<div class="info">Camera ready. You can now record video.</div>';
    } catch (error) {
        showNotification('Failed to access camera. Please grant camera permission.', 'error');
        document.getElementById('videoStatus').innerHTML = '<div class="error">Camera access denied or not available</div>';
    }
});

// Record video
document.getElementById('recordVideoBtn').addEventListener('click', async () => {
    const personName = document.getElementById('videoPersonName').value.trim();
    if (!personName) {
        showNotification('Please enter person name first', 'error');
        return;
    }
    
    if (!mediaStream) {
        showNotification('Please start camera first', 'error');
        return;
    }
    
    try {
        recordedChunks = [];
        mediaRecorder = new MediaRecorder(mediaStream, { mimeType: 'video/webm' });
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = async () => {
            const videoBlob = new Blob(recordedChunks, { type: 'video/webm' });
            await uploadVideoForTraining(videoBlob, personName);
        };
        
        mediaRecorder.start();
        
        document.getElementById('recordVideoBtn').disabled = true;
        document.getElementById('videoStatus').innerHTML = '<div class="info">Recording... (5 seconds)</div>';
        
        // Stop recording after 5 seconds
        setTimeout(() => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                document.getElementById('recordVideoBtn').disabled = false;
            }
        }, 5000);
        
    } catch (error) {
        showNotification('Failed to start recording', 'error');
        document.getElementById('videoStatus').innerHTML = '<div class="error">Recording failed</div>';
    }
});

// Stop camera
document.getElementById('stopVideoBtn').addEventListener('click', () => {
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    
    const videoPreview = document.getElementById('videoPreview');
    videoPreview.style.display = 'none';
    
    document.getElementById('startVideoBtn').style.display = 'inline-block';
    document.getElementById('recordVideoBtn').style.display = 'none';
    document.getElementById('stopVideoBtn').style.display = 'none';
    
    document.getElementById('videoStatus').innerHTML = '';
});

// Upload recorded video for training
async function uploadVideoForTraining(videoBlob, personName) {
    const formData = new FormData();
    formData.append('person_name', personName);
    formData.append('video', videoBlob, 'recording.webm');
    
    const statusDiv = document.getElementById('videoStatus');
    statusDiv.innerHTML = '<div class="info">Processing video and extracting frames...</div>';
    
    try {
        const response = await fetch(`${API_BASE}/process_video_training`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        if (response.ok) {
            showNotification(`Successfully processed video for ${personName}! Extracted ${result.images_saved} training images.`, 'success');
            statusDiv.innerHTML = `<div class="success">${result.message}</div>`;
            document.getElementById('videoPersonName').value = '';
        } else {
            const errorMsg = result.error || 'Video processing failed';
            showNotification(`Video processing failed: ${errorMsg}`, 'error');
            statusDiv.innerHTML = `<div class="error">Error: ${errorMsg}</div>`;
        }
    } catch (error) {
        handleApiError(error);
        statusDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    }
}

// Start training
document.getElementById('startTraining').addEventListener('click', async () => {
    const button = document.getElementById('startTraining');
    const statusDiv = document.getElementById('trainingStatus');
    
    setLoading(button, true);
    statusDiv.innerHTML = '<div class="info">Starting training...</div>';
    
    try {
        const response = await fetch(`${API_BASE}/start_training`, {
            method: 'POST'
        });
        
        const result = await response.json();
        if (response.ok) {
            showNotification('Training started successfully!', 'success');
            statusDiv.innerHTML = `<div class="success">${result.message}</div>`;
        } else {
            const errorMsg = result.error || 'Failed to start training';
            showNotification(`Training failed to start: ${errorMsg}`, 'error');
            statusDiv.innerHTML = `<div class="error">Error: ${errorMsg}</div>`;
        }
    } catch (error) {
        handleApiError(error);
        statusDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    } finally {
        setLoading(button, false);
    }
});

// Check training status
document.getElementById('checkStatus').addEventListener('click', async () => {
    const statusDiv = document.getElementById('trainingStatus');
    
    try {
        const response = await fetch(`${API_BASE}/training_status`);
        const result = await response.json();
        
        if (response.ok) {
            let statusHTML = `<div class="info">Status: ${result.status}</div>`;
            if (result.progress) {
                statusHTML += `<div class="info">Progress: ${result.progress}</div>`;
            }
            if (result.message) {
                statusHTML += `<div class="info">Message: ${result.message}</div>`;
            }
            statusDiv.innerHTML = statusHTML;
        } else {
            statusDiv.innerHTML = `<div class="error">Error: ${result.error}</div>`;
        }
    } catch (error) {
        statusDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    }
});

// Download model
document.getElementById('downloadModel').addEventListener('click', async () => {
    const button = document.getElementById('downloadModel');
    
    setLoading(button, true);
    
    try {
        const response = await fetch(`${API_BASE}/download_coreml_model`);
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'FaceClassifier.zip';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();
            showNotification('Model downloaded successfully!', 'success');
        } else {
            const result = await response.json();
            const errorMsg = result.error || 'Download failed';
            showNotification(`Download failed: ${errorMsg}`, 'error');
            document.getElementById('modelInfo').innerHTML = `<div class="error">Error: ${errorMsg}</div>`;
        }
    } catch (error) {
        handleApiError(error);
        document.getElementById('modelInfo').innerHTML = `<div class="error">Error: ${error.message}</div>`;
    } finally {
        setLoading(button, false);
    }
});

// Get model info
document.getElementById('getModelInfo').addEventListener('click', async () => {
    const infoDiv = document.getElementById('modelInfo');
    
    try {
        const response = await fetch(`${API_BASE}/model_info`);
        const result = await response.json();
        
        if (response.ok) {
            let infoHTML = '<div class="info"><h3>Model Information:</h3>';
            Object.entries(result).forEach(([key, value]) => {
                infoHTML += `<p><strong>${key}:</strong> ${JSON.stringify(value)}</p>`;
            });
            infoHTML += '</div>';
            infoDiv.innerHTML = infoHTML;
        } else {
            infoDiv.innerHTML = `<div class="error">Error: ${result.error}</div>`;
        }
    } catch (error) {
        infoDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    }
});

// List people
document.getElementById('listPeople').addEventListener('click', async () => {
    const listDiv = document.getElementById('peopleList');
    
    try {
        const response = await fetch(`${API_BASE}/list_people`);
        const result = await response.json();
        
        if (response.ok) {
            if (result.people && result.people.length > 0) {
                let listHTML = '<div class="info"><h3>People in training data:</h3><ul>';
                result.people.forEach(person => {
                    listHTML += `<li>${person.name} (${person.image_count} images)</li>`;
                });
                listHTML += '</ul></div>';
                listDiv.innerHTML = listHTML;
            } else {
                listDiv.innerHTML = '<div class="info">No people found in training data.</div>';
            }
        } else {
            listDiv.innerHTML = `<div class="error">Error: ${result.error}</div>`;
        }
    } catch (error) {
        listDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    }
});

// Delete person
document.getElementById('deletePersonBtn').addEventListener('click', async () => {
    const personName = document.getElementById('deletePerson').value;
    if (!personName) {
        alert('Please enter a person name to delete');
        return;
    }
    
    if (!confirm(`Are you sure you want to delete all data for ${personName}?`)) {
        return;
    }
    
    const listDiv = document.getElementById('peopleList');
    
    try {
        const response = await fetch(`${API_BASE}/delete_person`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ person_name: personName })
        });
        
        const result = await response.json();
        if (response.ok) {
            listDiv.innerHTML = `<div class="success">${result.message}</div>`;
            document.getElementById('deletePerson').value = '';
        } else {
            listDiv.innerHTML = `<div class="error">Error: ${result.error}</div>`;
        }
    } catch (error) {
        listDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    }
});

// Clear all data
document.getElementById('clearAllData').addEventListener('click', async () => {
    if (!confirm('Are you sure you want to clear ALL training data? This cannot be undone!')) {
        return;
    }
    
    const listDiv = document.getElementById('peopleList');
    
    try {
        const response = await fetch(`${API_BASE}/clear_all_data`, {
            method: 'DELETE'
        });
        
        const result = await response.json();
        if (response.ok) {
            listDiv.innerHTML = `<div class="success">${result.message}</div>`;
        } else {
            listDiv.innerHTML = `<div class="error">Error: ${result.error}</div>`;
        }
    } catch (error) {
        listDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    }
});