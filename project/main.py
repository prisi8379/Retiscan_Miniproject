from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import os
import json
from datetime import datetime
import base64
import matplotlib.pyplot as plt
import seaborn as sns

app = FastAPI(
    title="RetiScan - Diabetic Retinopathy Detection API",
    description="Advanced AI system for early detection of diabetic retinopathy using deep learning",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
severity_descriptions = {
    'No DR': 'No signs of diabetic retinopathy detected. Continue regular monitoring.',
    'Mild DR': 'Mild diabetic retinopathy detected. Regular monitoring recommended.',
    'Moderate DR': 'Moderate diabetic retinopathy detected. Consult ophthalmologist soon.',
    'Severe DR': 'Severe diabetic retinopathy detected. Immediate medical attention required.',
    'Proliferative DR': 'Proliferative diabetic retinopathy detected. Urgent medical intervention needed.'
}

risk_levels = {
    'No DR': 'Low',
    'Mild DR': 'Low-Medium',
    'Moderate DR': 'Medium',
    'Severe DR': 'High',
    'Proliferative DR': 'Critical'
}

def load_model():
    """Load the trained retinal model"""
    global model
    model_path = 'models/retina_model.h5'
    
    if not os.path.exists(model_path):
        # Try alternative path
        alt_path = 'models/best_retina_model.h5'
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Model file not found. Please train the model first using train_retina_model.py")
    
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"RetiScan model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def preprocess_retinal_image(image_bytes):
    """Advanced preprocessing for retinal images"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert PIL to numpy array
        image_array = np.array(image)
        
        # Apply advanced preprocessing
        processed_image = enhance_retinal_image(image_array)
        
        # Resize to model input size
        processed_image = cv2.resize(processed_image, (300, 300))
        
        # Normalize to [0, 1]
        processed_image = processed_image.astype('float32') / 255.0
        
        # Reshape for model input
        processed_image = processed_image.reshape(1, 300, 300, 3)
        
        return processed_image
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def enhance_retinal_image(image):
    """Apply advanced enhancement techniques for retinal images"""
    # Convert to LAB color space for better contrast enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Apply Gaussian blur to reduce noise
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Enhance green channel (most informative for retinal analysis)
    enhanced[:, :, 1] = cv2.equalizeHist(enhanced[:, :, 1])
    
    return enhanced

def generate_heatmap(image, predictions):
    """Generate attention heatmap for model predictions"""
    try:
        # This is a simplified heatmap generation
        # In a real implementation, you would use techniques like Grad-CAM
        heatmap = np.random.rand(300, 300)  # Placeholder
        heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Overlay on original image
        overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
        
        return overlay
    except:
        return image

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    if not load_model():
        print("Warning: RetiScan model not loaded. Please train the model first.")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the RetiScan main page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RetiScan - Diabetic Retinopathy Detection</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
                font-weight: 300;
            }
            
            .header p {
                font-size: 1.2em;
                opacity: 0.9;
            }
            
            .content {
                padding: 40px;
            }
            
            .upload-section {
                background: #f8f9fa;
                border: 3px dashed #dee2e6;
                border-radius: 15px;
                padding: 40px;
                text-align: center;
                margin: 30px 0;
                transition: all 0.3s ease;
            }
            
            .upload-section:hover {
                border-color: #3498db;
                background: #e3f2fd;
            }
            
            .upload-section.dragover {
                border-color: #2196f3;
                background: #bbdefb;
            }
            
            .file-input {
                display: none;
            }
            
            .upload-btn {
                background: linear-gradient(135deg, #3498db, #2980b9);
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 50px;
                font-size: 16px;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
            }
            
            .upload-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4);
            }
            
            .analyze-btn {
                background: linear-gradient(135deg, #e74c3c, #c0392b);
                color: white;
                padding: 15px 40px;
                border: none;
                border-radius: 50px;
                font-size: 18px;
                cursor: pointer;
                margin-top: 20px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
            }
            
            .analyze-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(231, 76, 60, 0.4);
            }
            
            .analyze-btn:disabled {
                background: #bdc3c7;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }
            
            .preview-section {
                display: none;
                margin: 30px 0;
                text-align: center;
            }
            
            .preview-image {
                max-width: 400px;
                max-height: 400px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                margin: 20px;
            }
            
            .result-section {
                display: none;
                margin-top: 30px;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            
            .result-success {
                background: linear-gradient(135deg, #2ecc71, #27ae60);
                color: white;
            }
            
            .result-warning {
                background: linear-gradient(135deg, #f39c12, #e67e22);
                color: white;
            }
            
            .result-danger {
                background: linear-gradient(135deg, #e74c3c, #c0392b);
                color: white;
            }
            
            .result-critical {
                background: linear-gradient(135deg, #8e44ad, #9b59b6);
                color: white;
            }
            
            .prediction-details {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            
            .detail-card {
                background: rgba(255,255,255,0.1);
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }
            
            .confidence-bar {
                background: rgba(255,255,255,0.2);
                height: 10px;
                border-radius: 5px;
                margin: 10px 0;
                overflow: hidden;
            }
            
            .confidence-fill {
                height: 100%;
                background: rgba(255,255,255,0.8);
                border-radius: 5px;
                transition: width 0.5s ease;
            }
            
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }
            
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .api-info {
                background: #f8f9fa;
                padding: 30px;
                border-radius: 15px;
                margin-top: 40px;
            }
            
            .api-info h3 {
                color: #2c3e50;
                margin-bottom: 20px;
            }
            
            .endpoint {
                background: white;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
                border-left: 4px solid #3498db;
            }
            
            .medical-disclaimer {
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                color: #856404;
                padding: 20px;
                border-radius: 10px;
                margin-top: 30px;
            }
            
            .medical-disclaimer strong {
                color: #d63031;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔬 RetiScan</h1>
                <p>Advanced AI-Powered Diabetic Retinopathy Detection System</p>
            </div>
            
            <div class="content">
                <div class="medical-disclaimer">
                    <strong>⚠️ Medical Disclaimer:</strong> This AI system is for educational and research purposes only. 
                    It should not be used as a substitute for professional medical diagnosis. Always consult with 
                    qualified healthcare professionals for medical advice and treatment decisions.
                </div>
                
                <div class="upload-section" id="uploadSection">
                    <h3>📸 Upload Retinal Image</h3>
                    <p>Select a high-quality retinal fundus image for analysis</p>
                    <input type="file" id="imageFile" class="file-input" accept="image/*">
                    <button class="upload-btn" onclick="document.getElementById('imageFile').click()">
                        Choose Image
                    </button>
                    <p style="margin-top: 15px; color: #666; font-size: 14px;">
                        Supported formats: JPG, PNG, TIFF | Max size: 10MB
                    </p>
                </div>
                
                <div class="preview-section" id="previewSection">
                    <h3>Image Preview</h3>
                    <img id="previewImage" class="preview-image" alt="Preview">
                    <br>
                    <button class="analyze-btn" id="analyzeBtn" onclick="analyzeImage()">
                        🔍 Analyze for Diabetic Retinopathy
                    </button>
                </div>
                
                <div class="loading" id="loadingSection">
                    <div class="spinner"></div>
                    <h3>Analyzing retinal image...</h3>
                    <p>Our AI is examining the image for signs of diabetic retinopathy</p>
                </div>
                
                <div class="result-section" id="resultSection">
                    <h2 id="resultTitle">Analysis Results</h2>
                    <div class="prediction-details" id="predictionDetails">
                        <!-- Results will be populated here -->
                    </div>
                </div>
                
                <div class="api-info">
                    <h3>🔌 API Endpoints</h3>
                    <div class="endpoint">
                        <strong>POST /analyze</strong> - Upload retinal image for DR analysis
                    </div>
                    <div class="endpoint">
                        <strong>GET /model/info</strong> - Get model information and statistics
                    </div>
                    <div class="endpoint">
                        <strong>POST /train</strong> - Train a new model (requires dataset)
                    </div>
                    <div class="endpoint">
                        <strong>GET /health</strong> - System health check
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let selectedFile = null;
            
            // File input change handler
            document.getElementById('imageFile').addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    selectedFile = file;
                    showPreview(file);
                }
            });
            
            // Drag and drop functionality
            const uploadSection = document.getElementById('uploadSection');
            
            uploadSection.addEventListener('dragover', function(e) {
                e.preventDefault();
                uploadSection.classList.add('dragover');
            });
            
            uploadSection.addEventListener('dragleave', function(e) {
                e.preventDefault();
                uploadSection.classList.remove('dragover');
            });
            
            uploadSection.addEventListener('drop', function(e) {
                e.preventDefault();
                uploadSection.classList.remove('dragover');
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    const file = files[0];
                    if (file.type.startsWith('image/')) {
                        selectedFile = file;
                        document.getElementById('imageFile').files = files;
                        showPreview(file);
                    }
                }
            });
            
            function showPreview(file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('previewImage').src = e.target.result;
                    document.getElementById('previewSection').style.display = 'block';
                    document.getElementById('analyzeBtn').disabled = false;
                };
                reader.readAsDataURL(file);
            }
            
            async function analyzeImage() {
                if (!selectedFile) {
                    alert('Please select an image first.');
                    return;
                }
                
                // Show loading
                document.getElementById('loadingSection').style.display = 'block';
                document.getElementById('resultSection').style.display = 'none';
                document.getElementById('analyzeBtn').disabled = true;
                
                try {
                    const formData = new FormData();
                    formData.append('file', selectedFile);
                    
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        showResults(result);
                    } else {
                        throw new Error(result.detail || 'Analysis failed');
                    }
                    
                } catch (error) {
                    alert('Error: ' + error.message);
                } finally {
                    document.getElementById('loadingSection').style.display = 'none';
                    document.getElementById('analyzeBtn').disabled = false;
                }
            }
            
            function showResults(result) {
                const resultSection = document.getElementById('resultSection');
                const resultTitle = document.getElementById('resultTitle');
                const predictionDetails = document.getElementById('predictionDetails');
                
                // Determine result class based on severity
                const severity = result.predicted_class;
                let resultClass = 'result-success';
                
                if (severity.includes('Mild')) resultClass = 'result-warning';
                else if (severity.includes('Moderate')) resultClass = 'result-warning';
                else if (severity.includes('Severe')) resultClass = 'result-danger';
                else if (severity.includes('Proliferative')) resultClass = 'result-critical';
                
                resultSection.className = 'result-section ' + resultClass;
                resultTitle.textContent = `Analysis Complete: ${severity}`;
                
                // Create detailed results
                const confidence = (result.confidence * 100).toFixed(1);
                const riskLevel = result.risk_level;
                const recommendation = result.recommendation;
                
                predictionDetails.innerHTML = `
                    <div class="detail-card">
                        <h4>🎯 Prediction</h4>
                        <p><strong>${severity}</strong></p>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidence}%"></div>
                        </div>
                        <p>Confidence: ${confidence}%</p>
                    </div>
                    
                    <div class="detail-card">
                        <h4>⚠️ Risk Level</h4>
                        <p><strong>${riskLevel}</strong></p>
                        <p style="margin-top: 10px; font-size: 14px;">
                            ${getRiskDescription(riskLevel)}
                        </p>
                    </div>
                    
                    <div class="detail-card">
                        <h4>💡 Recommendation</h4>
                        <p>${recommendation}</p>
                    </div>
                    
                    <div class="detail-card">
                        <h4>📊 All Probabilities</h4>
                        ${Object.entries(result.all_probabilities).map(([cls, prob]) => 
                            `<div style="margin: 5px 0;">
                                <span>${cls}: ${(prob * 100).toFixed(1)}%</span>
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width: ${prob * 100}%"></div>
                                </div>
                            </div>`
                        ).join('')}
                    </div>
                `;
                
                resultSection.style.display = 'block';
            }
            
            function getRiskDescription(riskLevel) {
                const descriptions = {
                    'Low': 'Regular monitoring recommended',
                    'Low-Medium': 'Increased monitoring advised',
                    'Medium': 'Consult ophthalmologist soon',
                    'High': 'Immediate medical attention required',
                    'Critical': 'Urgent medical intervention needed'
                };
                return descriptions[riskLevel] || 'Consult healthcare provider';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/analyze")
async def analyze_retinal_image(file: UploadFile = File(...)):
    """Analyze retinal image for diabetic retinopathy"""
    if model is None:
        raise HTTPException(status_code=503, detail="RetiScan model not loaded. Please train the model first.")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess image
        processed_image = preprocess_retinal_image(image_bytes)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        predicted_class = class_names[predicted_class_idx]
        risk_level = risk_levels[predicted_class]
        recommendation = severity_descriptions[predicted_class]
        
        # Get all probabilities
        all_probabilities = {
            class_names[i]: float(predictions[0][i]) for i in range(len(class_names))
        }
        
        return JSONResponse({
            "predicted_class": predicted_class,
            "confidence": confidence,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "all_probabilities": all_probabilities,
            "timestamp": datetime.now().isoformat(),
            "model_version": "RetiScan v2.0"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get information about the RetiScan model"""
    if model is None:
        return JSONResponse({"error": "No model loaded"})
    
    try:
        model_config = {
            "model_name": "RetiScan - Diabetic Retinopathy Detection",
            "model_type": "EfficientNetB3-based CNN",
            "input_shape": list(model.input_shape[1:]),
            "output_classes": len(class_names),
            "class_names": class_names,
            "total_parameters": model.count_params(),
            "trainable_parameters": sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
            "layers": len(model.layers),
            "version": "2.0.0"
        }
        
        # Check if model file exists and get its info
        for model_path in ['models/retina_model.h5', 'models/best_retina_model.h5']:
            if os.path.exists(model_path):
                model_config["model_file_size"] = os.path.getsize(model_path)
                model_config["model_path"] = model_path
                break
        
        return JSONResponse(model_config)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.post("/train")
async def train_new_model():
    """Train a new RetiScan model"""
    try:
        import subprocess
        import sys
        
        # Run training script
        result = subprocess.run([sys.executable, "train_retina_model.py"], 
                              capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            # Reload the model
            load_model()
            return JSONResponse({
                "message": "RetiScan model training completed successfully",
                "output": result.stdout[-1000:],  # Last 1000 characters
                "timestamp": datetime.now().isoformat()
            })
        else:
            raise HTTPException(status_code=500, detail=f"Training failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Training timeout. Process took too long.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "service": "RetiScan - Diabetic Retinopathy Detection",
        "model_loaded": model is not None,
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    })

@app.get("/statistics")
async def get_statistics():
    """Get model performance statistics"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # This would typically come from a database of predictions
    # For demo purposes, we'll return mock statistics
    stats = {
        "total_analyses": 1247,
        "accuracy": 94.2,
        "sensitivity": 92.8,
        "specificity": 95.6,
        "class_distribution": {
            "No DR": 45.2,
            "Mild DR": 28.1,
            "Moderate DR": 15.3,
            "Severe DR": 8.7,
            "Proliferative DR": 2.7
        },
        "last_updated": datetime.now().isoformat()
    }
    
    return JSONResponse(stats)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)