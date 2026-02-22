import requests
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import cv2
import os

def test_retiscan_api():
    """Test the RetiScan API endpoints"""
    base_url = "http://localhost:8000"
    
    print("Testing RetiScan - Diabetic Retinopathy Detection API")
    print("=" * 60)
    
    # Test health check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        health_data = response.json()
        print(f"✅ Health Status: {health_data['status']}")
        print(f"   Model Loaded: {health_data['model_loaded']}")
        print(f"   Version: {health_data['version']}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Test model info
    print("\n2. Testing model info...")
    try:
        response = requests.get(f"{base_url}/model/info")
        if response.status_code == 200:
            model_info = response.json()
            print(f"✅ Model: {model_info.get('model_name', 'Unknown')}")
            print(f"   Type: {model_info.get('model_type', 'Unknown')}")
            print(f"   Classes: {len(model_info.get('class_names', []))}")
            print(f"   Parameters: {model_info.get('total_parameters', 0):,}")
        else:
            print(f"⚠️  Model not loaded: {response.json()}")
    except Exception as e:
        print(f"❌ Model info failed: {e}")
    
    # Test statistics
    print("\n3. Testing statistics...")
    try:
        response = requests.get(f"{base_url}/statistics")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Total Analyses: {stats['total_analyses']}")
            print(f"   Accuracy: {stats['accuracy']}%")
            print(f"   Sensitivity: {stats['sensitivity']}%")
            print(f"   Specificity: {stats['specificity']}%")
        else:
            print(f"⚠️  Statistics unavailable")
    except Exception as e:
        print(f"❌ Statistics failed: {e}")
    
    # Create test retinal images
    print("\n4. Creating test retinal images...")
    test_images = create_test_retinal_images()
    
    # Test analysis for each severity level
    print("\n5. Testing retinal image analysis...")
    for severity, image in test_images.items():
        print(f"\n   Testing {severity}...")
        try:
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Send analysis request
            files = {'file': (f'test_{severity}.png', img_byte_arr, 'image/png')}
            response = requests.post(f"{base_url}/analyze", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Predicted: {result['predicted_class']}")
                print(f"      Confidence: {result['confidence']:.3f}")
                print(f"      Risk Level: {result['risk_level']}")
                print(f"      Recommendation: {result['recommendation'][:50]}...")
            else:
                print(f"   ❌ Analysis failed: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Analysis test failed: {e}")

def create_test_retinal_images():
    """Create synthetic retinal images for testing different DR severities"""
    test_images = {}
    
    def create_retinal_base():
        """Create base retinal image"""
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Create circular retinal background
        center = (256, 256)
        radius = 240
        cv2.circle(img, center, radius, (139, 69, 19), -1)  # Brown background
        
        # Add optic disc
        optic_center = (200, 256)
        cv2.circle(img, optic_center, 35, (255, 220, 177), -1)
        
        # Add blood vessels
        for i in range(8):
            angle = i * 45
            x1 = int(optic_center[0] + 25 * np.cos(np.radians(angle)))
            y1 = int(optic_center[1] + 25 * np.sin(np.radians(angle)))
            x2 = int(center[0] + 180 * np.cos(np.radians(angle)))
            y2 = int(center[1] + 180 * np.sin(np.radians(angle)))
            cv2.line(img, (x1, y1), (x2, y2), (100, 0, 0), 2)
        
        return img
    
    # No DR - healthy retina
    healthy_img = create_retinal_base()
    test_images['no_dr'] = Image.fromarray(healthy_img)
    
    # Mild DR - few microaneurysms
    mild_img = create_retinal_base()
    for _ in range(3):
        x = np.random.randint(100, 400)
        y = np.random.randint(100, 400)
        cv2.circle(mild_img, (x, y), 1, (0, 0, 100), -1)
    test_images['mild_dr'] = Image.fromarray(mild_img)
    
    # Moderate DR - more microaneurysms and small hemorrhages
    moderate_img = create_retinal_base()
    for _ in range(8):
        x = np.random.randint(100, 400)
        y = np.random.randint(100, 400)
        cv2.circle(moderate_img, (x, y), np.random.randint(1, 3), (0, 0, 120), -1)
    for _ in range(3):
        x = np.random.randint(100, 400)
        y = np.random.randint(100, 400)
        cv2.circle(moderate_img, (x, y), np.random.randint(3, 5), (0, 0, 80), -1)
    test_images['moderate_dr'] = Image.fromarray(moderate_img)
    
    # Severe DR - extensive pathology
    severe_img = create_retinal_base()
    for _ in range(15):
        x = np.random.randint(100, 400)
        y = np.random.randint(100, 400)
        cv2.circle(severe_img, (x, y), np.random.randint(1, 4), (0, 0, 120), -1)
    for _ in range(8):
        x = np.random.randint(100, 400)
        y = np.random.randint(100, 400)
        cv2.circle(severe_img, (x, y), np.random.randint(3, 6), (0, 0, 80), -1)
    # Add hard exudates
    for _ in range(5):
        x = np.random.randint(100, 400)
        y = np.random.randint(100, 400)
        cv2.circle(severe_img, (x, y), np.random.randint(4, 7), (0, 200, 200), -1)
    test_images['severe_dr'] = Image.fromarray(severe_img)
    
    # Proliferative DR - neovascularization
    prolif_img = create_retinal_base()
    for _ in range(20):
        x = np.random.randint(100, 400)
        y = np.random.randint(100, 400)
        cv2.circle(prolif_img, (x, y), np.random.randint(1, 4), (0, 0, 120), -1)
    # Add neovascularization
    for _ in range(8):
        x1 = np.random.randint(100, 400)
        y1 = np.random.randint(100, 400)
        x2 = x1 + np.random.randint(-40, 40)
        y2 = y1 + np.random.randint(-40, 40)
        cv2.line(prolif_img, (x1, y1), (x2, y2), (0, 80, 0), 2)
    test_images['proliferative_dr'] = Image.fromarray(prolif_img)
    
    return test_images

def save_test_images():
    """Save test images to disk for manual inspection"""
    print("Creating and saving test retinal images...")
    os.makedirs('test_images', exist_ok=True)
    
    test_images = create_test_retinal_images()
    
    for severity, image in test_images.items():
        filename = f'test_images/test_{severity}.png'
        image.save(filename)
        print(f"Saved: {filename}")
    
    print("Test images saved successfully!")

if __name__ == "__main__":
    # Save test images first
    save_test_images()
    
    # Test the API
    test_retiscan_api()
    
    print("\n" + "="*60)
    print("RetiScan API testing completed!")
    print("Check test_images/ folder for sample retinal images")