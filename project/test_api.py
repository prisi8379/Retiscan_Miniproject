import requests
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

def test_api():
    """Test the FastAPI endpoints"""
    base_url = "http://localhost:8000"
    
    print("Testing Digit Recognition API...")
    print("=" * 50)
    
    # Test health check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health Status: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    # Test model info
    print("\n2. Testing model info...")
    try:
        response = requests.get(f"{base_url}/model/info")
        model_info = response.json()
        print(f"Model Info: {model_info}")
    except Exception as e:
        print(f"Model info failed: {e}")
    
    # Create a test image (digit 5)
    print("\n3. Creating test image...")
    test_image = create_test_digit()
    
    # Test prediction
    print("\n4. Testing prediction...")
    try:
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        test_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Send prediction request
        files = {'file': ('test_digit.png', img_byte_arr, 'image/png')}
        response = requests.post(f"{base_url}/predict", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction Result: {result}")
            print(f"Predicted Digit: {result['predicted_digit']}")
            print(f"Confidence: {result['confidence']:.4f}")
        else:
            print(f"Prediction failed: {response.text}")
            
    except Exception as e:
        print(f"Prediction test failed: {e}")

def create_test_digit():
    """Create a simple test image of digit 5"""
    # Create a 28x28 black image
    img_array = np.zeros((100, 100), dtype=np.uint8)
    
    # Draw a simple "5"
    img_array[20:30, 20:70] = 255  # Top horizontal line
    img_array[20:50, 20:30] = 255  # Left vertical line (top)
    img_array[45:55, 20:60] = 255  # Middle horizontal line
    img_array[50:80, 60:70] = 255  # Right vertical line (bottom)
    img_array[70:80, 20:70] = 255  # Bottom horizontal line
    
    # Convert to PIL Image
    test_image = Image.fromarray(img_array, mode='L')
    
    return test_image

if __name__ == "__main__":
    test_api()