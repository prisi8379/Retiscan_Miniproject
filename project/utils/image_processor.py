import cv2
import numpy as np
from PIL import Image
import io

class ImageProcessor:
    """Utility class for image processing operations"""
    
    @staticmethod
    def preprocess_for_prediction(image_array):
        """
        Preprocess image for digit recognition
        
        Args:
            image_array: numpy array of the image
            
        Returns:
            Preprocessed image ready for model prediction
        """
        # Ensure grayscale
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Resize to 28x28
        resized = cv2.resize(image_array, (28, 28))
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(resized, (3, 3), 0)
        
        # Threshold to create binary image
        _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if necessary (MNIST expects white digits on black background)
        if np.mean(threshold) > 127:
            threshold = 255 - threshold
        
        # Normalize
        normalized = threshold.astype('float32') / 255.0
        
        # Reshape for model
        final_image = normalized.reshape(1, 28, 28, 1)
        
        return final_image
    
    @staticmethod
    def enhance_image(image_array):
        """
        Apply image enhancement techniques
        
        Args:
            image_array: numpy array of the image
            
        Returns:
            Enhanced image
        """
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image_array)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    @staticmethod
    def find_digit_contour(image_array):
        """
        Find and extract the main digit from the image
        
        Args:
            image_array: numpy array of the image
            
        Returns:
            Cropped image containing the main digit
        """
        # Find contours
        contours, _ = cv2.findContours(image_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image_array
        
        # Find the largest contour (presumably the digit)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding
        padding = 4
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image_array.shape[1] - x, w + 2 * padding)
        h = min(image_array.shape[0] - y, h + 2 * padding)
        
        # Crop the digit
        cropped = image_array[y:y+h, x:x+w]
        
        return cropped