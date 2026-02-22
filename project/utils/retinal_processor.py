import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io

class RetinalImageProcessor:
    """Advanced retinal image processing utilities for diabetic retinopathy detection"""
    
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size
    
    def preprocess_fundus_image(self, image_array):
        """
        Comprehensive preprocessing pipeline for fundus images
        
        Args:
            image_array: numpy array of the retinal image
            
        Returns:
            Preprocessed image optimized for DR detection
        """
        # Step 1: Crop to circular region (remove black borders)
        cropped = self.crop_circular_region(image_array)
        
        # Step 2: Enhance contrast using CLAHE
        enhanced = self.apply_clahe_enhancement(cropped)
        
        # Step 3: Normalize illumination
        normalized = self.normalize_illumination(enhanced)
        
        # Step 4: Enhance blood vessels
        vessel_enhanced = self.enhance_blood_vessels(normalized)
        
        # Step 5: Resize to target size
        resized = cv2.resize(vessel_enhanced, self.target_size)
        
        # Step 6: Final normalization
        final_image = resized.astype('float32') / 255.0
        
        return final_image
    
    def crop_circular_region(self, image):
        """Extract the circular retinal region from fundus image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Find the largest circular region
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
            param1=50, param2=30, minRadius=100, maxRadius=0
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Take the largest circle
            circle = max(circles, key=lambda x: x[2])
            x, y, r = circle
            
            # Create mask
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            
            # Apply mask
            if len(image.shape) == 3:
                result = cv2.bitwise_and(image, image, mask=mask)
            else:
                result = cv2.bitwise_and(image, image, mask=mask)
            
            # Crop to bounding box
            x_start, x_end = max(0, x-r), min(image.shape[1], x+r)
            y_start, y_end = max(0, y-r), min(image.shape[0], y+r)
            
            cropped = result[y_start:y_end, x_start:x_end]
            return cropped
        
        return image
    
    def apply_clahe_enhancement(self, image):
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def normalize_illumination(self, image):
        """Normalize illumination variations across the image"""
        if len(image.shape) == 3:
            # Work with each channel separately
            normalized = np.zeros_like(image)
            for i in range(3):
                channel = image[:, :, i]
                # Apply Gaussian blur to estimate background illumination
                background = cv2.GaussianBlur(channel, (0, 0), sigmaX=30, sigmaY=30)
                # Normalize
                normalized[:, :, i] = cv2.divide(channel, background, scale=255)
        else:
            background = cv2.GaussianBlur(image, (0, 0), sigmaX=30, sigmaY=30)
            normalized = cv2.divide(image, background, scale=255)
        
        return normalized
    
    def enhance_blood_vessels(self, image):
        """Enhance blood vessel visibility using morphological operations"""
        if len(image.shape) == 3:
            # Use green channel (most informative for retinal analysis)
            green_channel = image[:, :, 1]
        else:
            green_channel = image
        
        # Apply morphological opening to enhance vessels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(green_channel, cv2.MORPH_OPEN, kernel)
        
        # Subtract to enhance vessels
        enhanced_vessels = cv2.subtract(green_channel, opened)
        
        if len(image.shape) == 3:
            # Replace green channel with enhanced version
            result = image.copy()
            result[:, :, 1] = enhanced_vessels
            return result
        else:
            return enhanced_vessels
    
    def detect_optic_disc(self, image):
        """Detect optic disc location in retinal image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Find brightest region (optic disc)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
        
        return max_loc, max_val
    
    def detect_microaneurysms(self, image):
        """Detect microaneurysms (early signs of DR)"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply morphological operations to detect small dark spots
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Subtract to highlight microaneurysms
        microaneurysms = cv2.subtract(gray, opened)
        
        # Threshold to get binary image
        _, binary = cv2.threshold(microaneurysms, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by size (microaneurysms are small)
        microaneurysm_contours = [c for c in contours if 5 < cv2.contourArea(c) < 50]
        
        return microaneurysm_contours
    
    def detect_hemorrhages(self, image):
        """Detect hemorrhages in retinal image"""
        if len(image.shape) == 3:
            # Use red channel for hemorrhage detection
            red_channel = image[:, :, 2]
        else:
            red_channel = image
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened = cv2.morphologyEx(red_channel, cv2.MORPH_OPEN, kernel)
        
        # Detect dark regions (hemorrhages)
        hemorrhages = cv2.subtract(opened, red_channel)
        
        # Threshold
        _, binary = cv2.threshold(hemorrhages, 15, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by size
        hemorrhage_contours = [c for c in contours if 20 < cv2.contourArea(c) < 500]
        
        return hemorrhage_contours
    
    def detect_exudates(self, image):
        """Detect hard exudates (bright lesions)"""
        if len(image.shape) == 3:
            # Convert to HSV for better bright spot detection
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            value_channel = hsv[:, :, 2]
        else:
            value_channel = image
        
        # Threshold for bright regions
        _, binary = cv2.threshold(value_channel, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by size and shape
        exudate_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 30 < area < 1000:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:  # Reasonably circular
                        exudate_contours.append(contour)
        
        return exudate_contours
    
    def calculate_severity_score(self, image):
        """Calculate a severity score based on detected pathologies"""
        # Detect various pathologies
        microaneurysms = self.detect_microaneurysms(image)
        hemorrhages = self.detect_hemorrhages(image)
        exudates = self.detect_exudates(image)
        
        # Calculate scores
        ma_score = min(len(microaneurysms) * 0.1, 2.0)  # Max 2.0
        hem_score = min(len(hemorrhages) * 0.2, 3.0)    # Max 3.0
        ex_score = min(len(exudates) * 0.15, 2.0)       # Max 2.0
        
        total_score = ma_score + hem_score + ex_score
        
        # Normalize to 0-1 scale
        severity_score = min(total_score / 7.0, 1.0)
        
        return {
            'total_score': severity_score,
            'microaneurysms': len(microaneurysms),
            'hemorrhages': len(hemorrhages),
            'exudates': len(exudates),
            'ma_score': ma_score,
            'hem_score': hem_score,
            'ex_score': ex_score
        }
    
    def create_pathology_overlay(self, image, show_detections=True):
        """Create an overlay showing detected pathologies"""
        overlay = image.copy()
        
        if show_detections:
            # Detect pathologies
            microaneurysms = self.detect_microaneurysms(image)
            hemorrhages = self.detect_hemorrhages(image)
            exudates = self.detect_exudates(image)
            
            # Draw microaneurysms in blue
            cv2.drawContours(overlay, microaneurysms, -1, (0, 0, 255), 2)
            
            # Draw hemorrhages in red
            cv2.drawContours(overlay, hemorrhages, -1, (255, 0, 0), 2)
            
            # Draw exudates in yellow
            cv2.drawContours(overlay, exudates, -1, (255, 255, 0), 2)
        
        return overlay