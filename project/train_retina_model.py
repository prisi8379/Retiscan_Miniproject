import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import cv2
import os
import glob
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings

print(tf.config.list_physical_devices('GPU'))
warnings.filterwarnings('ignore')

class RetinaDataGenerator:
    """Custom data generator for retinal images with augmentation"""
    
    def __init__(self, image_size=(300, 300), batch_size=8):
        self.image_size = image_size
        self.batch_size = batch_size
        self.augmentation = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.validation_transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path, is_training=True):
        """Preprocess retinal image with enhancement"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply CLAHE for contrast enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Apply augmentation
            if is_training:
                augmented = self.augmentation(image=image)
            else:
                augmented = self.validation_transform(image=image)
            
            return augmented['image']
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            # Return a black image as fallback
            return np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.float32)

def create_synthetic_dataset():
    """Create synthetic retinal dataset for demonstration"""
    print("Creating synthetic retinal dataset...")
    
    # Create directories
    os.makedirs('data/retinal_images/train/no_dr', exist_ok=True)
    os.makedirs('data/retinal_images/train/mild_dr', exist_ok=True)
    os.makedirs('data/retinal_images/train/moderate_dr', exist_ok=True)
    os.makedirs('data/retinal_images/train/severe_dr', exist_ok=True)
    os.makedirs('data/retinal_images/train/proliferative_dr', exist_ok=True)
    
    os.makedirs('data/retinal_images/val/no_dr', exist_ok=True)
    os.makedirs('data/retinal_images/val/mild_dr', exist_ok=True)
    os.makedirs('data/retinal_images/val/moderate_dr', exist_ok=True)
    os.makedirs('data/retinal_images/val/severe_dr', exist_ok=True)
    os.makedirs('data/retinal_images/val/proliferative_dr', exist_ok=True)
    
    def create_synthetic_retina(severity, image_id, split='train'):
        """Create synthetic retinal image based on DR severity"""
        # Base retinal image (circular with optic disc and vessels)
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Create circular retinal background
        center = (256, 256)
        radius = 240
        cv2.circle(img, center, radius, (139, 69, 19), -1)  # Brown background
        
        # Add optic disc
        optic_center = (200, 256)
        cv2.circle(img, optic_center, 40, (255, 220, 177), -1)
        
        # Add blood vessels
        for i in range(8):
            angle = i * 45
            x1 = int(optic_center[0] + 30 * np.cos(np.radians(angle)))
            y1 = int(optic_center[1] + 30 * np.sin(np.radians(angle)))
            x2 = int(center[0] + 200 * np.cos(np.radians(angle)))
            y2 = int(center[1] + 200 * np.sin(np.radians(angle)))
            cv2.line(img, (x1, y1), (x2, y2), (139, 0, 0), 3)
        
        # Add pathological features based on severity
        if severity == 'no_dr':
            # Healthy retina - no additional features
            pass
        elif severity == 'mild_dr':
            # Add microaneurysms (small red dots)
            for _ in range(5):
                x = np.random.randint(100, 400)
                y = np.random.randint(100, 400)
                cv2.circle(img, (x, y), 2, (0, 0, 139), -1)
        elif severity == 'moderate_dr':
            # Add more microaneurysms and hemorrhages
            for _ in range(10):
                x = np.random.randint(100, 400)
                y = np.random.randint(100, 400)
                cv2.circle(img, (x, y), np.random.randint(2, 4), (0, 0, 139), -1)
            # Add hemorrhages
            for _ in range(5):
                x = np.random.randint(100, 400)
                y = np.random.randint(100, 400)
                cv2.circle(img, (x, y), np.random.randint(3, 6), (0, 0, 100), -1)
        elif severity == 'severe_dr':
            # Add extensive hemorrhages and exudates
            for _ in range(15):
                x = np.random.randint(100, 400)
                y = np.random.randint(100, 400)
                cv2.circle(img, (x, y), np.random.randint(2, 5), (0, 0, 139), -1)
            # Add hard exudates (yellow spots)
            for _ in range(8):
                x = np.random.randint(100, 400)
                y = np.random.randint(100, 400)
                cv2.circle(img, (x, y), np.random.randint(4, 8), (0, 255, 255), -1)
        elif severity == 'proliferative_dr':
            # Add neovascularization and extensive pathology
            for _ in range(20):
                x = np.random.randint(100, 400)
                y = np.random.randint(100, 400)
                cv2.circle(img, (x, y), np.random.randint(2, 6), (0, 0, 139), -1)
            # Add neovascularization (new blood vessels)
            for _ in range(10):
                x1 = np.random.randint(100, 400)
                y1 = np.random.randint(100, 400)
                x2 = x1 + np.random.randint(-50, 50)
                y2 = y1 + np.random.randint(-50, 50)
                cv2.line(img, (x1, y1), (x2, y2), (0, 100, 0), 2)
        
        # Add noise and blur for realism
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Save image
        filename = f'data/retinal_images/{split}/{severity}/retina_{image_id:04d}.jpg'
        cv2.imwrite(filename, img)
    
    # Create training images
    severities = ['no_dr', 'mild_dr', 'moderate_dr', 'severe_dr', 'proliferative_dr']
    for severity in severities:
        for i in range(200):  # 200 images per class for training
            create_synthetic_retina(severity, i, 'train')
        for i in range(50):   # 50 images per class for validation
            create_synthetic_retina(severity, i, 'val')
    
    print("Synthetic dataset created successfully!")

def create_advanced_model(input_shape=(300, 300, 3), num_classes=5):
    """Create advanced CNN model for diabetic retinopathy detection"""
    
    # Use EfficientNetB3 as backbone
    base_model = applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom classification head
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

def load_data():
    """Load and prepare retinal image data"""
    print("Loading retinal image data...")
    
    # Check if synthetic data exists, create if not
    if not os.path.exists('data/retinal_images'):
        create_synthetic_dataset()
    
    # Data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        'data/retinal_images/train',
        target_size=(300, 300),
        batch_size=8,
        class_mode='categorical',
        shuffle=True
    )
    
    # Load validation data
    val_generator = val_datagen.flow_from_directory(
        'data/retinal_images/val',
        target_size=(512, 512),
        batch_size=16,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator

def train_model():
    """Train the diabetic retinopathy detection model"""
    print("Starting RetiScan model training...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load data
    train_generator, val_generator = load_data()
    
    # Create model
    print("Creating advanced CNN model...")
    model, base_model = create_advanced_model()
    
    # Compile model
    from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=[CategoricalAccuracy(), Precision(), Recall()]
    )

    
    print("Model architecture:")
    model.summary()
    
    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_categorical_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'models/best_retina_model.h5',
            monitor='val_categorical_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model (Phase 1: Frozen base)
    print("Phase 1: Training with frozen base model...")
    history1 = model.fit(
        train_generator,
        epochs=20,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tuning (Phase 2: Unfreeze base model)
    print("Phase 2: Fine-tuning with unfrozen base model...")
    base_model.trainable = True
    
    # Use lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=[CategoricalAccuracy(), Precision(), Recall()]
    )

    
    history2 = model.fit(
        train_generator,
        epochs=30,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Combine histories
    history = {
        'categorical_accuracy': history1.history['categorical_accuracy'] + history2.history['categorical_accuracy'],
        'val_categorical_accuracy': history1.history['val_categorical_accuracy'] + history2.history['val_categorical_accuracy'],
        'loss': history1.history['loss'] + history2.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss']
    }

    
    # Load best model
    model = keras.models.load_model('models/best_retina_model.h5')
    
    # Evaluate model
    print("\nEvaluating model...")
    val_loss, val_accuracy, val_precision, val_recall = model.evaluate(val_generator, verbose=0)
    
    print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")
    
    # Generate detailed predictions
    val_generator.reset()
    predictions = model.predict(val_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_generator.classes
    
    # Class names
    class_names = list(val_generator.class_indices.keys())
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Diabetic Retinopathy Detection')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved to: models/confusion_matrix.png")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['categorical_accuracy'], label='Training Accuracy')
    plt.plot(history['val_categorical_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    # ROC curve for binary classification (DR vs No DR)
    binary_true = (true_classes > 0).astype(int)
    binary_pred = (np.sum(predictions[:, 1:], axis=1))
    
    fpr, tpr, _ = roc_curve(binary_true, binary_pred)
    auc_score = roc_auc_score(binary_true, binary_pred)
    
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - DR Detection')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_metrics.png', dpi=300, bbox_inches='tight')
    print("Training metrics plot saved to: models/training_metrics.png")
    
    # Save final model
    model.save('models/retina_model.h5')
    print(f"Final model saved to: models/retina_model.h5")
    
    return model, val_accuracy, class_names

if __name__ == "__main__":
    # Set memory growth for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    model, accuracy, class_names = train_model()
    
    print(f"\n{'='*60}")
    print(f"RETISCAN MODEL TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Final Model Accuracy: {accuracy*100:.2f}%")
    print(f"Classes: {class_names}")
    
    if accuracy >= 0.90:
        print("✅ SUCCESS: Model achieved >90% accuracy!")
        print("🏥 RetiScan is ready for diabetic retinopathy detection!")
    else:
        print("⚠️  Model accuracy below 90%. Consider:")
        print("   - More training data")
        print("   - Longer training time")
        print("   - Hyperparameter tuning")