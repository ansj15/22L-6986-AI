import os
import zipfile
import shutil
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SET FILE PATHS ---
zip_path = "archive (3).zip"
extract_path = "instrument_data"

def extract_features(image_path):
    """Extract features from an image."""
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((128, 128))
        img_array = np.array(img)
        
        # Calculate color features
        color_means = np.mean(img_array, axis=(0,1))
        color_stds = np.std(img_array, axis=(0,1))
        
        # Calculate texture features (using grayscale image)
        gray = np.mean(img_array, axis=2)
        texture_features = np.percentile(gray, [25, 50, 75])
        
        # Calculate shape features
        edges = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
        
        # Combine all features
        features = np.concatenate([
            color_means,  # RGB means
            color_stds,  # RGB standard deviations
            texture_features,  # Texture percentiles
            [edges]  # Edge intensity
        ])
        
        return features
        
    except Exception as e:
        logger.error(f"Feature extraction error for {image_path}: {str(e)}")
        raise

def prepare_dataset():
    # Extract dataset
    if not os.path.exists(extract_path):
        logger.info("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    # Move files if needed
    nested_folder = os.path.join(extract_path, 'music_instruments')
    if os.path.exists(nested_folder):
        for item in os.listdir(nested_folder):
            shutil.move(os.path.join(nested_folder, item), extract_path)
        shutil.rmtree(nested_folder)

    logger.info(f"✅ Dataset ready at: {extract_path}")

def train_model():
    # Prepare data
    X = []  # Features
    y = []  # Labels
    classes = []  # Class names
    
    logger.info("Extracting features from images...")
    
    # Iterate through all instrument folders
    for class_name in sorted(os.listdir(extract_path)):
        class_path = os.path.join(extract_path, class_name)
        if os.path.isdir(class_path):
            classes.append(class_name)
            logger.info(f"Processing class: {class_name}")
            
            # Process each image in the class folder
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    features = extract_features(img_path)
                    X.append(features)
                    y.append(len(classes) - 1)  # Use class index as label
                except Exception as e:
                    logger.warning(f"Failed to process {img_path}: {str(e)}")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    logger.info("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=classes)
    logger.info(f"\nClassification Report:\n{report}")
    
    # Save model
    logger.info("Saving model...")
    joblib.dump(model, 'instrument_classifier.joblib')
    logger.info("✅ Model saved successfully!")

if __name__ == "__main__":
    prepare_dataset()
    train_model() 