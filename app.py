from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from PIL import Image
import numpy as np
import cv2
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Class names for musical instruments
CLASS_NAMES = ['Accordion', 'Banjo', 'Cello', 'Drum', 'Guitar', 'Piano', 'Saxophone', 'Trumpet', 'Violin', 'Xylophone']

def extract_advanced_features(image):
    """Extract comprehensive features from the image."""
    try:
        # Ensure image is in the correct format
        if image is None or image.size == 0:
            raise ValueError("Invalid image input")

        # Convert to different color spaces
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        except cv2.error as e:
            logger.error(f"Color conversion error: {e}")
            raise ValueError("Failed to convert image color space")

        features = {}
        
        # 1. Basic Statistics
        features['brightness'] = float(np.mean(gray))
        features['brightness_std'] = float(np.std(gray))
        
        # 2. Color Features
        for i, channel in enumerate(['hue', 'saturation', 'value']):
            features[f'{channel}_mean'] = float(np.mean(hsv[:,:,i]))
            features[f'{channel}_std'] = float(np.std(hsv[:,:,i]))
        
        # 3. Edge Features
        edges = cv2.Canny(gray, 100, 200)
        features['edge_density'] = float(np.mean(edges > 0))
        
        # Calculate edge orientation histogram
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy)
        features['edge_magnitude_mean'] = float(np.mean(mag))
        
        # Calculate edge direction histogram
        angles = ang * 180 / np.pi
        hist, _ = np.histogram(angles, bins=8, range=(0, 180))
        features['vertical_edges'] = float(hist[0] + hist[7])  # Near 0 and 180 degrees
        features['horizontal_edges'] = float(hist[3] + hist[4])  # Near 90 degrees
        
        # 4. Shape Features
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            features['contour_area'] = float(cv2.contourArea(largest_contour))
            features['contour_perimeter'] = float(cv2.arcLength(largest_contour, True))
            
            # Calculate circularity
            if features['contour_perimeter'] > 0:
                features['circularity'] = float(4 * np.pi * features['contour_area'] / (features['contour_perimeter'] ** 2))
            else:
                features['circularity'] = 0.0
            
            if len(largest_contour) >= 5:
                try:
                    (x, y), (width, height), angle = cv2.fitEllipse(largest_contour)
                    features['aspect_ratio'] = float(max(width, height) / min(width, height) if min(width, height) > 0 else 1)
                    features['orientation'] = float(angle)
                    
                    # Calculate elongation
                    features['elongation'] = float(abs(width - height) / (width + height) if (width + height) > 0 else 0)
                except cv2.error:
                    features['aspect_ratio'] = 1.0
                    features['orientation'] = 0.0
                    features['elongation'] = 0.0
            else:
                features['aspect_ratio'] = 1.0
                features['orientation'] = 0.0
                features['elongation'] = 0.0
            
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            features['convexity'] = float(hull_area / features['contour_area'] if features['contour_area'] > 0 else 1)
            
            # Calculate solidity
            features['solidity'] = float(features['contour_area'] / hull_area if hull_area > 0 else 1)
        else:
            features['contour_area'] = 0.0
            features['contour_perimeter'] = 0.0
            features['aspect_ratio'] = 1.0
            features['orientation'] = 0.0
            features['convexity'] = 1.0
            features['circularity'] = 0.0
            features['elongation'] = 0.0
            features['solidity'] = 1.0
        
        # 5. Texture Features
        kernel_sizes = [3, 7, 11]
        for size in kernel_sizes:
            blur = cv2.GaussianBlur(gray, (size, size), 0)
            features[f'texture_contrast_{size}'] = float(np.std(blur))
        
        # 6. Symmetry Features
        height, width = gray.shape
        left_half = gray[:, :width//2]
        right_half = cv2.flip(gray[:, width//2:], 1)
        min_width = min(left_half.shape[1], right_half.shape[1])
        features['horizontal_symmetry'] = float(np.mean(np.abs(left_half[:, :min_width] - right_half[:, :min_width])))
        
        # Calculate vertical symmetry
        top_half = gray[:height//2, :]
        bottom_half = cv2.flip(gray[height//2:, :], 0)
        min_height = min(top_half.shape[0], bottom_half.shape[0])
        features['vertical_symmetry'] = float(np.mean(np.abs(top_half[:min_height, :] - bottom_half[:min_height, :])))

        # Validate features
        for key, value in features.items():
            if not np.isfinite(value):
                features[key] = 0.0
                logger.warning(f"Invalid value for feature {key}, setting to 0.0")

        return features

    except Exception as e:
        logger.error(f"Feature extraction error: {str(e)}")
        raise

def calculate_instrument_scores(features):
    """Calculate scores for each instrument based on their characteristic features."""
    scores = {}
    
    # Accordion characteristics
    accordion_score = 0
    if 1.3 < features['aspect_ratio'] < 2.4:  # Wider range for rectangular shape
        accordion_score += 30
    if features['edge_density'] > 0.15:  # More lenient on edge patterns
        accordion_score += 25
    if features['brightness'] > 140:  # More lenient on brightness
        accordion_score += 15
    if features['vertical_edges'] > 0.8 * features['horizontal_edges']:  # More lenient on vertical lines
        accordion_score += 20
    scores['Accordion'] = accordion_score

    # Banjo characteristics
    banjo_score = 0
    if features['circularity'] > 0.6:  # More lenient on roundness
        banjo_score += 25
    if 0.1 < features['edge_density'] < 0.4:  # Wider range for edges
        banjo_score += 20
    if features['brightness'] > 150:  # More lenient on brightness
        banjo_score += 15
    if features['texture_contrast_7'] > 12:  # More lenient on texture
        banjo_score += 15
    scores['Banjo'] = banjo_score

    # Cello characteristics
    cello_score = 0
    if 2.5 < features['aspect_ratio'] < 4.2:  # Wider range for tall shape
        cello_score += 30
    if 50 < features['brightness'] < 150:  # Wider range for wood color
        cello_score += 20
    if features['texture_contrast_7'] > 15:  # More lenient on wood grain
        cello_score += 20
    if 0.1 < features['edge_density'] < 0.35:  # Wider range for edges
        cello_score += 15
    scores['Cello'] = cello_score

    # Drum characteristics
    drum_score = 0
    if features['circularity'] > 0.8:  # Very circular
        drum_score += 30
    if features['aspect_ratio'] < 1.3:  # Nearly circular
        drum_score += 25
    if 0.1 < features['edge_density'] < 0.25:  # Rim details
        drum_score += 15
    if features['horizontal_symmetry'] < 40:  # Symmetrical
        drum_score += 15
    scores['Drum'] = drum_score

    # Guitar characteristics
    guitar_score = 0
    if 2.2 < features['aspect_ratio'] < 3.5:  # Guitar shape
        guitar_score += 30
    if 0.15 < features['edge_density'] < 0.3:  # Strings and frets
        guitar_score += 25
    if 70 < features['brightness'] < 160:  # Wood color
        guitar_score += 15
    if features['texture_contrast_7'] > 15:  # Wood grain
        guitar_score += 15
    scores['Guitar'] = guitar_score

    # Piano characteristics
    piano_score = 0
    if features['horizontal_edges'] > 1.5 * features['vertical_edges']:  # Keys pattern
        piano_score += 30
    if features['brightness'] > 180:  # White keys
        piano_score += 20
    if features['contour_area'] > 35000:  # Large instrument
        piano_score += 20
    if features['horizontal_symmetry'] < 50:  # Symmetric
        piano_score += 15
    scores['Piano'] = piano_score

    # Saxophone characteristics
    sax_score = 0
    if 2.0 < features['aspect_ratio'] < 3.0:  # Curved shape
        sax_score += 25
    if features['value_mean'] > 160:  # Metallic
        sax_score += 25
    if 0.2 < features['edge_density'] < 0.4:  # Complex shape with keys
        sax_score += 20
    if features['saturation_mean'] < 60:  # Metallic color
        sax_score += 15
    scores['Saxophone'] = sax_score

    # Trumpet characteristics
    trumpet_score = 0
    if 1.8 < features['aspect_ratio'] < 2.5:  # Horizontal shape
        trumpet_score += 25
    if features['value_mean'] > 170:  # Very bright/metallic
        trumpet_score += 25
    if 0.15 < features['edge_density'] < 0.3:  # Valves and tubes
        trumpet_score += 20
    if features['saturation_mean'] < 50:  # Metallic color
        trumpet_score += 15
    scores['Trumpet'] = trumpet_score

    # Violin characteristics
    violin_score = 0
    if 2.5 < features['aspect_ratio'] < 3.8:  # Violin shape
        violin_score += 30
    if 60 < features['brightness'] < 130:  # Dark wood
        violin_score += 20
    if features['texture_contrast_7'] > 20:  # Wood grain
        violin_score += 20
    if 0.15 < features['edge_density'] < 0.35:  # F-holes and strings
        violin_score += 15
    scores['Violin'] = violin_score

    # Xylophone characteristics
    xylophone_score = 0
    if features['horizontal_edges'] > 2 * features['vertical_edges']:  # Strong horizontal bars
        xylophone_score += 35
    if features['brightness'] > 150:  # Usually bright
        xylophone_score += 20
    if 0.2 < features['edge_density'] < 0.4:  # Gaps between bars
        xylophone_score += 20
    if features['aspect_ratio'] > 2.0:  # Wide instrument
        xylophone_score += 15
    scores['Xylophone'] = xylophone_score

    return scores

def enhanced_classifier(features):
    """A scoring-based classifier that considers multiple features for each instrument."""
    try:
        # Get scores for each instrument
        scores = calculate_instrument_scores(features)
        
        # Log the scores for debugging
        logger.info(f"Instrument scores: {scores}")
        
        # Find the instrument with the highest score
        max_score = -1
        predicted_class = None
        
        for instrument, score in scores.items():
            if score > max_score:
                max_score = score
                predicted_class = instrument
        
        # Only make a prediction if we have a reasonable confidence
        if max_score < 30:  # Lowered from 50 to 30
            logger.warning(f"Low confidence scores (max: {max_score}), prediction may be unreliable")
            if max_score < 20:  # Lowered from 35 to 20
                return "Unknown", max_score
        
        # Check if second highest score is too close
        second_max = max(score for instr, score in scores.items() if instr != predicted_class)
        if max_score - second_max < 10:  # Lowered from 15 to 10
            logger.warning(f"Ambiguous prediction: top score {max_score} vs second {second_max}")
            return "Unknown", max_score
        
        return predicted_class, max_score

    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        return "Unknown", 0  # Return unknown on error

def preprocess_image(image_bytes):
    """Convert image bytes to numpy array and resize."""
    try:
        # Convert bytes to PIL Image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if img.mode not in ['RGB']:
            img = img.convert('RGB')
        
        # Resize to a standard size
        img = img.resize((224, 224))
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Ensure the image is in the correct format
        if img_array.shape != (224, 224, 3):
            raise ValueError(f"Invalid image shape: {img_array.shape}")
        
        return img_array
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        return JSONResponse(content={"error": "No file uploaded"}, status_code=400)
    
    try:
        # Read the file
        contents = await file.read()
        
        # Log file info
        logger.info(f"Processing file: {file.filename}, size: {len(contents)} bytes")
        
        # Preprocess the image
        img_array = preprocess_image(contents)
        logger.info(f"Image preprocessed successfully, shape: {img_array.shape}")
        
        # Extract features
        features = extract_advanced_features(img_array)
        logger.info(f"Features extracted: {features}")
        
        # Make prediction
        predicted_class, score = enhanced_classifier(features)
        logger.info(f"Predicted class: {predicted_class}, Score: {score}")
        
        # Calculate normalized confidence (scale score from 0-100 to 0-1)
        confidence = min(0.95, max(0.5, score / 100))
        
        return {
            "prediction": predicted_class,
            "confidence": float(confidence)
        }
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return JSONResponse(
            content={"error": f"Error processing image: {str(e)}"},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="localhost", port=8000, reload=True) 