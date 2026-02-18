import cv2
import numpy as np
import pickle
import os
try:
    from tensorflow import keras
except ImportError:
    import keras

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'braille_cnn.h5')
ENCODER_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'label_encoder.pkl')

# ── Load model and label encoder once at startup ──────────────────────────────
model = keras.models.load_model(MODEL_PATH)

with open(ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

print(f"[detector] Model loaded. Classes: {le.classes_}")


# ── Segment braille cells from a preprocessed image ───────────────────────────
def segment_cells(gray_image):
    """
    Takes a grayscale image and returns a list of cropped cell images
    each containing one braille character, along with their (x, y) positions.

    Returns:
        cells: list of (x, y, crop) tuples sorted left-to-right, top-to-bottom
    """
    # Threshold to binary
    _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate to merge dots within the same cell
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 30))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # Find contours of each braille cell
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter out noise (too small or too large)
        if w < 10 or h < 10:
            continue
        if w > gray_image.shape[1] * 0.8:
            continue

        # Crop the cell from the original grayscale image
        crop = gray_image[y:y+h, x:x+w]
        cells.append((x, y, crop))

    # Sort cells left-to-right, top-to-bottom (like reading order)
    if not cells:
        return []

    # Group into rows by y coordinate (within 20px = same row)
    cells.sort(key=lambda c: c[1])
    rows = []
    current_row = [cells[0]]

    for cell in cells[1:]:
        if abs(cell[1] - current_row[0][1]) < 20:
            current_row.append(cell)
        else:
            rows.append(sorted(current_row, key=lambda c: c[0]))
            current_row = [cell]
    rows.append(sorted(current_row, key=lambda c: c[0]))

    # Flatten rows back into a single sorted list
    sorted_cells = [cell for row in rows for cell in row]
    return sorted_cells


# ── Classify a single cell crop ───────────────────────────────────────────────
def classify_cell(crop):
    """
    Takes a single cell crop (grayscale), resizes to 28x28,
    converts to BGR, normalizes, and runs through the CNN.

    Returns:
        letter (str): predicted letter a-z
        confidence (float): prediction confidence 0-1
    """
    # Resize to 28x28 (what the model expects)
    resized = cv2.resize(crop, (28, 28))

    # Convert grayscale to BGR (model was trained on BGR images)
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    # Normalize to 0-1
    normalized = resized.astype('float32') / 255.0

    # Add batch dimension: (1, 28, 28, 3)
    input_tensor = np.expand_dims(normalized, axis=0)

    # Predict
    predictions = model.predict(input_tensor, verbose=0)
    class_index = np.argmax(predictions[0])
    confidence = float(predictions[0][class_index])

    letter = le.inverse_transform([class_index])[0]
    return letter, confidence


# ── Main detection function ────────────────────────────────────────────────────
def detect_braille(preprocessed_image, min_confidence=0.3):
    """
    Main function called by app.py.
    Takes a preprocessed BGR or grayscale image (numpy array).
    Returns detected text as a string.

    Args:
        preprocessed_image: numpy array (output from preprocessor.py)
        min_confidence: minimum confidence threshold (default 0.3)

    Returns:
        text (str): detected letters joined as a string
        results (list): list of dicts with letter, confidence, position
    """
    # Convert to grayscale if needed
    if len(preprocessed_image.shape) == 3:
        gray = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = preprocessed_image

    # Segment into individual cells
    cells = segment_cells(gray)

    if not cells:
        return "", []

    # Classify each cell
    results = []
    letters = []

    for x, y, crop in cells:
        letter, confidence = classify_cell(crop)

        if confidence >= min_confidence:
            letters.append(letter)
            results.append({
                'letter': letter,
                'confidence': round(confidence, 3),
                'position': {'x': int(x), 'y': int(y)}
            })
        else:
            # Low confidence — mark as unknown
            letters.append('?')
            results.append({
                'letter': '?',
                'confidence': round(confidence, 3),
                'position': {'x': int(x), 'y': int(y)}
            })

    text = ''.join(letters)
    return text, results