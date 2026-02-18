import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from utils.preprocessor import preprocess
from utils.detector import detect_braille
from utils.llm_handler import process_braille_text

load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ── Health check ───────────────────────────────────────────────────────────────
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'ok', 'message': 'Braille API is running'})


# ── Main conversion endpoint ───────────────────────────────────────────────────
@app.route('/convert', methods=['POST'])
def convert():
    """
    Accepts an image file from Flutter.
    Runs it through the full pipeline:
        preprocess → detect → llm cleanup
    Returns JSON with the detected English text.
    """

    # ── Check image was sent ───────────────────────────────────────────────────
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image file provided. Send image as multipart/form-data with key "image"'
        }), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({
            'success': False,
            'error': 'Empty filename'
        }), 400

    # ── Save image temporarily ─────────────────────────────────────────────────
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(filepath)

    try:
        # ── Step 1: Preprocess ─────────────────────────────────────────────────
        with open(filepath, 'rb') as f:
            image_bytes = f.read()

        preprocessed = preprocess(image_bytes)

        if preprocessed is None:
            return jsonify({
                'success': False,
                'error': 'Failed to preprocess image'
            }), 500

        # ── Step 2: Detect braille characters ─────────────────────────────────
        raw_text, detection_results = detect_braille(preprocessed)

        if not raw_text:
            return jsonify({
                'success': False,
                'error': 'No braille characters detected in image',
                'tip': 'Make sure the image is clear and well-lit'
            }), 200

        # ── Step 3: LLM cleanup ────────────────────────────────────────────────
        llm_result = process_braille_text(raw_text)

        # ── Return result ──────────────────────────────────────────────────────
        return jsonify({
            'success': True,
            'raw_text': raw_text,
            'corrected_text': llm_result['corrected'],
            'llm_success': llm_result['success'],
            'detections': detection_results,
            'character_count': len(raw_text)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

    finally:
        # ── Clean up uploaded file ─────────────────────────────────────────────
        if os.path.exists(filepath):
            os.remove(filepath)


# ── Run server ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'development') == 'development'
    print(f"Starting Braille API on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=debug)