from datetime import datetime
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import joblib
import sys
from pathlib import Path
import logging
import hdbscan
from sklearn.preprocessing import normalize

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model loading code remains the same
try:
    scaler = joblib.load('scaler.pkl')
    logger.info("✅ Successfully loaded scaler.pkl")
except Exception as e:
    logger.error(f"❌ Failed to load scaler.pkl: {e}")
    sys.exit(1)

try:
    pca = joblib.load('pca.pkl')
    logger.info("✅ Successfully loaded pca.pkl")
except Exception as e:
    logger.error(f"❌ Failed to load pca.pkl: {e}")
    sys.exit(1)

try:
    clusterer = joblib.load('clusterer.pkl')
    logger.info("✅ Successfully loaded clusterer.pkl")
except Exception as e:
    logger.error(f"❌ Failed to load clusterer.pkl: {e}")
    sys.exit(1)

MODEL_FILES = ['scaler.pkl', 'pca.pkl', 'clusterer.pkl']
for file in MODEL_FILES:
    if not Path(file).exists():
        logger.error(f"❌ Missing required model file: {file}")
        sys.exit(1)

def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (37, 50))
        flattened = img.reshape(1, -1)
        scaled = scaler.transform(flattened)
        pca_reduced = pca.transform(scaled)
        normalized = normalize(pca_reduced)
        return normalized
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

TEMP_DIR = os.path.join(os.getcwd(), 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        os.makedirs(TEMP_DIR, exist_ok=True)
        filename = secure_filename(file.filename)
        filepath = os.path.join(TEMP_DIR, filename)
        file.save(filepath)
        logger.info(f"Request received: {filename}")

        processed_data = preprocess_image(filepath)
        cluster_label, _ = hdbscan.approximate_predict(clusterer, processed_data)

        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Removed temp file: {filepath}")

        return jsonify({"cluster": int(cluster_label[0])})

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)