from datetime import datetime
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import joblib
import logging
import hdbscan  # Add this import
from sklearn.preprocessing import normalize  # Add this import
# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model artifacts
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
clusterer = joblib.load('clusterer.pkl')

# Preprocess image (same as Lab 4)
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (37, 50))  # LFW dimensions (width=37, height=50)
        flattened = img.reshape(1, -1)
        scaled = scaler.transform(flattened)
        pca_reduced = pca.transform(scaled)
        normalized = normalize(pca_reduced)
        return normalized
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join('/tmp', filename)
        file.save(filepath)
        
        logger.info(f"Request received at {datetime.now()}: {filename}")


        # Preprocess and predict
        processed_data = preprocess_image(filepath)
        cluster_label, _ = hdbscan.approximate_predict(clusterer, processed_data)
        
        # Log the prediction
        logger.info(f"Prediction: Cluster {cluster_label[0]} for {filename}")

        return jsonify({"cluster": int(cluster_label[0])})

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500
    


# Simple frontend for testing
@app.route('/')
def home():
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)