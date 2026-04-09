# =============================================================================
# ALZHEIMER'S DISEASE DETECTION - FLASK BACKEND SERVER
# =============================================================================
# This is the main backend server for the Alzheimer's Detection System.
# It handles:
#   - Serving the frontend HTML/CSS/JS files
#   - Receiving brain MRI images from the frontend
#   - Running predictions using the trained deep learning model
#   - Generating Grad-CAM visual explanations
#   - Optionally generating LIME and SHAP advanced explanations
#   - Storing and retrieving prediction history
#   - Returning model performance metrics
# =============================================================================

# --- Standard Library Imports ---
import io          # For handling image byte streams
import base64      # For encoding images to base64 strings (for JSON transport)
import sys         # For exiting the program if model fails to load
import json        # For reading/writing JSON files (history, metrics)
import os          # For file path checks
from datetime import datetime  # For timestamping predictions

# --- Third-Party Imports ---
from flask import Flask, request, jsonify   # Flask web framework
from flask_cors import CORS                 # Allows cross-origin requests from browser
import tensorflow as tf                     # Deep learning framework
import numpy as np                          # Numerical array operations
import cv2                                  # OpenCV for image processing and Grad-CAM
from PIL import Image                       # Pillow for image loading and conversion

# =============================================================================
# APP CONFIGURATION
# =============================================================================

# Initialize Flask app
# static_folder='.' means Flask will serve static files from the current directory
# static_url_path='' means static files are served at the root URL
app = Flask(__name__, static_folder='.', static_url_path='')

# Enable CORS so the browser can make requests to this server
# (needed when frontend and backend run on different ports)
CORS(app)

# Path to the trained Keras model file
MODEL_PATH = 'alzheimer_128_best.h5'

# Global model variable — loaded once at startup and reused for all predictions
model = None

# Class names must match the order used during model training
# These correspond to the 4 Alzheimer's severity stages
CLASS_NAMES = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

# File path for storing prediction history as JSON
HISTORY_FILE = 'prediction_history.json'

# Create an empty history file if it doesn't already exist
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w') as f:
        json.dump([], f)

# =============================================================================
# MODEL LOADING
# =============================================================================

print("\n" + "=" * 60)
print("ALZHEIMER'S DETECTION SERVER")
print("=" * 60)

try:
    print("Loading model...")

    # Load the trained Keras model from disk
    # compile=False skips recompiling the model (faster loading, safe for inference)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Recompile the model manually after loading
    # This is required when compile=False is used
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"✅ Model loaded successfully")
    print(f"   Path: {MODEL_PATH}")
    print(f"   Input shape: {model.input_shape}")  # Should be (None, 128, 128, 3)

except Exception as e:
    # If the model fails to load, exit immediately — the server cannot function without it
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# =============================================================================
# STATIC FILE ROUTES
# =============================================================================

@app.route('/')
def home():
    """Serve the main HTML page (index.html) at the root URL."""
    return app.send_static_file('index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve any other static file (CSS, JS, images) by filename."""
    return app.send_static_file(filename)

# =============================================================================
# HEALTH CHECK ROUTE
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    Returns server status and whether the model is loaded.
    Useful for debugging and monitoring.
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH
    })

# =============================================================================
# PREDICTION ROUTE
# =============================================================================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    Accepts one or more brain MRI images via multipart form data.
    Returns predictions, confidence scores, Grad-CAM visualizations,
    and optionally LIME and SHAP explanations.
    """
    print("\n" + "-" * 60)
    print("📥 Prediction request received")

    # Ensure model is loaded before attempting prediction
    if model is None:
        print("❌ Model not loaded")
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # --- Validate that at least one image was uploaded ---
        if 'image' not in request.files:
            print("❌ No image in request")
            return jsonify({'error': 'No image provided'}), 400

        # Get all uploaded image files (supports batch upload of up to 5)
        image_files = request.files.getlist('image')
        print(f"   Number of images: {len(image_files)}")

        # Check if the user requested advanced explainability (LIME + SHAP)
        # This is slower (~30-60 seconds) so it's optional
        use_advanced = request.form.get('advanced', 'false').lower() == 'true'

        # --- Collect patient information from the form ---
        patient_data = {
            'patientName': request.form.get('patientName', ''),
            'patientId': request.form.get('patientId', ''),
            'age': request.form.get('age', ''),
            'gender': request.form.get('gender', ''),
            'contactNumber': request.form.get('contactNumber', ''),
            'email': request.form.get('email', ''),
            'medicalHistory': request.form.get('medicalHistory', ''),
            'symptoms': request.form.get('symptoms', ''),
            'duration': request.form.get('duration', ''),
            'notes': request.form.get('notes', ''),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # List to collect results for each uploaded image
        results = []

        # --- Process each uploaded image ---
        for idx, image_file in enumerate(image_files):
            print(f"\n   Processing image {idx + 1}/{len(image_files)}: {image_file.filename}")

            # Read raw bytes from the uploaded file
            image_bytes = image_file.read()
            print(f"   Size: {len(image_bytes)} bytes")

            # Open the image using PIL from the byte stream
            image = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if needed (e.g., grayscale or RGBA images)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Save the original (unresized) image as base64 for display in the frontend
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            original_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Resize image to 128x128 — the input size expected by the model
            image = image.resize((128, 128))

            # Convert PIL image to NumPy array and normalize pixel values to [0, 1]
            img_array = np.array(image, dtype=np.float32) / 255.0

            # Add batch dimension: shape becomes (1, 128, 128, 3)
            img_array = np.expand_dims(img_array, axis=0)
            print(f"   Preprocessed shape: {img_array.shape}")

            # --- Run Model Prediction ---
            print("   Predicting...")
            predictions = model.predict(img_array, verbose=0)

            # Get the index of the highest probability class
            pred_idx = np.argmax(predictions[0])

            # Get the confidence score (probability) for the predicted class
            confidence = float(predictions[0][pred_idx])

            # Map index to human-readable class name
            predicted_class = CLASS_NAMES[pred_idx]

            print(f"   ✅ Prediction: {predicted_class} ({confidence*100:.2f}%)")

            # =================================================================
            # GRAD-CAM VISUALIZATION
            # =================================================================
            # Grad-CAM (Gradient-weighted Class Activation Mapping) highlights
            # which regions of the brain scan the model focused on when making
            # its prediction. Red/yellow = high importance, blue = low importance.
            # =================================================================

            print("   Creating Grad-CAM...")
            img_rgb = np.array(image)  # Convert PIL image to NumPy for OpenCV

            try:
                # Find the last Conv2D layer in the model
                # This is the layer whose activations we use for Grad-CAM
                last_conv_layer = None
                for layer in reversed(model.layers):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        last_conv_layer = layer.name
                        break

                if last_conv_layer is None:
                    raise Exception("No Conv2D layer found")

                # Build a sub-model that outputs both:
                # 1. The activations of the last conv layer
                # 2. The final prediction probabilities
                grad_model = tf.keras.models.Model(
                    inputs=model.inputs,
                    outputs=[model.get_layer(last_conv_layer).output, model.output]
                )

                # Use GradientTape to record operations for automatic differentiation
                with tf.GradientTape() as tape:
                    img_tensor = tf.cast(img_array, tf.float32)
                    conv_outputs, predictions_tape = grad_model(img_tensor)

                    # We want gradients of the predicted class score
                    # with respect to the conv layer outputs
                    loss = predictions_tape[:, pred_idx]

                # Compute gradients of the class score w.r.t. conv layer outputs
                grads = tape.gradient(loss, conv_outputs)

                # Pool gradients over spatial dimensions to get importance weights
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

                # Weight the conv outputs by the pooled gradients
                conv_outputs = conv_outputs[0]
                heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap).numpy()

                # Apply ReLU (keep only positive activations) and normalize to [0, 1]
                heatmap = np.maximum(heatmap, 0)
                if heatmap.max() > 0:
                    heatmap = heatmap / heatmap.max()

                # Resize heatmap to match the input image size (128x128)
                heatmap = cv2.resize(heatmap, (128, 128))

                # Convert to uint8 and apply JET colormap (blue→green→yellow→red)
                heatmap = np.uint8(255 * heatmap)
                heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                # Convert from BGR (OpenCV default) to RGB
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

                # Blend original image (60%) with heatmap (40%) for overlay
                overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)
                print("   ✅ Real Grad-CAM generated")

            except Exception as e:
                # Fallback: use gradient saliency map if Grad-CAM fails
                # This computes gradients of the loss w.r.t. the input image directly
                print(f"   ⚠️ Grad-CAM failed ({e}), using saliency fallback...")
                img_tensor = tf.cast(img_array, tf.float32)
                with tf.GradientTape() as tape:
                    tape.watch(img_tensor)
                    preds = model(img_tensor)
                    loss = preds[:, pred_idx]

                # Compute absolute gradients and average across color channels
                grads = tape.gradient(loss, img_tensor)
                saliency = tf.abs(grads).numpy()[0]
                heatmap = np.mean(saliency, axis=-1)

                # Normalize and apply colormap
                heatmap = np.maximum(heatmap, 0)
                if heatmap.max() > 0:
                    heatmap = heatmap / heatmap.max()
                heatmap = np.uint8(255 * heatmap)
                heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)

            # Encode the Grad-CAM overlay image as base64 PNG for JSON transport
            _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            overlay_base64 = base64.b64encode(buffer).decode('utf-8')

            # --- Build result dictionary for this image ---
            result = {
                'filename': image_file.filename,
                'prediction': predicted_class,
                'confidence': f"{confidence * 100:.2f}%",
                'confidence_score': confidence,
                # All class probabilities as percentages (for the confidence bar chart)
                'all_predictions': {
                    CLASS_NAMES[i]: f"{predictions[0][i] * 100:.2f}%"
                    for i in range(len(CLASS_NAMES))
                },
                'gradcam': f"data:image/png;base64,{overlay_base64}",
                'original_image': f"data:image/png;base64,{original_image_base64}"
            }

            # --- Advanced Explainability (LIME + SHAP) ---
            # Only run on the first image to save processing time
            # Only runs if the user checked the "Advanced Explainability" checkbox
            if use_advanced and idx == 0:
                print("   Generating advanced explainability (LIME & SHAP)...")
                from explainability import generate_all_explanations

                advanced_explanations = generate_all_explanations(model, img_array, CLASS_NAMES)
                result['lime'] = advanced_explanations['lime']
                result['shap'] = advanced_explanations['shap']
                print("   ✅ Advanced explanations generated")

            results.append(result)

        # =================================================================
        # SAVE TO HISTORY
        # =================================================================

        # Build a history entry combining patient data and all image results
        history_entry = {
            'patient': patient_data,
            'results': results,
            'id': datetime.now().strftime('%Y%m%d%H%M%S')  # Unique ID based on timestamp
        }

        # Load existing history from file
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except:
            history = []  # Start fresh if file is missing or corrupted

        # Append new entry
        history.append(history_entry)

        # Limit history to the last 100 entries to prevent file from growing too large
        if len(history) > 100:
            history = history[-100:]

        # Save updated history back to file
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"✅ Saved to history (ID: {history_entry['id']})")

        # --- Build and return the final JSON response ---
        response = {
            'results': results,
            'count': len(results)
        }

        print("✅ Response sent")
        print("-" * 60)
        return jsonify(response)

    except Exception as e:
        # Catch-all error handler — log the full traceback and return error response
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("-" * 60)
        return jsonify({'error': str(e)}), 500

# =============================================================================
# METRICS ROUTE
# =============================================================================

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Returns model performance metrics from metrics_data.json.
    Includes accuracy, precision, recall, F1 score, and confusion matrix image.
    """
    try:
        # Check if the metrics file exists
        if not os.path.exists('metrics_data.json'):
            return jsonify({'error': 'Metrics not generated. Run generate_metrics.py first'}), 404

        with open('metrics_data.json', 'r') as f:
            metrics = json.load(f)

        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# MODEL INFO ROUTE
# =============================================================================

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """
    Returns static information about the loaded model.
    Used by the frontend to display model details.
    """
    return jsonify({
        'model_path': MODEL_PATH,
        'architecture': 'EfficientNetB0 + MobileNetV2',
        'input_size': '128x128',
        'accuracy': '73%',
        'classes': CLASS_NAMES,
        'training_samples': 8192
    })

# =============================================================================
# HISTORY ROUTES
# =============================================================================

@app.route('/history', methods=['GET'])
def get_history():
    """
    Returns all stored prediction history entries.
    Each entry includes patient details, images, and prediction results.
    """
    try:
        if not os.path.exists(HISTORY_FILE):
            return jsonify([])  # Return empty list if no history exists

        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)

        return jsonify(history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/history/<history_id>', methods=['DELETE'])
def delete_history(history_id):
    """
    Deletes a specific history entry by its ID.
    The ID is a timestamp string (e.g., '20260224103045').
    """
    try:
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)

        # Filter out the entry with the matching ID
        history = [h for h in history if h['id'] != history_id]

        # Save the updated history back to file
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# SERVER ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    print("\n🚀 Starting server on http://localhost:5001")
    print("=" * 60 + "\n")

    # Run the Flask development server
    # host='0.0.0.0' makes it accessible on all network interfaces
    # threaded=True allows handling multiple requests simultaneously
    # debug=False disables auto-reload (use True only during development)
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
