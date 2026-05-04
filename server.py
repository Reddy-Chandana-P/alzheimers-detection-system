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

            print("   Creating Grad-CAM++...")
            img_rgb = np.array(image)  # Convert PIL image to NumPy for OpenCV

            # Create brain mask to restrict heatmap to inside brain only
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            _, brain_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
            brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, kernel)
            brain_mask_3ch = cv2.merge([brain_mask, brain_mask, brain_mask])

            try:
                # Find the last Conv2D layer
                last_conv_layer = None
                for layer in reversed(model.layers):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        last_conv_layer = layer.name
                        break

                if last_conv_layer is None:
                    raise Exception("No Conv2D layer found")

                # Build grad model outputting conv activations + predictions
                grad_model = tf.keras.models.Model(
                    inputs=model.inputs,
                    outputs=[model.get_layer(last_conv_layer).output, model.output]
                )

                # =============================================================
                # GRAD-CAM++ ALGORITHM
                # =============================================================
                # Grad-CAM++ improves on Grad-CAM by using second-order gradients
                # (second derivatives of the loss w.r.t. conv activations).
                #
                # Key difference from Grad-CAM:
                #   - Grad-CAM:   weights = mean(gradients)
                #   - Grad-CAM++: weights = sum(alpha * ReLU(gradients))
                #     where alpha accounts for the importance of each gradient
                #     using second-order and third-order gradient information.
                #
                # This produces more accurate localization, especially when
                # multiple instances of the same class appear in the image
                # (e.g., bilateral hippocampal atrophy in Alzheimer's).
                # =============================================================

                with tf.GradientTape() as tape2:
                    with tf.GradientTape() as tape1:
                        with tf.GradientTape() as tape0:
                            img_tensor = tf.cast(img_array, tf.float32)
                            conv_outputs, predictions_tape = grad_model(img_tensor)
                            loss = predictions_tape[:, pred_idx]

                        # First-order gradients: dL/dA
                        grads_1 = tape0.gradient(loss, conv_outputs)

                    # Second-order gradients: d²L/dA²
                    grads_2 = tape1.gradient(grads_1, conv_outputs)

                # Third-order gradients: d³L/dA³
                grads_3 = tape2.gradient(grads_2, conv_outputs)

                # Compute alpha weights using second and third order gradients
                # alpha_k_c = grad_2 / (2 * grad_2 + sum(A * grad_3) + eps)
                conv_outputs_val = conv_outputs.numpy()[0]   # Shape: (H, W, C)
                grads_1_val = grads_1.numpy()[0]
                grads_2_val = grads_2.numpy()[0]
                grads_3_val = grads_3.numpy()[0]

                # Sum of activations weighted by third-order gradients
                global_sum = np.sum(conv_outputs_val, axis=(0, 1))  # Shape: (C,)

                # Alpha: importance weight for each channel
                alpha_num   = grads_2_val
                alpha_denom = 2.0 * grads_2_val + global_sum[np.newaxis, np.newaxis, :] * grads_3_val + 1e-7
                alpha = alpha_num / alpha_denom

                # Apply ReLU to first-order gradients before weighting
                weights = np.sum(alpha * np.maximum(grads_1_val, 0), axis=(0, 1))  # Shape: (C,)

                # Weighted combination of conv feature maps
                heatmap = np.sum(weights * conv_outputs_val, axis=-1)  # Shape: (H, W)

                # Apply ReLU and normalize
                heatmap = np.maximum(heatmap, 0)
                if heatmap.max() > 0:
                    heatmap = heatmap / heatmap.max()

                # Resize to image size
                heatmap = cv2.resize(heatmap, (128, 128))

                # Apply JET colormap
                heatmap = np.uint8(255 * heatmap)
                heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

                # Apply brain mask — no color bleed outside brain
                heatmap_colored = np.where(brain_mask_3ch > 0, heatmap_colored, 0)

                # Blend with original image inside brain only
                overlay = img_rgb.copy()
                brain_pixels = brain_mask_3ch > 0
                overlay[brain_pixels] = cv2.addWeighted(img_rgb, 0.5, heatmap_colored, 0.5, 0)[brain_pixels]
                print("   ✅ Grad-CAM++ generated")

            except Exception as e:
                # Fallback to standard Grad-CAM if Grad-CAM++ fails
                print(f"   ⚠️ Grad-CAM++ failed ({e}), falling back to Grad-CAM...")
                try:
                    grad_model = tf.keras.models.Model(
                        inputs=model.inputs,
                        outputs=[model.get_layer(last_conv_layer).output, model.output]
                    )
                    with tf.GradientTape() as tape:
                        img_tensor = tf.cast(img_array, tf.float32)
                        conv_outputs, predictions_tape = grad_model(img_tensor)
                        loss = predictions_tape[:, pred_idx]
                    grads = tape.gradient(loss, conv_outputs)
                    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                    conv_outputs = conv_outputs[0]
                    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
                    heatmap = tf.squeeze(heatmap).numpy()
                    heatmap = np.maximum(heatmap, 0)
                    if heatmap.max() > 0:
                        heatmap = heatmap / heatmap.max()
                    heatmap = cv2.resize(heatmap, (128, 128))
                    heatmap = np.uint8(255 * heatmap)
                    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                    heatmap_colored = np.where(brain_mask_3ch > 0, heatmap_colored, 0)
                    overlay = img_rgb.copy()
                    brain_pixels = brain_mask_3ch > 0
                    overlay[brain_pixels] = cv2.addWeighted(img_rgb, 0.5, heatmap_colored, 0.5, 0)[brain_pixels]
                    print("   ✅ Grad-CAM fallback generated")
                except Exception as e2:
                    print(f"   ⚠️ Grad-CAM also failed ({e2}), using saliency fallback...")
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

                # Apply brain mask to fallback too — no color bleed outside brain
                heatmap_colored = np.where(brain_mask_3ch > 0, heatmap_colored, 0)
                overlay = img_rgb.copy()
                brain_pixels = brain_mask_3ch > 0
                overlay[brain_pixels] = cv2.addWeighted(img_rgb, 0.5, heatmap_colored, 0.5, 0)[brain_pixels]

            # Encode the Grad-CAM overlay image as base64 PNG for JSON transport
            _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            overlay_base64 = base64.b64encode(buffer).decode('utf-8')

            # =================================================================
            # ANATOMICAL REGION ANALYSIS
            # =================================================================
            # Divide the brain scan into 9 anatomical regions based on spatial
            # position. Calculate the average Grad-CAM activation in each region
            # to identify which brain areas the model focused on most.
            #
            # Region map (3x3 grid on 128x128 image):
            #   ┌─────────────────────────────────┐
            #   │  Frontal  │  Parietal │  Frontal │  (top row)
            #   │  (Left)   │  (Top)    │  (Right) │
            #   ├───────────┼───────────┼──────────┤
            #   │ Temporal  │Hippocampus│ Temporal │  (middle row)
            #   │  (Left)   │ /Ventricle│  (Right) │
            #   ├───────────┼───────────┼──────────┤
            #   │ Occipital │ Brainstem │ Occipital│  (bottom row)
            #   │  (Left)   │           │  (Right) │
            #   └─────────────────────────────────┘
            # =================================================================

            # Use the normalized heatmap (before colormap) for region analysis
            # heatmap_norm is float [0,1] — higher = more important
            heatmap_norm = cv2.resize(
                np.maximum(heatmap.astype(np.float32) / 255.0, 0),
                (128, 128)
            )

            # Apply brain mask to exclude background from region scores
            heatmap_norm = heatmap_norm * (brain_mask / 255.0)

            h, w = 128, 128
            r, c = h // 3, w // 3  # Each region is ~42x42 pixels

            # Define 9 anatomical regions with their grid positions and names
            regions = {
                'Left Frontal Lobe':    heatmap_norm[0:r,   0:c],
                'Parietal Lobe (Top)':  heatmap_norm[0:r,   c:2*c],
                'Right Frontal Lobe':   heatmap_norm[0:r,   2*c:w],
                'Left Temporal Lobe':   heatmap_norm[r:2*r, 0:c],
                'Hippocampus/Ventricle':heatmap_norm[r:2*r, c:2*c],
                'Right Temporal Lobe':  heatmap_norm[r:2*r, 2*c:w],
                'Left Occipital Lobe':  heatmap_norm[2*r:h, 0:c],
                'Brainstem':            heatmap_norm[2*r:h, c:2*c],
                'Right Occipital Lobe': heatmap_norm[2*r:h, 2*c:w],
            }

            # Calculate mean activation per region (only non-zero pixels)
            region_scores = {}
            for region_name, region_data in regions.items():
                nonzero = region_data[region_data > 0]
                region_scores[region_name] = float(np.mean(nonzero)) if len(nonzero) > 0 else 0.0

            # Sort regions by activation score (highest first)
            sorted_regions = sorted(region_scores.items(), key=lambda x: x[1], reverse=True)

            # Top 3 most activated regions
            top_regions = sorted_regions[:3]

            # Build anatomical summary text
            anatomical_summary = {
                'top_regions': [
                    {
                        'name': name,
                        'score': round(score * 100, 1),
                        'level': 'High' if score > 0.5 else 'Moderate' if score > 0.25 else 'Low'
                    }
                    for name, score in top_regions
                ],
                'all_regions': {name: round(score * 100, 1) for name, score in sorted_regions}
            }

            print(f"   ✅ Anatomical analysis: top region = {top_regions[0][0]} ({top_regions[0][1]*100:.1f}%)")

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
                'original_image': f"data:image/png;base64,{original_image_base64}",
                'anatomical_regions': anatomical_summary
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
