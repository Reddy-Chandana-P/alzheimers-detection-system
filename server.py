from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import base64
import sys
import json
import os
from datetime import datetime

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

MODEL_PATH = 'alzheimer_128_best.h5'
model = None
CLASS_NAMES = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']
HISTORY_FILE = 'prediction_history.json'

# Initialize history file if it doesn't exist
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w') as f:
        json.dump([], f)

print("\n" + "=" * 60)
print("ALZHEIMER'S DETECTION SERVER")
print("=" * 60)

# Load model
try:
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"✅ Model loaded successfully")
    print(f"   Path: {MODEL_PATH}")
    print(f"   Input shape: {model.input_shape}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return app.send_static_file(filename)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH
    })

@app.route('/predict', methods=['POST'])
def predict():
    print("\n" + "-" * 60)
    print("📥 Prediction request received")
    
    if model is None:
        print("❌ Model not loaded")
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Check for images
        if 'image' not in request.files:
            print("❌ No image in request")
            return jsonify({'error': 'No image provided'}), 400
        
        # Get all uploaded images
        image_files = request.files.getlist('image')
        print(f"   Number of images: {len(image_files)}")
        
        # Check if advanced explainability is requested
        use_advanced = request.form.get('advanced', 'false').lower() == 'true'
        
        # Get patient data
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
        
        # Process each image
        results = []
        
        for idx, image_file in enumerate(image_files):
            print(f"\n   Processing image {idx + 1}/{len(image_files)}: {image_file.filename}")
            
            # Read and preprocess
            image_bytes = image_file.read()
            print(f"   Size: {len(image_bytes)} bytes")
            
            # Load image
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save original image as base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            original_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Resize
            image = image.resize((128, 128))
            
            # To array
            img_array = np.array(image, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            print(f"   Preprocessed shape: {img_array.shape}")
            
            # Predict
            print("   Predicting...")
            predictions = model.predict(img_array, verbose=0)
            
            pred_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][pred_idx])
            predicted_class = CLASS_NAMES[pred_idx]
            
            print(f"   ✅ Prediction: {predicted_class} ({confidence*100:.2f}%)")
            
            # Real Grad-CAM implementation
            print("   Creating Grad-CAM...")
            img_rgb = np.array(image)
            
            try:
                # Find the last convolutional layer
                last_conv_layer = None
                for layer in reversed(model.layers):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        last_conv_layer = layer.name
                        break
                
                if last_conv_layer is None:
                    raise Exception("No Conv2D layer found")
                
                # Build grad model
                grad_model = tf.keras.models.Model(
                    inputs=model.inputs,
                    outputs=[model.get_layer(last_conv_layer).output, model.output]
                )
                
                # Compute gradients
                with tf.GradientTape() as tape:
                    img_tensor = tf.cast(img_array, tf.float32)
                    conv_outputs, predictions_tape = grad_model(img_tensor)
                    loss = predictions_tape[:, pred_idx]
                
                grads = tape.gradient(loss, conv_outputs)
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                conv_outputs = conv_outputs[0]
                heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap).numpy()
                
                # Normalize
                heatmap = np.maximum(heatmap, 0)
                if heatmap.max() > 0:
                    heatmap = heatmap / heatmap.max()
                
                # Resize to image size
                heatmap = cv2.resize(heatmap, (128, 128))
                heatmap = np.uint8(255 * heatmap)
                heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)
                print("   ✅ Real Grad-CAM generated")
                
            except Exception as e:
                print(f"   ⚠️ Grad-CAM failed ({e}), using saliency fallback...")
                img_tensor = tf.cast(img_array, tf.float32)
                with tf.GradientTape() as tape:
                    tape.watch(img_tensor)
                    preds = model(img_tensor)
                    loss = preds[:, pred_idx]
                grads = tape.gradient(loss, img_tensor)
                saliency = tf.abs(grads).numpy()[0]
                heatmap = np.mean(saliency, axis=-1)
                heatmap = np.maximum(heatmap, 0)
                if heatmap.max() > 0:
                    heatmap = heatmap / heatmap.max()
                heatmap = np.uint8(255 * heatmap)
                heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)
            
            # Encode
            _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            overlay_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Build result for this image
            result = {
                'filename': image_file.filename,
                'prediction': predicted_class,
                'confidence': f"{confidence * 100:.2f}%",
                'confidence_score': confidence,
                'all_predictions': {
                    CLASS_NAMES[i]: f"{predictions[0][i] * 100:.2f}%" 
                    for i in range(len(CLASS_NAMES))
                },
                'gradcam': f"data:image/png;base64,{overlay_base64}",
                'original_image': f"data:image/png;base64,{original_image_base64}"
            }
            
            # Add LIME and SHAP if requested (only for first image to save time)
            if use_advanced and idx == 0:
                print("   Generating advanced explainability (LIME & SHAP)...")
                from explainability import generate_all_explanations
                
                advanced_explanations = generate_all_explanations(model, img_array, CLASS_NAMES)
                result['lime'] = advanced_explanations['lime']
                result['shap'] = advanced_explanations['shap']
                print("   ✅ Advanced explanations generated")
            
            results.append(result)
        
        # Save to history
        history_entry = {
            'patient': patient_data,
            'results': results,
            'id': datetime.now().strftime('%Y%m%d%H%M%S')
        }
        
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except:
            history = []
        
        history.append(history_entry)
        
        # Keep only last 100 entries
        if len(history) > 100:
            history = history[-100:]
        
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"✅ Saved to history (ID: {history_entry['id']})")
        
        # Response
        response = {
            'results': results,
            'count': len(results)
        }
        
        print("✅ Response sent")
        print("-" * 60)
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("-" * 60)
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Return model performance metrics"""
    try:
        import json
        import os
        
        # Check if metrics file exists
        if not os.path.exists('metrics_data.json'):
            return jsonify({'error': 'Metrics not generated. Run generate_metrics.py first'}), 404
        
        with open('metrics_data.json', 'r') as f:
            metrics = json.load(f)
        
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Return model information"""
    return jsonify({
        'model_path': MODEL_PATH,
        'architecture': 'EfficientNetB0 + MobileNetV2',
        'input_size': '128x128',
        'accuracy': '73%',
        'classes': CLASS_NAMES,
        'training_samples': 8192
    })

@app.route('/history', methods=['GET'])
def get_history():
    """Return prediction history"""
    try:
        if not os.path.exists(HISTORY_FILE):
            return jsonify([])
        
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
        
        return jsonify(history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history/<history_id>', methods=['DELETE'])
def delete_history(history_id):
    """Delete a specific history entry"""
    try:
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
        
        history = [h for h in history if h['id'] != history_id]
        
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n🚀 Starting server on http://localhost:5001")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
