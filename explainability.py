import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap
import tensorflow as tf
import cv2
import io
import base64

def generate_lime_explanation(model, image, class_names, num_samples=100):
    """
    Generate LIME explanation for image classification
    
    Args:
        model: Trained Keras model
        image: Preprocessed image (1, 128, 128, 3)
        class_names: List of class names
        num_samples: Number of samples for LIME (lower = faster)
    
    Returns:
        lime_image_base64: Base64 encoded LIME visualization
        explanation_text: Text description
    """
    try:
        print("   Generating LIME explanation...")
        
        # Get original image
        img = image[0]
        
        # Create LIME explainer
        explainer = lime_image.LimeImageExplainer()
        
        # Prediction function for LIME
        def predict_fn(images):
            # LIME passes images in range [0, 1]
            return model.predict(images, verbose=0)
        
        # Generate explanation
        explanation = explainer.explain_instance(
            img,
            predict_fn,
            top_labels=len(class_names),
            hide_color=0,
            num_samples=num_samples
        )
        
        # Get the predicted class
        pred_class = np.argmax(model.predict(image, verbose=0)[0])
        
        # Get image and mask
        temp, mask = explanation.get_image_and_mask(
            pred_class,
            positive_only=True,
            num_features=10,
            hide_rest=False
        )
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(mark_boundaries(temp, mask))
        ax.set_title(f'LIME Explanation\nHighlighted: Important regions for "{class_names[pred_class]}"',
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        lime_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Generate explanation text
        explanation_text = f"LIME identified {len(mask[mask > 0])} superpixels as most influential for the '{class_names[pred_class]}' prediction."
        
        return f"data:image/png;base64,{lime_base64}", explanation_text
        
    except Exception as e:
        print(f"   LIME error: {e}")
        return None, f"LIME explanation failed: {str(e)}"


def generate_shap_explanation(model, image, class_names, background_samples=10):
    """
    Generate SHAP explanation for image classification using GradientExplainer
    
    Args:
        model: Trained Keras model
        image: Preprocessed image (1, 128, 128, 3)
        class_names: List of class names
        background_samples: Number of background samples (lower = faster)
    
    Returns:
        shap_image_base64: Base64 encoded SHAP visualization
        explanation_text: Text description
    """
    try:
        print("   Generating SHAP explanation...")
        
        # Create background dataset (random samples)
        background = np.random.rand(background_samples, 128, 128, 3).astype(np.float32)
        
        # Use GradientExplainer instead of DeepExplainer to avoid Relu6 gradient issues
        explainer = shap.GradientExplainer(model, background)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(image)
        
        # Get predicted class
        pred_class = np.argmax(model.predict(image, verbose=0)[0])
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_vals = shap_values[pred_class][0]
        else:
            shap_vals = shap_values[0]
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        ax1.imshow(image[0])
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # SHAP heatmap
        shap_sum = np.abs(shap_vals).sum(axis=-1)
        shap_normalized = (shap_sum - shap_sum.min()) / (shap_sum.max() - shap_sum.min() + 1e-8)
        
        im = ax2.imshow(shap_normalized, cmap='hot', alpha=0.8)
        ax2.imshow(image[0], alpha=0.4)
        ax2.set_title(f'SHAP Values\nPrediction: "{class_names[pred_class]}"',
                     fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        plt.colorbar(im, ax=ax2, label='Feature Importance')
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        shap_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Generate explanation text
        avg_importance = np.mean(np.abs(shap_vals))
        explanation_text = f"SHAP analysis shows average feature importance of {avg_importance:.4f} for '{class_names[pred_class]}' prediction."
        
        return f"data:image/png;base64,{shap_base64}", explanation_text
        
    except Exception as e:
        print(f"   SHAP error: {e}")
        # Fallback to a simpler gradient-based explanation
        try:
            print("   Attempting fallback gradient-based explanation...")
            pred_class = np.argmax(model.predict(image, verbose=0)[0])
            
            # Use integrated gradients as fallback
            with tf.GradientTape() as tape:
                img_tensor = tf.convert_to_tensor(image)
                tape.watch(img_tensor)
                predictions = model(img_tensor)
                target_class = predictions[:, pred_class]
            
            gradients = tape.gradient(target_class, img_tensor)
            gradients = tf.abs(gradients).numpy()[0]
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.imshow(image[0])
            ax1.set_title('Original Image', fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            grad_sum = gradients.sum(axis=-1)
            grad_normalized = (grad_sum - grad_sum.min()) / (grad_sum.max() - grad_sum.min() + 1e-8)
            
            im = ax2.imshow(grad_normalized, cmap='hot', alpha=0.8)
            ax2.imshow(image[0], alpha=0.4)
            ax2.set_title(f'Gradient-Based Attribution\nPrediction: "{class_names[pred_class]}"',
                         fontsize=12, fontweight='bold')
            ax2.axis('off')
            
            plt.colorbar(im, ax=ax2, label='Feature Importance')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            shap_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            explanation_text = f"Gradient-based attribution (SHAP fallback) for '{class_names[pred_class]}' prediction."
            return f"data:image/png;base64,{shap_base64}", explanation_text
            
        except Exception as fallback_error:
            print(f"   Fallback also failed: {fallback_error}")
            return None, f"SHAP explanation failed: {str(e)}"


def generate_all_explanations(model, image, class_names):
    """
    Generate all explainability visualizations
    
    Returns:
        dict with gradcam, lime, and shap explanations
    """
    explanations = {}
    
    # LIME (faster, ~10-15 seconds)
    lime_img, lime_text = generate_lime_explanation(model, image, class_names, num_samples=50)
    explanations['lime'] = {'image': lime_img, 'text': lime_text}
    
    # SHAP (slower, ~20-30 seconds)
    shap_img, shap_text = generate_shap_explanation(model, image, class_names, background_samples=5)
    explanations['shap'] = {'image': shap_img, 'text': shap_text}
    
    return explanations
