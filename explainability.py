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
    """
    try:
        print("   Generating SHAP explanation...")
        
        # Create background dataset
        background = np.random.rand(background_samples, 128, 128, 3).astype(np.float32)
        
        # Use GradientExplainer
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(image)
        
        # Get predicted class
        pred_class = np.argmax(model.predict(image, verbose=0)[0])
        pred_probs = model.predict(image, verbose=0)[0]
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_vals = shap_values[pred_class][0]
        else:
            shap_vals = shap_values[0]
        
        # Compute per-pixel importance (sum across channels)
        shap_sum = np.sum(np.abs(shap_vals), axis=-1)
        shap_normalized = (shap_sum - shap_sum.min()) / (shap_sum.max() - shap_sum.min() + 1e-8)

        # Positive vs negative contributions
        shap_pos = np.sum(np.maximum(shap_vals, 0), axis=-1)
        shap_neg = np.sum(np.minimum(shap_vals, 0), axis=-1)
        shap_pos = (shap_pos - shap_pos.min()) / (shap_pos.max() - shap_pos.min() + 1e-8)
        shap_neg = (shap_neg - shap_neg.min()) / (shap_neg.max() - shap_neg.min() + 1e-8)

        # Top contributing pixels
        threshold = np.percentile(shap_normalized, 80)
        top_mask = shap_normalized >= threshold

        # Create a 2x2 figure with 4 distinct panels
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.patch.set_facecolor('#0a0a1a')

        # Panel 1: Original image
        axes[0, 0].imshow(image[0])
        axes[0, 0].set_title('Original MRI Scan', fontsize=11, fontweight='bold', color='white')
        axes[0, 0].axis('off')

        # Panel 2: SHAP importance heatmap (standalone, no overlay)
        im2 = axes[0, 1].imshow(shap_normalized, cmap='hot', vmin=0, vmax=1)
        axes[0, 1].set_title('SHAP Feature Importance Map\n(Brighter = More Important)', fontsize=11, fontweight='bold', color='white')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], label='Importance Score')

        # Panel 3: Positive contributions (regions pushing TOWARD this class)
        im3 = axes[1, 0].imshow(shap_pos, cmap='Greens', vmin=0, vmax=1)
        axes[1, 0].set_title(f'Positive Contributions\n(Regions supporting "{class_names[pred_class]}")', fontsize=11, fontweight='bold', color='white')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], label='Positive SHAP Value')

        # Panel 4: Top 20% most important pixels highlighted on original
        highlighted = image[0].copy()
        overlay_mask = np.stack([top_mask * 0, top_mask * 0.8, top_mask * 0.8], axis=-1)
        highlighted = np.clip(highlighted + overlay_mask, 0, 1)
        axes[1, 1].imshow(highlighted)
        axes[1, 1].set_title('Top 20% Most Influential Pixels\n(Cyan highlight on original)', fontsize=11, fontweight='bold', color='white')
        axes[1, 1].axis('off')

        for ax in axes.flat:
            ax.set_facecolor('#0a0a1a')

        plt.suptitle(f'SHAP Analysis — Prediction: "{class_names[pred_class]}" ({pred_probs[pred_class]*100:.1f}% confidence)',
                     fontsize=13, fontweight='bold', color='#00eaff', y=1.01)
        plt.tight_layout()

        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#0a0a1a')
        buf.seek(0)
        shap_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # Stats for explanation text
        avg_importance = float(np.mean(np.abs(shap_vals)))
        top_pixel_count = int(np.sum(top_mask))
        pos_ratio = float(np.sum(shap_pos > 0.5) / (128 * 128) * 100)

        explanation_text = (
            f"SHAP identified <strong>{top_pixel_count} key pixels</strong> (top 20%) as most influential for the "
            f"<strong>'{class_names[pred_class]}'</strong> prediction. "
            f"Approximately <strong>{pos_ratio:.1f}%</strong> of the brain scan contains regions that positively support this diagnosis. "
            f"Average feature importance score: <strong>{avg_importance:.5f}</strong>. "
            f"The four panels show: (1) original scan, (2) full importance map, "
            f"(3) regions pushing toward this diagnosis, and (4) the top influential pixels highlighted directly on the scan."
        )

        return f"data:image/png;base64,{shap_base64}", explanation_text

    except Exception as e:
        print(f"   SHAP error: {e}")
        try:
            print("   Attempting fallback gradient-based explanation...")
            pred_class = np.argmax(model.predict(image, verbose=0)[0])
            pred_probs = model.predict(image, verbose=0)[0]

            with tf.GradientTape() as tape:
                img_tensor = tf.convert_to_tensor(image)
                tape.watch(img_tensor)
                predictions = model(img_tensor)
                target_class = predictions[:, pred_class]

            gradients = tape.gradient(target_class, img_tensor)
            gradients = tf.abs(gradients).numpy()[0]

            grad_sum = np.sum(gradients, axis=-1)
            grad_normalized = (grad_sum - grad_sum.min()) / (grad_sum.max() - grad_sum.min() + 1e-8)
            threshold = np.percentile(grad_normalized, 80)
            top_mask = grad_normalized >= threshold

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.patch.set_facecolor('#0a0a1a')

            axes[0].imshow(image[0])
            axes[0].set_title('Original MRI Scan', fontsize=11, fontweight='bold', color='white')
            axes[0].axis('off')

            im = axes[1].imshow(grad_normalized, cmap='hot', vmin=0, vmax=1)
            axes[1].set_title('Gradient Importance Map\n(Brighter = More Important)', fontsize=11, fontweight='bold', color='white')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], label='Importance Score')

            highlighted = image[0].copy()
            overlay_mask = np.stack([top_mask * 0, top_mask * 0.8, top_mask * 0.8], axis=-1)
            highlighted = np.clip(highlighted + overlay_mask, 0, 1)
            axes[2].imshow(highlighted)
            axes[2].set_title('Top 20% Influential Pixels\n(Cyan highlight)', fontsize=11, fontweight='bold', color='white')
            axes[2].axis('off')

            for ax in axes:
                ax.set_facecolor('#0a0a1a')

            plt.suptitle(f'Gradient Attribution — Prediction: "{class_names[pred_class]}" ({pred_probs[pred_class]*100:.1f}%)',
                         fontsize=13, fontweight='bold', color='#00eaff')
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#0a0a1a')
            buf.seek(0)
            shap_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

            explanation_text = (
                f"Gradient-based attribution (SHAP fallback) for '{class_names[pred_class]}' prediction. "
                f"Top 20% most influential pixels are highlighted in cyan on the original scan."
            )
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
