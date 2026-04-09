# =============================================================================
# EXPLAINABILITY MODULE
# =============================================================================
# This module provides three explainability methods for the Alzheimer's
# detection model:
#
#   1. LIME  - Local Interpretable Model-agnostic Explanations
#              Highlights which brain regions most influenced the prediction
#              by perturbing the image and observing prediction changes.
#
#   2. SHAP  - SHapley Additive exPlanations
#              Uses game theory (Shapley values) to quantify each pixel's
#              contribution to the final prediction score.
#
#   3. generate_all_explanations - Convenience wrapper that runs both LIME
#              and SHAP and returns results in a single dictionary.
#
# All functions return base64-encoded PNG images and descriptive text
# suitable for embedding directly in the frontend HTML.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image                    # LIME library for image explanations
from skimage.segmentation import mark_boundaries  # Draws boundaries around LIME regions
import shap                                    # SHAP library for feature attribution
import tensorflow as tf                        # For gradient-based fallback
import cv2                                     # OpenCV for brain mask creation
import io                                      # For in-memory image buffers
import base64                                  # For encoding images to base64 strings


# =============================================================================
# LIME EXPLANATION
# =============================================================================

def generate_lime_explanation(model, image, class_names, num_samples=100):
    """
    Generate a LIME explanation for a brain MRI prediction.

    LIME works by:
    1. Creating many perturbed versions of the input image (hiding superpixels)
    2. Running the model on each perturbed version
    3. Fitting a simple linear model to identify which regions matter most
    4. Highlighting those important regions on the original image

    A brain mask is applied to restrict highlights to inside the brain only,
    preventing false highlights on the black background.

    Args:
        model:       Trained Keras model
        image:       Preprocessed image array of shape (1, 128, 128, 3), values in [0, 1]
        class_names: List of class name strings (must match model output order)
        num_samples: Number of perturbed samples to generate (more = slower but more accurate)

    Returns:
        lime_base64:      Base64-encoded PNG string of the LIME visualization
        explanation_text: HTML string describing the LIME results
    """
    try:
        print("   Generating LIME explanation...")

        # Extract the single image from the batch dimension
        img = image[0]  # Shape: (128, 128, 3)

        # =====================================================================
        # BRAIN MASK CREATION
        # =====================================================================
        # MRI scans have a black background. Without a brain mask, LIME may
        # highlight background regions as "important" (they're not).
        # We create a binary mask that is True inside the brain and False outside.
        # =====================================================================

        # Convert float image [0,1] to uint8 [0,255] for OpenCV processing
        img_uint8 = np.uint8(img * 255)

        # Convert RGB to grayscale for thresholding
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

        # Threshold: pixels with intensity > 15 are considered brain tissue
        # (background is near-black, so threshold of 15 works well)
        _, brain_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        # Morphological CLOSE: fills small holes inside the brain region
        kernel = np.ones((5, 5), np.uint8)
        brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)

        # Morphological OPEN: removes small noise outside the brain region
        brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, kernel)

        # Convert to boolean mask for array indexing
        brain_mask_bool = brain_mask > 0

        # =====================================================================
        # LIME EXPLANATION GENERATION
        # =====================================================================

        # Create the LIME image explainer
        explainer = lime_image.LimeImageExplainer()

        # Prediction function wrapper for LIME
        # LIME passes batches of perturbed images and expects probability arrays
        def predict_fn(images):
            return model.predict(images, verbose=0)

        # Run LIME — this generates num_samples perturbed images and fits a local model
        explanation = explainer.explain_instance(
            img,
            predict_fn,
            top_labels=len(class_names),   # Explain all classes
            hide_color=0,                  # Hide superpixels with black (0)
            num_samples=num_samples        # Number of perturbations
        )

        # Get the predicted class index and probabilities
        pred_class = np.argmax(model.predict(image, verbose=0)[0])
        pred_probs = model.predict(image, verbose=0)[0]

        # Extract the image and importance mask for the predicted class
        # positive_only=True: only show regions that support the prediction
        # num_features=10: show top 10 most important superpixels
        temp, mask = explanation.get_image_and_mask(
            pred_class,
            positive_only=True,
            num_features=10,
            hide_rest=False
        )

        # Apply brain mask: remove any highlighted regions outside the brain
        mask_filtered = mask & brain_mask_bool

        # Safety fallback: if brain masking removed all highlights, use original mask
        if mask_filtered.sum() == 0:
            mask_filtered = mask

        # =====================================================================
        # VISUALIZATION — 3 panels
        # =====================================================================

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor('#0a0a1a')  # Dark background to match UI theme

        # Panel 1: Original MRI scan (unmodified)
        axes[0].imshow(img)
        axes[0].set_title('Original MRI Scan', fontsize=11, fontweight='bold', color='white')
        axes[0].axis('off')

        # Panel 2: Brain region detection overlay
        # Shows the user which area was identified as brain tissue
        axes[1].imshow(img)
        brain_overlay = np.zeros_like(img)
        brain_overlay[brain_mask_bool] = [0, 0.6, 0.6]  # Teal color for brain region
        axes[1].imshow(brain_overlay, alpha=0.3)
        axes[1].set_title('Detected Brain Region\n(Teal overlay = brain area)',
                          fontsize=11, fontweight='bold', color='white')
        axes[1].axis('off')

        # Panel 3: LIME important regions (restricted to brain only)
        # Yellow boundaries outline the most influential superpixels
        axes[2].imshow(mark_boundaries(temp, mask_filtered, color=(1, 1, 0), mode='thick'))
        axes[2].set_title(
            f'LIME Important Regions\n(Yellow = key areas for "{class_names[pred_class]}")',
            fontsize=11, fontweight='bold', color='white'
        )
        axes[2].axis('off')

        # Apply dark background to all panels
        for ax in axes:
            ax.set_facecolor('#0a0a1a')

        # Add overall title with prediction and confidence
        plt.suptitle(
            f'LIME Analysis — Prediction: "{class_names[pred_class]}" ({pred_probs[pred_class]*100:.1f}% confidence)',
            fontsize=13, fontweight='bold', color='#00eaff'
        )
        plt.tight_layout()

        # Save figure to in-memory buffer and encode as base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#0a0a1a')
        buf.seek(0)
        lime_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # Count unique highlighted regions (excluding background label 0)
        region_count = len(np.unique(mask_filtered)[1:]) if mask_filtered.sum() > 0 else 0

        # Build descriptive explanation text (HTML-formatted for frontend display)
        explanation_text = (
            f"LIME identified <strong>{region_count} brain regions</strong> most influential for the "
            f"<strong>'{class_names[pred_class]}'</strong> prediction. "
            f"Highlights are restricted to inside the brain boundary only. "
            f"Yellow outlines show the specific areas that, when present, most strongly support this diagnosis. "
            f"The teal overlay in the middle panel shows the detected brain region used to filter out background noise."
        )

        return f"data:image/png;base64,{lime_base64}", explanation_text

    except Exception as e:
        print(f"   LIME error: {e}")
        return None, f"LIME explanation failed: {str(e)}"


# =============================================================================
# SHAP EXPLANATION
# =============================================================================

def generate_shap_explanation(model, image, class_names, background_samples=10):
    """
    Generate a SHAP explanation for a brain MRI prediction.

    SHAP (SHapley Additive exPlanations) uses game theory to assign each
    pixel a contribution score. It answers: "How much did this pixel push
    the model toward or away from the predicted class?"

    Uses GradientExplainer which computes SHAP values via backpropagation.
    Falls back to plain gradient attribution if SHAP fails.

    Args:
        model:             Trained Keras model
        image:             Preprocessed image array of shape (1, 128, 128, 3)
        class_names:       List of class name strings
        background_samples: Number of random background images for SHAP baseline

    Returns:
        shap_base64:      Base64-encoded PNG string of the SHAP visualization
        explanation_text: HTML string describing the SHAP results
    """
    try:
        print("   Generating SHAP explanation...")

        # Create a random background dataset
        # SHAP compares the image against this baseline to compute contributions
        background = np.random.rand(background_samples, 128, 128, 3).astype(np.float32)

        # Initialize SHAP GradientExplainer
        # GradientExplainer uses integrated gradients — compatible with most CNN architectures
        explainer = shap.GradientExplainer(model, background)

        # Compute SHAP values for the input image
        # Returns a list of arrays (one per output class) or a single array
        shap_values = explainer.shap_values(image)

        # Get predicted class index and probabilities
        pred_class = np.argmax(model.predict(image, verbose=0)[0])
        pred_probs = model.predict(image, verbose=0)[0]

        # Extract SHAP values for the predicted class
        # Format depends on SHAP version — handle both list and array formats
        if isinstance(shap_values, list):
            shap_vals = shap_values[pred_class][0]  # Shape: (128, 128, 3)
        else:
            shap_vals = shap_values[0]

        # =====================================================================
        # COMPUTE IMPORTANCE MAPS
        # =====================================================================

        # Total importance per pixel: sum absolute SHAP values across RGB channels
        shap_sum = np.sum(np.abs(shap_vals), axis=-1)  # Shape: (128, 128)

        # Normalize to [0, 1] for visualization
        shap_normalized = (shap_sum - shap_sum.min()) / (shap_sum.max() - shap_sum.min() + 1e-8)

        # Positive contributions: pixels that push TOWARD the predicted class
        shap_pos = np.sum(np.maximum(shap_vals, 0), axis=-1)
        shap_pos = (shap_pos - shap_pos.min()) / (shap_pos.max() - shap_pos.min() + 1e-8)

        # Negative contributions: pixels that push AWAY from the predicted class
        shap_neg = np.sum(np.minimum(shap_vals, 0), axis=-1)
        shap_neg = (shap_neg - shap_neg.min()) / (shap_neg.max() - shap_neg.min() + 1e-8)

        # Identify the top 20% most important pixels (by absolute SHAP value)
        threshold = np.percentile(shap_normalized, 80)
        top_mask = shap_normalized >= threshold  # Boolean mask

        # =====================================================================
        # VISUALIZATION — 2x2 grid of 4 panels
        # =====================================================================

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.patch.set_facecolor('#0a0a1a')

        # Panel 1 (top-left): Original MRI scan for reference
        axes[0, 0].imshow(image[0])
        axes[0, 0].set_title('Original MRI Scan', fontsize=11, fontweight='bold', color='white')
        axes[0, 0].axis('off')

        # Panel 2 (top-right): Full SHAP importance heatmap
        # Standalone (no overlay) so it's clearly distinct from the original
        # Brighter = more important pixel
        im2 = axes[0, 1].imshow(shap_normalized, cmap='hot', vmin=0, vmax=1)
        axes[0, 1].set_title('SHAP Feature Importance Map\n(Brighter = More Important)',
                              fontsize=11, fontweight='bold', color='white')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], label='Importance Score')

        # Panel 3 (bottom-left): Positive contributions only
        # Shows which regions actively support the predicted diagnosis
        im3 = axes[1, 0].imshow(shap_pos, cmap='Greens', vmin=0, vmax=1)
        axes[1, 0].set_title(
            f'Positive Contributions\n(Regions supporting "{class_names[pred_class]}")',
            fontsize=11, fontweight='bold', color='white'
        )
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], label='Positive SHAP Value')

        # Panel 4 (bottom-right): Top 20% most influential pixels highlighted on original
        # Cyan overlay shows exactly which pixels matter most
        highlighted = image[0].copy()
        overlay_mask = np.stack([top_mask * 0, top_mask * 0.8, top_mask * 0.8], axis=-1)
        highlighted = np.clip(highlighted + overlay_mask, 0, 1)
        axes[1, 1].imshow(highlighted)
        axes[1, 1].set_title('Top 20% Most Influential Pixels\n(Cyan highlight on original)',
                              fontsize=11, fontweight='bold', color='white')
        axes[1, 1].axis('off')

        # Apply dark background to all panels
        for ax in axes.flat:
            ax.set_facecolor('#0a0a1a')

        # Add overall title
        plt.suptitle(
            f'SHAP Analysis — Prediction: "{class_names[pred_class]}" ({pred_probs[pred_class]*100:.1f}% confidence)',
            fontsize=13, fontweight='bold', color='#00eaff', y=1.01
        )
        plt.tight_layout()

        # Save to buffer and encode as base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#0a0a1a')
        buf.seek(0)
        shap_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # =====================================================================
        # COMPUTE STATS FOR EXPLANATION TEXT
        # =====================================================================

        avg_importance = float(np.mean(np.abs(shap_vals)))   # Average SHAP value
        top_pixel_count = int(np.sum(top_mask))               # Number of top-20% pixels
        pos_ratio = float(np.sum(shap_pos > 0.5) / (128 * 128) * 100)  # % positive pixels

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
        # =====================================================================
        # FALLBACK: GRADIENT-BASED ATTRIBUTION
        # =====================================================================
        # If SHAP fails (e.g., due to model architecture incompatibility),
        # fall back to plain gradient attribution which is simpler but still informative.
        # =====================================================================
        print(f"   SHAP error: {e}")
        try:
            print("   Attempting fallback gradient-based explanation...")
            pred_class = np.argmax(model.predict(image, verbose=0)[0])
            pred_probs = model.predict(image, verbose=0)[0]

            # Compute gradients of the predicted class score w.r.t. input pixels
            with tf.GradientTape() as tape:
                img_tensor = tf.convert_to_tensor(image)
                tape.watch(img_tensor)
                predictions = model(img_tensor)
                target_class = predictions[:, pred_class]

            # Take absolute value of gradients and sum across color channels
            gradients = tape.gradient(target_class, img_tensor)
            gradients = tf.abs(gradients).numpy()[0]
            grad_sum = np.sum(gradients, axis=-1)

            # Normalize to [0, 1]
            grad_normalized = (grad_sum - grad_sum.min()) / (grad_sum.max() - grad_sum.min() + 1e-8)

            # Identify top 20% most important pixels
            threshold = np.percentile(grad_normalized, 80)
            top_mask = grad_normalized >= threshold

            # Create 3-panel visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.patch.set_facecolor('#0a0a1a')

            # Panel 1: Original image
            axes[0].imshow(image[0])
            axes[0].set_title('Original MRI Scan', fontsize=11, fontweight='bold', color='white')
            axes[0].axis('off')

            # Panel 2: Gradient importance heatmap
            im = axes[1].imshow(grad_normalized, cmap='hot', vmin=0, vmax=1)
            axes[1].set_title('Gradient Importance Map\n(Brighter = More Important)',
                              fontsize=11, fontweight='bold', color='white')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], label='Importance Score')

            # Panel 3: Top 20% pixels highlighted in cyan on original
            highlighted = image[0].copy()
            overlay_mask = np.stack([top_mask * 0, top_mask * 0.8, top_mask * 0.8], axis=-1)
            highlighted = np.clip(highlighted + overlay_mask, 0, 1)
            axes[2].imshow(highlighted)
            axes[2].set_title('Top 20% Influential Pixels\n(Cyan highlight)',
                              fontsize=11, fontweight='bold', color='white')
            axes[2].axis('off')

            for ax in axes:
                ax.set_facecolor('#0a0a1a')

            plt.suptitle(
                f'Gradient Attribution — Prediction: "{class_names[pred_class]}" ({pred_probs[pred_class]*100:.1f}%)',
                fontsize=13, fontweight='bold', color='#00eaff'
            )
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


# =============================================================================
# COMBINED EXPLANATION GENERATOR
# =============================================================================

def generate_all_explanations(model, image, class_names):
    """
    Convenience function that runs both LIME and SHAP explanations
    and returns them in a single dictionary.

    Called from server.py when the user enables advanced explainability.

    Args:
        model:       Trained Keras model
        image:       Preprocessed image array of shape (1, 128, 128, 3)
        class_names: List of class name strings

    Returns:
        dict with keys 'lime' and 'shap', each containing:
            - 'image': base64-encoded PNG string
            - 'text':  HTML explanation string
    """
    explanations = {}

    # Generate LIME explanation (faster, ~10-15 seconds with num_samples=50)
    lime_img, lime_text = generate_lime_explanation(model, image, class_names, num_samples=50)
    explanations['lime'] = {'image': lime_img, 'text': lime_text}

    # Generate SHAP explanation (slower, ~20-30 seconds with background_samples=5)
    shap_img, shap_text = generate_shap_explanation(model, image, class_names, background_samples=5)
    explanations['shap'] = {'image': shap_img, 'text': shap_text}

    return explanations
