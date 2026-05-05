# =============================================================================
# EXPLAINABILITY MODULE
# =============================================================================
# This module provides three explainability methods for the Alzheimer's
# detection model:
#
#   1. LIME  - Local Interpretable Model-agnostic Explanations
#              Enhanced: shows supporting (green) AND contradicting (red) regions
#
#   2. SHAP  - SHapley Additive exPlanations
#              Uses game theory (Shapley values) to quantify each pixel's
#              contribution to the final prediction score.
#
#   3. generate_all_explanations - Convenience wrapper that runs both LIME
#              and SHAP and returns results in a single dictionary.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap
import tensorflow as tf
import cv2
import io
import base64


# =============================================================================
# LIME EXPLANATION (ENHANCED)
# =============================================================================

def generate_lime_explanation(model, image, class_names, num_samples=200):
    """
    Enhanced LIME explanation with positive/negative regions and 4-panel visualization.

    Enhancements:
    - Shows BOTH supporting (green) and contradicting (red) regions
    - More samples (200) for higher accuracy
    - Color-coded combined overlay panel
    - Numerical importance scores in explanation text
    - Brain mask restricts highlights to inside brain only
    """
    try:
        print("   Generating enhanced LIME explanation...")

        img = image[0]  # Shape: (128, 128, 3)

        # Brain mask — restrict highlights to inside brain only
        img_uint8 = np.uint8(img * 255)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        _, brain_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
        brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, kernel)
        brain_mask_bool = brain_mask > 0

        # LIME explainer
        explainer = lime_image.LimeImageExplainer()

        def predict_fn(images):
            return model.predict(images, verbose=0)

        # Run LIME with more samples for better accuracy
        explanation = explainer.explain_instance(
            img,
            predict_fn,
            top_labels=len(class_names),
            hide_color=0,
            num_samples=num_samples
        )

        pred_class = np.argmax(model.predict(image, verbose=0)[0])
        pred_probs = model.predict(image, verbose=0)[0]
        confidence = pred_probs[pred_class] * 100

        # Positive regions: support the predicted class
        temp_pos, mask_pos = explanation.get_image_and_mask(
            pred_class, positive_only=True, num_features=10, hide_rest=False
        )

        # Negative regions: argue AGAINST the predicted class
        try:
            temp_neg, mask_neg = explanation.get_image_and_mask(
                pred_class, positive_only=False, negative_only=True, num_features=5, hide_rest=False
            )
        except Exception:
            mask_neg = np.zeros_like(mask_pos)
            temp_neg = img.copy()

        # Apply brain mask to both
        mask_pos = mask_pos & brain_mask_bool
        mask_neg = mask_neg & brain_mask_bool

        # Fallback if brain masking removes everything
        if mask_pos.sum() == 0:
            temp_pos, mask_pos = explanation.get_image_and_mask(
                pred_class, positive_only=True, num_features=10, hide_rest=False
            )

        # Get numerical importance scores
        local_exp = explanation.local_exp[pred_class]
        pos_features = [(seg, w) for seg, w in local_exp if w > 0]
        neg_features = [(seg, w) for seg, w in local_exp if w < 0]
        top_pos_weight = max([w for _, w in pos_features], default=0)
        top_neg_weight = min([w for _, w in neg_features], default=0)

        # 4-panel visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.patch.set_facecolor('#0a0a1a')

        # Panel 1: Original MRI scan
        axes[0].imshow(img)
        axes[0].set_title('Original MRI Scan', fontsize=11, fontweight='bold', color='white')
        axes[0].axis('off')

        # Panel 2: Supporting regions (GREEN — push toward diagnosis)
        axes[1].imshow(mark_boundaries(temp_pos, mask_pos, color=(0, 1, 0.4), mode='thick'))
        axes[1].set_title(f'Supporting Regions\n(Green = supports "{class_names[pred_class]}")',
                          fontsize=10, fontweight='bold', color='white')
        axes[1].axis('off')

        # Panel 3: Contradicting regions (RED — argue against diagnosis)
        if mask_neg.sum() > 0:
            axes[2].imshow(mark_boundaries(temp_neg, mask_neg, color=(1, 0.2, 0.2), mode='thick'))
        else:
            axes[2].imshow(img)
            axes[2].text(64, 64, 'No significant\nnegative regions',
                        ha='center', va='center', color='white', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        axes[2].set_title('Contradicting Regions\n(Red = argues against diagnosis)',
                          fontsize=10, fontweight='bold', color='white')
        axes[2].axis('off')

        # Panel 4: Combined color-coded overlay
        combined_img = img.copy()
        if mask_pos.sum() > 0:
            green_overlay = np.zeros_like(img)
            green_overlay[mask_pos] = [0, 0.8, 0.4]
            combined_img = np.clip(combined_img + green_overlay * 0.4, 0, 1)
        if mask_neg.sum() > 0:
            red_overlay = np.zeros_like(img)
            red_overlay[mask_neg] = [0.8, 0.2, 0.2]
            combined_img = np.clip(combined_img + red_overlay * 0.4, 0, 1)
        axes[3].imshow(combined_img)
        axes[3].set_title('Combined View\n(Green=supports, Red=contradicts)',
                          fontsize=10, fontweight='bold', color='white')
        axes[3].axis('off')

        for ax in axes:
            ax.set_facecolor('#0a0a1a')

        plt.suptitle(
            f'LIME Analysis — "{class_names[pred_class]}" ({confidence:.1f}% confidence) | '
            f'Supporting: {len(pos_features)} regions | Contradicting: {len(neg_features)} regions',
            fontsize=12, fontweight='bold', color='#00eaff'
        )
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#0a0a1a')
        buf.seek(0)
        lime_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        explanation_text = (
            f"Enhanced LIME identified <strong>{len(pos_features)} supporting regions</strong> (green) and "
            f"<strong>{len(neg_features)} contradicting regions</strong> (red) for the "
            f"<strong>'{class_names[pred_class]}'</strong> prediction. "
            f"Supporting regions are brain areas that push the model toward this diagnosis. "
            f"Contradicting regions argue against it — their presence reduces confidence. "
            f"Strongest supporting signal: <strong>{top_pos_weight:.4f}</strong> | "
            f"Strongest contradicting signal: <strong>{abs(top_neg_weight):.4f}</strong>."
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
    Generate SHAP explanation using GradientExplainer.
    Shows 4 panels: original, importance map, positive contributions, top pixels.
    """
    try:
        print("   Generating SHAP explanation...")

        background = np.random.rand(background_samples, 128, 128, 3).astype(np.float32)
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(image)

        pred_class = np.argmax(model.predict(image, verbose=0)[0])
        pred_probs = model.predict(image, verbose=0)[0]

        if isinstance(shap_values, list):
            shap_vals = shap_values[pred_class][0]
        else:
            shap_vals = shap_values[0]

        shap_sum = np.sum(np.abs(shap_vals), axis=-1)
        shap_normalized = (shap_sum - shap_sum.min()) / (shap_sum.max() - shap_sum.min() + 1e-8)

        shap_pos = np.sum(np.maximum(shap_vals, 0), axis=-1)
        shap_pos = (shap_pos - shap_pos.min()) / (shap_pos.max() - shap_pos.min() + 1e-8)

        threshold = np.percentile(shap_normalized, 80)
        top_mask = shap_normalized >= threshold

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.patch.set_facecolor('#0a0a1a')

        axes[0, 0].imshow(image[0])
        axes[0, 0].set_title('Original MRI Scan', fontsize=11, fontweight='bold', color='white')
        axes[0, 0].axis('off')

        im2 = axes[0, 1].imshow(shap_normalized, cmap='hot', vmin=0, vmax=1)
        axes[0, 1].set_title('SHAP Feature Importance Map\n(Brighter = More Important)',
                              fontsize=11, fontweight='bold', color='white')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], label='Importance Score')

        im3 = axes[1, 0].imshow(shap_pos, cmap='Greens', vmin=0, vmax=1)
        axes[1, 0].set_title(f'Positive Contributions\n(Regions supporting "{class_names[pred_class]}")',
                              fontsize=11, fontweight='bold', color='white')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], label='Positive SHAP Value')

        highlighted = image[0].copy()
        overlay_mask = np.stack([top_mask * 0, top_mask * 0.8, top_mask * 0.8], axis=-1)
        highlighted = np.clip(highlighted + overlay_mask, 0, 1)
        axes[1, 1].imshow(highlighted)
        axes[1, 1].set_title('Top 20% Most Influential Pixels\n(Cyan highlight on original)',
                              fontsize=11, fontweight='bold', color='white')
        axes[1, 1].axis('off')

        for ax in axes.flat:
            ax.set_facecolor('#0a0a1a')

        plt.suptitle(
            f'SHAP Analysis — Prediction: "{class_names[pred_class]}" ({pred_probs[pred_class]*100:.1f}% confidence)',
            fontsize=13, fontweight='bold', color='#00eaff', y=1.01
        )
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#0a0a1a')
        buf.seek(0)
        shap_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

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
            axes[1].set_title('Gradient Importance Map\n(Brighter = More Important)',
                              fontsize=11, fontweight='bold', color='white')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], label='Importance Score')

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
    Runs LIME, SHAP, and Counterfactual explanations.
    Called from server.py when the user enables advanced explainability.
    """
    explanations = {}

    lime_img, lime_text = generate_lime_explanation(model, image, class_names, num_samples=100)
    explanations['lime'] = {'image': lime_img, 'text': lime_text}

    shap_img, shap_text = generate_shap_explanation(model, image, class_names, background_samples=5)
    explanations['shap'] = {'image': shap_img, 'text': shap_text}

    cf_img, cf_text, cf_data = generate_counterfactual(model, image, class_names)
    explanations['counterfactual'] = {'image': cf_img, 'text': cf_text, 'data': cf_data}

    return explanations


# =============================================================================
# COUNTERFACTUAL EXPLANATION
# =============================================================================

def generate_counterfactual(model, image, class_names, steps=200, lr=0.01):
    """
    Generate a counterfactual explanation by finding the minimal pixel-level
    change to the input image that flips the model's prediction to a different class.

    Answers: "What is the smallest change to this brain scan that would change
    the diagnosis from X to Y?"

    Method: Gradient-based iterative optimization.
    Starting from the original image, we iteratively apply small gradient steps
    that push the prediction toward the target class (the next most likely class),
    while minimizing the total change to the image (L2 regularization).

    Args:
        model:       Trained Keras model
        image:       Preprocessed image array of shape (1, 128, 128, 3)
        class_names: List of class name strings
        steps:       Number of optimization steps
        lr:          Learning rate for gradient steps

    Returns:
        cf_image_base64: Base64-encoded PNG of the 3-panel visualization
        explanation_text: HTML string describing the counterfactual
        cf_data:         Dict with numeric results (confidence changes etc.)
    """
    try:
        print("   Generating counterfactual explanation...")

        # Get original prediction
        orig_probs = model.predict(image, verbose=0)[0]
        orig_class = np.argmax(orig_probs)
        orig_confidence = float(orig_probs[orig_class])

        # Target class = second most likely class
        # This is the "nearest neighbor" in prediction space
        sorted_classes = np.argsort(orig_probs)[::-1]
        target_class = int(sorted_classes[1])
        target_confidence_start = float(orig_probs[target_class])

        print(f"   Original: {class_names[orig_class]} ({orig_confidence*100:.1f}%)")
        print(f"   Target  : {class_names[target_class]} ({target_confidence_start*100:.1f}%)")

        # Start from the original image
        cf_image = tf.Variable(image.copy(), dtype=tf.float32)
        original_image = tf.constant(image, dtype=tf.float32)

        # Optimization loop — iteratively nudge the image toward the target class
        flipped = False
        flip_step = None

        for step in range(steps):
            with tf.GradientTape() as tape:
                tape.watch(cf_image)
                preds = model(cf_image)

                # Loss = maximize target class probability
                #      + minimize change from original (L2 regularization)
                target_loss = -preds[0, target_class]
                l2_loss     = 0.1 * tf.reduce_sum(tf.square(cf_image - original_image))
                total_loss  = target_loss + l2_loss

            # Compute gradient of loss w.r.t. the counterfactual image
            grads = tape.gradient(total_loss, cf_image)

            # Apply gradient step (gradient descent)
            cf_image.assign_sub(lr * grads)

            # Clip to valid image range [0, 1]
            cf_image.assign(tf.clip_by_value(cf_image, 0.0, 1.0))

            # Check if prediction has flipped
            current_preds = model.predict(cf_image.numpy(), verbose=0)[0]
            current_class = np.argmax(current_preds)

            if current_class == target_class and not flipped:
                flipped = True
                flip_step = step + 1
                print(f"   ✅ Prediction flipped at step {flip_step}")
                break

        # Get final counterfactual image and predictions
        cf_array = cf_image.numpy()[0]
        final_probs = model.predict(cf_image.numpy(), verbose=0)[0]
        final_class = np.argmax(final_probs)
        final_confidence = float(final_probs[final_class])

        # Compute the difference map (what changed)
        diff = np.abs(cf_array - image[0])
        diff_sum = np.sum(diff, axis=-1)  # Sum across RGB channels
        diff_normalized = (diff_sum - diff_sum.min()) / (diff_sum.max() - diff_sum.min() + 1e-8)

        # Percentage of pixels that changed significantly (>1% change)
        changed_pixels = int(np.sum(diff_sum > 0.03))
        total_pixels = 128 * 128
        change_pct = changed_pixels / total_pixels * 100

        # =================================================================
        # VISUALIZATION — 3 panels
        # =================================================================

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor('#0a0a1a')

        # Panel 1: Original image with original prediction
        axes[0].imshow(image[0])
        axes[0].set_title(
            f'Original Scan\n"{class_names[orig_class]}" ({orig_confidence*100:.1f}%)',
            fontsize=11, fontweight='bold', color='white'
        )
        axes[0].axis('off')

        # Panel 2: Difference map — what changed
        im = axes[1].imshow(diff_normalized, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title(
            f'Change Map\n(Brighter = More Changed)',
            fontsize=11, fontweight='bold', color='white'
        )
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], label='Change Magnitude')

        # Panel 3: Counterfactual image with new prediction
        axes[2].imshow(np.clip(cf_array, 0, 1))
        axes[2].set_title(
            f'Counterfactual Scan\n"{class_names[final_class]}" ({final_confidence*100:.1f}%)',
            fontsize=11, fontweight='bold', color='white'
        )
        axes[2].axis('off')

        for ax in axes:
            ax.set_facecolor('#0a0a1a')

        status = f"Flipped in {flip_step} steps" if flipped else f"Not flipped ({steps} steps)"
        plt.suptitle(
            f'Counterfactual: "{class_names[orig_class]}" → "{class_names[final_class]}" | {status}',
            fontsize=12, fontweight='bold', color='#00eaff'
        )
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#0a0a1a')
        buf.seek(0)
        cf_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # Build explanation text
        if flipped:
            explanation_text = (
                f"The model's prediction was changed from <strong>'{class_names[orig_class]}'</strong> "
                f"({orig_confidence*100:.1f}% confidence) to <strong>'{class_names[final_class]}'</strong> "
                f"({final_confidence*100:.1f}% confidence) by modifying only "
                f"<strong>{change_pct:.1f}% of pixels</strong> ({changed_pixels} out of {total_pixels}). "
                f"This was achieved in <strong>{flip_step} optimization steps</strong>. "
                f"The change map (middle panel) shows which brain regions needed to be altered — "
                f"brighter areas required more modification, indicating these are the most critical "
                f"regions distinguishing the two diagnoses."
            )
        else:
            explanation_text = (
                f"After {steps} optimization steps, the prediction remained "
                f"<strong>'{class_names[orig_class]}'</strong> ({orig_confidence*100:.1f}% confidence). "
                f"The model moved toward <strong>'{class_names[final_class]}'</strong> "
                f"({final_confidence*100:.1f}%) but did not fully flip. "
                f"This indicates a <strong>high-confidence prediction</strong> — the model is very certain "
                f"about this diagnosis and requires substantial brain structure changes to reconsider it. "
                f"Approximately <strong>{change_pct:.1f}% of pixels</strong> were modified in the attempt."
            )

        cf_data = {
            'original_class': class_names[orig_class],
            'original_confidence': f"{orig_confidence*100:.1f}%",
            'target_class': class_names[target_class],
            'final_class': class_names[final_class],
            'final_confidence': f"{final_confidence*100:.1f}%",
            'flipped': flipped,
            'flip_step': flip_step,
            'changed_pixels_pct': f"{change_pct:.1f}%"
        }

        return f"data:image/png;base64,{cf_base64}", explanation_text, cf_data

    except Exception as e:
        print(f"   Counterfactual error: {e}")
        return None, f"Counterfactual explanation failed: {str(e)}", {}
