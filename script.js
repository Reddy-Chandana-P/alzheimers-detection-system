// =============================================================================
// ALZHEIMER'S DISEASE DETECTION - FRONTEND JAVASCRIPT
// =============================================================================
// This file handles all client-side logic for the Alzheimer's Detection System:
//   - Form validation (required fields, age range, phone format)
//   - Image upload preview and validation (type, size, count)
//   - Sending prediction requests to the Flask backend
//   - Rendering prediction results with Grad-CAM and clinical insights
//   - Displaying LIME and SHAP advanced explainability results
//   - Prediction history modal (view, delete entries)
// =============================================================================

console.log("Script loaded successfully!");

// --- DOM Element References ---
const imageInput = document.getElementById("imageInput");   // File input element
const preview = document.getElementById("preview");         // Image preview container
const form = document.getElementById("uploadForm");         // Main form element
const resultBox = document.getElementById("result");        // Results display container

// =============================================================================
// FORM VALIDATION — REAL-TIME FEEDBACK
// =============================================================================

// Add blur/input listeners to all required fields for real-time validation feedback
const inputs = form.querySelectorAll('input[required], select[required]');
inputs.forEach(input => {
    // On blur (when user leaves the field): highlight invalid fields in red
    input.addEventListener('blur', () => {
        if (!input.validity.valid) {
            input.style.borderColor = '#ff4444';
        }
    });

    // On input (while user is typing): restore normal border when valid
    input.addEventListener('input', () => {
        if (input.validity.valid) {
            input.style.borderColor = 'rgba(0, 234, 255, 0.3)';
        }
    });
});

// --- Phone Number Formatting ---
// Strip non-numeric characters and limit to 15 digits
const contactInput = document.getElementById('contactNumber');
if (contactInput) {
    contactInput.addEventListener('input', (e) => {
        let value = e.target.value.replace(/\D/g, '');  // Remove non-digits
        if (value.length > 15) value = value.slice(0, 15);
        e.target.value = value;
    });
}

// --- Age Range Validation ---
// Clamp age input to valid range [1, 120]
const ageInput = document.getElementById('age');
if (ageInput) {
    ageInput.addEventListener('input', (e) => {
        const value = parseInt(e.target.value);
        if (value < 1) e.target.value = 1;
        if (value > 120) e.target.value = 120;
    });
}

// --- Character Counter for Notes Field ---
// Shows remaining characters as user types (max 500)
const notesInput = document.getElementById('notes');
if (notesInput) {
    const counter = document.createElement('div');
    counter.style.cssText = 'text-align: right; font-size: 0.85rem; color: #00eaff; margin-top: -8px; margin-bottom: 10px;';
    notesInput.parentNode.appendChild(counter);

    notesInput.addEventListener('input', () => {
        const remaining = 500 - notesInput.value.length;
        counter.textContent = `${remaining} characters remaining`;
        // Turn red when fewer than 50 characters remain
        counter.style.color = remaining < 50 ? '#ff4444' : '#00eaff';
    });
}

// =============================================================================
// IMAGE UPLOAD PREVIEW
// =============================================================================

// When user selects image files, validate and show previews
imageInput.addEventListener("change", () => {
    preview.innerHTML = "";  // Clear previous previews
    const files = [...imageInput.files];

    if (files.length === 0) return;

    // Limit to 5 images maximum
    if (files.length > 5) {
        alert("Maximum 5 images allowed");
        imageInput.value = "";
        return;
    }

    // Validate each file and create preview thumbnails
    files.forEach(file => {
        // Check file size (max 10MB per image)
        if (file.size > 10 * 1024 * 1024) {
            alert(`File ${file.name} is too large. Maximum size is 10MB.`);
            return;
        }

        // Check file type (must be an image)
        if (!file.type.startsWith('image/')) {
            alert(`File ${file.name} is not a valid image.`);
            return;
        }

        // Create and append preview thumbnail
        const img = document.createElement("img");
        img.src = URL.createObjectURL(file);  // Create temporary URL for preview
        img.title = file.name;
        preview.appendChild(img);
    });
});

// =============================================================================
// FORM SUBMISSION — PREDICTION REQUEST
// =============================================================================

form.onsubmit = async function(e) {
    e.preventDefault();       // Prevent default form submission (page reload)
    e.stopPropagation();      // Stop event from bubbling up

    console.log("Form submission started");

    // --- Validate all required fields before sending ---
    const requiredFields = form.querySelectorAll('[required]');
    let isValid = true;

    requiredFields.forEach(field => {
        if (!field.validity.valid) {
            field.style.borderColor = '#ff4444';  // Highlight invalid fields
            isValid = false;
        }
    });

    if (!isValid) {
        alert("Please fill in all required fields correctly.");
        return false;
    }

    // --- Build FormData with images and patient info ---
    const formData = new FormData();
    const imageFiles = imageInput.files;

    if (imageFiles.length === 0) {
        alert("Please select at least one image!");
        return false;
    }

    // Append all selected image files (supports multiple)
    for (let i = 0; i < imageFiles.length; i++) {
        formData.append("image", imageFiles[i]);
    }

    // Append advanced explainability flag (LIME + SHAP)
    const advancedEnabled = document.getElementById("advancedExplain").checked;
    if (advancedEnabled) {
        formData.append("advanced", "true");
    }

    // Append all patient information fields
    formData.append("patientName", document.getElementById("patientName").value);
    formData.append("patientId", document.getElementById("patientId").value);
    formData.append("age", document.getElementById("age").value);
    formData.append("gender", document.getElementById("gender").value);
    formData.append("contactNumber", document.getElementById("contactNumber").value);
    formData.append("email", document.getElementById("email").value);
    formData.append("medicalHistory", document.getElementById("medicalHistory").value);
    formData.append("symptoms", document.getElementById("symptoms").value);
    formData.append("duration", document.getElementById("duration").value);
    formData.append("notes", document.getElementById("notes").value);

    // --- Show loading state on submit button ---
    const button = form.querySelector("button[type='submit']");
    const originalText = button.innerHTML;
    button.innerHTML = `⏳ Analyzing ${imageFiles.length} image(s)...`;
    button.disabled = true;

    try {
        console.log("Sending request to server...");

        // Send POST request to Flask backend
        const response = await fetch("http://localhost:5001/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Prediction failed");
        }

        // Parse JSON response from server
        const data = await response.json();
        console.log("Data received:", data);

        // --- Render results for each image ---
        const multipleResults = document.getElementById("multipleResults");
        multipleResults.innerHTML = '';  // Clear previous results

        data.results.forEach((result, index) => {
            // Build HTML for each result and append to container
            multipleResults.innerHTML += createResultHTML(result, index, data.results.length > 1);
        });

        console.log("Results added to page");

        // Show the results section (was hidden by default)
        resultBox.classList.remove("hidden");
        resultBox.style.display = "block";

        console.log("Result box displayed");

        // Smooth scroll to results after a short delay
        setTimeout(() => {
            resultBox.scrollIntoView({ behavior: "smooth" });
        }, 200);

    } catch (error) {
        console.error("Error:", error);
        alert("Error: " + error.message + "\n\nMake sure the Flask server is running!");
    } finally {
        // Always restore button state, even if an error occurred
        button.innerHTML = originalText;
        button.disabled = false;
    }

    return false;
};

// =============================================================================
// RESULT HTML BUILDER
// =============================================================================

/**
 * Builds the complete HTML string for a single image's prediction result.
 * Includes: diagnosis, confidence bars, Grad-CAM, clinical insights,
 * and optionally LIME + SHAP advanced explanations.
 *
 * @param {Object}  result     - Prediction result object from the server
 * @param {number}  index      - Index of this result (0-based)
 * @param {boolean} isMultiple - Whether multiple images were uploaded
 * @returns {string} HTML string
 */
function createResultHTML(result, index, isMultiple) {

    // --- Clinical insight data for each diagnosis class ---
    // Each entry contains color coding, severity label, and detailed medical text
    const insights = {
        'No Impairment': {
            color: '#00ff88',       // Green — normal/healthy
            icon: '✅',
            severity: 'Normal',
            clinical: 'The AI model indicates no significant signs of cognitive impairment in the analyzed brain scan. The brain structure appears within normal parameters for the patient\'s age group.',
            indicators: 'Normal hippocampal volume, preserved cortical thickness, no significant ventricular enlargement, intact white matter integrity.',
            brainRegions: 'The model found no abnormal atrophy in the hippocampus, entorhinal cortex, or temporal lobes — regions typically affected in Alzheimer\'s disease.',
            recommendation: 'Continue routine monitoring with annual cognitive assessments. Maintain healthy lifestyle habits including regular exercise, balanced diet, and mental stimulation. Consider baseline cognitive testing for future comparison.',
            nextSteps: 'Schedule follow-up in 12 months. Maintain healthy lifestyle. No immediate intervention required.'
        },
        'Very Mild Impairment': {
            color: '#ffdd00',       // Yellow — early warning
            icon: '⚠️',
            severity: 'Early Stage',
            clinical: 'The model detected subtle structural changes that may indicate very early-stage cognitive decline. These changes are often associated with the preclinical phase of Alzheimer\'s disease.',
            indicators: 'Minimal hippocampal atrophy, slight ventricular enlargement, early white matter changes, subtle cortical thinning in temporal regions.',
            brainRegions: 'Early changes detected in the hippocampus and entorhinal cortex — the first regions typically affected by Alzheimer\'s pathology. These areas are critical for memory formation and spatial navigation.',
            recommendation: 'Conduct comprehensive neuropsychological testing including memory, attention, and executive function assessments. Monitor progression with follow-up MRI scans in 6-12 months.',
            nextSteps: 'Neuropsychological evaluation within 1 month. Follow-up MRI in 6 months. Consider lifestyle modifications and cognitive training programs.'
        },
        'Mild Impairment': {
            color: '#ff8800',       // Orange — moderate concern
            icon: '🔶',
            severity: 'Moderate Stage',
            clinical: 'The model identified moderate structural changes consistent with mild cognitive impairment (MCI). This stage often represents a transitional phase between normal aging and Alzheimer\'s dementia.',
            indicators: 'Noticeable hippocampal atrophy, cortical thinning in temporal and parietal regions, moderate ventricular enlargement, reduced glucose metabolism patterns.',
            brainRegions: 'Significant atrophy detected in the hippocampus, temporal lobes, and parietal cortex. These regions are responsible for memory consolidation, language processing, and spatial awareness — functions commonly impaired in Alzheimer\'s disease.',
            recommendation: 'Comprehensive clinical evaluation is strongly recommended. Consider biomarker testing (CSF analysis, PET imaging), detailed cognitive assessment battery, and genetic counseling if family history is present.',
            nextSteps: 'Urgent referral to neurologist within 2 weeks. Comprehensive cognitive battery. Discuss treatment options including cholinesterase inhibitors. Family counseling recommended.'
        },
        'Moderate Impairment': {
            color: '#ff4444',       // Red — serious/urgent
            icon: '🔴',
            severity: 'Advanced Stage',
            clinical: 'The model detected significant structural changes indicating moderate-to-severe cognitive impairment consistent with advanced Alzheimer\'s disease. Widespread neurodegeneration is evident across multiple brain regions.',
            indicators: 'Substantial hippocampal and cortical atrophy, pronounced ventricular enlargement, widespread brain volume loss, significant white matter lesions.',
            brainRegions: 'Extensive atrophy across hippocampus, temporal lobes, parietal cortex, and frontal regions. The widespread neurodegeneration affects memory, language, reasoning, and behavioral regulation — hallmarks of advanced Alzheimer\'s disease.',
            recommendation: 'Immediate comprehensive evaluation by a specialist team. Discuss treatment options, care planning, and support services with patient and family. Consider safety assessments for daily living activities.',
            nextSteps: 'Immediate specialist referral. Comprehensive care plan development. Safety assessment for daily activities. Family support and counseling. Discuss legal and financial planning.'
        }
    };

    // Show image filename as header when multiple images are uploaded
    const imageTitle = isMultiple
        ? `<h3 style="color: #00eaff; margin-bottom: 15px;">Image ${index + 1}: ${result.filename}</h3>`
        : '';

    // Look up clinical insight for the predicted class
    const insight = insights[result.prediction] || null;

    // Use the class-specific color, or default cyan if class not found
    const severityColor = insight ? insight.color : '#00eaff';

    // --- Build the result HTML ---
    let html = `
        <div class="result-item" style="margin-bottom: 40px; padding: 25px; background: rgba(0, 234, 255, 0.05); border-radius: 15px; border: 1px solid rgba(0, 234, 255, 0.2);">
            ${imageTitle}

            <!-- PRIMARY DIAGNOSIS BOX -->
            <div class="result-summary">
                <div class="prediction-box">
                    <h3>Primary Diagnosis</h3>
                    <p class="prediction-label"><strong>Status:</strong>
                        <span style="color: ${severityColor}; font-weight: bold; font-size: 1.3rem;">
                            ${insight ? insight.icon : ''} ${result.prediction}
                        </span>
                    </p>
                    ${insight ? `<p style="color: ${severityColor}; font-size: 0.95rem; margin: 5px 0;">Severity Level: <strong>${insight.severity}</strong></p>` : ''}
                    <p class="confidence-label"><strong>Confidence:</strong>
                        <span style="color: #00eaff; font-weight: bold; font-size: 1.3rem;">${result.confidence}</span>
                    </p>
                    <!-- Confidence progress bar — width is set to the confidence percentage -->
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${result.confidence}; background: ${severityColor};">${result.confidence}</div>
                    </div>
                </div>
            </div>

            <!-- CONFIDENCE DISTRIBUTION ACROSS ALL CLASSES -->
            <div class="explainability-section">
                <h3>📊 Confidence Distribution Across All Classes</h3>
                <p class="explanation-text">The model evaluated all four possible diagnoses simultaneously. The percentages below show how strongly the AI associates this scan with each Alzheimer's stage:</p>
                <div class="predictions-grid">
    `;

    // Render a bar for each class showing its probability
    for (const [className, confidence] of Object.entries(result.all_predictions)) {
        const isHighest = className === result.prediction;  // Highlight the predicted class
        const classInsight = insights[className];
        html += `
            <div class="prediction-item" style="${isHighest ? `border: 2px solid ${severityColor}; background: rgba(0, 234, 255, 0.15);` : ''}">
                <div class="prediction-item-label">${classInsight ? classInsight.icon : ''} ${className}</div>
                <div class="prediction-item-value">${confidence}</div>
                <div class="prediction-item-bar">
                    <div class="prediction-item-fill" style="width: ${confidence}; background: ${classInsight ? classInsight.color : '#00eaff'};"></div>
                </div>
            </div>
        `;
    }

    html += `
                </div>
            </div>

            <!-- GRAD-CAM VISUALIZATION -->
            <div class="explainability-section">
                <h3>🧠 Visual Explanation (Grad-CAM)</h3>
                <p class="explanation-text">
                    <strong>What is Grad-CAM?</strong> Gradient-weighted Class Activation Mapping (Grad-CAM) is a technique that produces a visual explanation for the AI's decision.
                    It uses the gradients flowing into the final convolutional layer to highlight the regions that were most important for the prediction.
                </p>
                <p class="explanation-text" style="margin-top: 10px;">
                    <strong>How to read this map:</strong> The heatmap is overlaid on your brain scan.
                    <span style="color: #ff4444;">■ Red/Orange areas</span> = regions the model focused on most heavily.
                    <span style="color: #ffff00;">■ Yellow/Green areas</span> = moderately important regions.
                    <span style="color: #0088ff;">■ Blue/Purple areas</span> = regions with little influence on the decision.
                </p>
                <p class="explanation-text" style="margin-top: 10px;">
                    <strong>Clinical relevance:</strong> In Alzheimer's detection, the model should ideally focus on the hippocampus (central memory region),
                    temporal lobes (sides of brain), and ventricular spaces — areas known to show atrophy in Alzheimer's disease.
                </p>
                <div class="gradcam-container">
                    <img src="${result.gradcam}" alt="Grad-CAM Visualization"
                         style="max-width: 400px; width: 100%; border-radius: 10px; border: 3px solid #00eaff; box-shadow: 0 4px 20px rgba(0, 234, 255, 0.3);">
                    <!-- Color scale legend -->
                    <div class="heatmap-legend">
                        <div class="legend-item">
                            <div class="legend-color" style="background: linear-gradient(to right, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000); height: 20px; border-radius: 10px; margin-bottom: 5px;"></div>
                            <div class="legend-labels" style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #fff; opacity: 0.8;">
                                <span>🔵 Low Influence</span>
                                <span>🔴 High Influence</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- CLINICAL INTERPRETATION (only shown if insight data exists for this class) -->
            ${insight ? `
            <div class="explainability-section">
                <h3>💡 Clinical Interpretation: ${insight.icon} ${result.prediction}</h3>
                <div class="clinical-insights">
                    <!-- Clinical significance -->
                    <div style="margin-bottom: 15px; padding: 15px; background: rgba(${insight.color === '#00ff88' ? '0,255,136' : insight.color === '#ffdd00' ? '255,221,0' : insight.color === '#ff8800' ? '255,136,0' : '255,68,68'},0.1); border-left: 4px solid ${insight.color}; border-radius: 5px;">
                        <p><strong style="color: ${insight.color};">Clinical Significance:</strong></p>
                        <p style="margin-top: 8px;">${insight.clinical}</p>
                    </div>
                    <!-- Structural indicators -->
                    <div style="margin-bottom: 15px; padding: 15px; background: rgba(0,234,255,0.05); border-left: 4px solid #00eaff; border-radius: 5px;">
                        <p><strong style="color: #00eaff;">Structural Indicators Detected:</strong></p>
                        <p style="margin-top: 8px;">${insight.indicators}</p>
                    </div>
                    <!-- Brain regions analysis -->
                    <div style="margin-bottom: 15px; padding: 15px; background: rgba(0,234,255,0.05); border-left: 4px solid #00eaff; border-radius: 5px;">
                        <p><strong style="color: #00eaff;">Brain Regions Analysis:</strong></p>
                        <p style="margin-top: 8px;">${insight.brainRegions}</p>
                    </div>
                    <!-- Medical recommendation -->
                    <div style="margin-bottom: 15px; padding: 15px; background: rgba(0,234,255,0.05); border-left: 4px solid #00eaff; border-radius: 5px;">
                        <p><strong style="color: #00eaff;">Medical Recommendation:</strong></p>
                        <p style="margin-top: 8px;">${insight.recommendation}</p>
                    </div>
                    <!-- Next steps -->
                    <div style="padding: 15px; background: rgba(${insight.color === '#00ff88' ? '0,255,136' : insight.color === '#ffdd00' ? '255,221,0' : insight.color === '#ff8800' ? '255,136,0' : '255,68,68'},0.1); border-left: 4px solid ${insight.color}; border-radius: 5px;">
                        <p><strong style="color: ${insight.color};">Suggested Next Steps:</strong></p>
                        <p style="margin-top: 8px;">${insight.nextSteps}</p>
                    </div>
                </div>
            </div>
            ` : ''}
    `;

    // --- ADVANCED EXPLAINABILITY (LIME + SHAP) ---
    // Only rendered if the server returned LIME and SHAP results
    if (result.lime && result.shap) {
        html += `
            <div class="explainability-section">
                <h3>🔬 Advanced Explainability Analysis</h3>
                <p class="explanation-text">The following methods provide deeper insight into <em>why</em> the model made this prediction, going beyond visual attention to quantify feature importance.</p>

                <!-- LIME SECTION -->
                <div style="margin: 20px 0; padding: 20px; background: rgba(0,234,255,0.05); border-radius: 10px; border: 1px solid rgba(0,234,255,0.2);">
                    <h4 style="color: #00eaff; margin-bottom: 10px;">🔍 LIME (Local Interpretable Model-agnostic Explanations)</h4>
                    <p class="explanation-text"><strong>What LIME does:</strong> LIME works by creating hundreds of slightly modified versions of your brain scan (hiding different regions each time) and observing how the model's prediction changes. This reveals which specific regions are truly essential for the diagnosis.</p>
                    <p class="explanation-text" style="margin-top: 8px;"><strong>Result:</strong> ${result.lime.text}</p>
                    <p class="explanation-text" style="margin-top: 8px;"><strong>How to read:</strong> The yellow highlighted boundaries outline the brain regions that, when present, most strongly support the "${result.prediction}" diagnosis. Regions outside the boundaries had minimal impact on the decision.</p>
                    <div style="text-align: center; margin-top: 15px;">
                        <img src="${result.lime.image}" alt="LIME Explanation"
                             style="max-width: 500px; width: 100%; border-radius: 10px; border: 2px solid #00eaff;">
                    </div>
                    <p class="explanation-text" style="margin-top: 15px; font-size: 0.9rem; opacity: 0.8;">
                        <strong>Clinical value:</strong> LIME helps verify that the model is focusing on anatomically relevant brain structures rather than image artifacts or background noise, increasing trust in the AI's decision-making process.
                    </p>
                </div>

                <!-- SHAP SECTION -->
                <div style="margin: 30px 0; padding: 20px; background: rgba(0,234,255,0.05); border-radius: 10px; border: 1px solid rgba(0,234,255,0.2);">
                    <h4 style="color: #00eaff; margin-bottom: 10px;">📐 SHAP (SHapley Additive exPlanations)</h4>
                    <p class="explanation-text"><strong>What SHAP does:</strong> SHAP is grounded in game theory (Shapley values). It treats each pixel as a "player" in a game and calculates how much each pixel contributed to the final prediction score. Unlike LIME, SHAP provides mathematically rigorous, quantitative attribution values.</p>
                    <p class="explanation-text" style="margin-top: 8px;"><strong>Result:</strong> ${result.shap.text}</p>
                    <p class="explanation-text" style="margin-top: 8px;"><strong>How to read:</strong> Brighter/warmer areas in the SHAP map indicate pixels that pushed the model strongly toward the "${result.prediction}" classification. Darker areas had little to no contribution. The colorbar on the right shows the scale of feature importance values.</p>
                    <div style="text-align: center; margin-top: 15px;">
                        <img src="${result.shap.image}" alt="SHAP Explanation"
                             style="max-width: 600px; width: 100%; border-radius: 10px; border: 2px solid #00eaff;">
                    </div>
                    <p class="explanation-text" style="margin-top: 15px; font-size: 0.9rem; opacity: 0.8;">
                        <strong>Clinical value:</strong> SHAP provides a unified, consistent measure of feature importance. The low average importance values typical in medical imaging indicate the model is making decisions based on subtle, distributed patterns — which is expected and appropriate for complex neurological conditions like Alzheimer's disease.
                    </p>
                </div>

                <!-- COMPARISON TABLE: All three methods side by side -->
                <div style="padding: 15px; background: rgba(0,234,255,0.08); border-radius: 10px; border: 1px solid rgba(0,234,255,0.3);">
                    <h4 style="color: #00eaff; margin-bottom: 10px;">🔄 Comparing All Three Methods</h4>
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
                        <tr style="border-bottom: 1px solid rgba(0,234,255,0.3);">
                            <th style="text-align: left; padding: 8px; color: #00eaff;">Method</th>
                            <th style="text-align: left; padding: 8px; color: #00eaff;">What It Shows</th>
                            <th style="text-align: left; padding: 8px; color: #00eaff;">Best Used For</th>
                        </tr>
                        <tr style="border-bottom: 1px solid rgba(0,234,255,0.1);">
                            <td style="padding: 8px;">Grad-CAM</td>
                            <td style="padding: 8px;">Where the model "looks"</td>
                            <td style="padding: 8px;">Quick visual validation</td>
                        </tr>
                        <tr style="border-bottom: 1px solid rgba(0,234,255,0.1);">
                            <td style="padding: 8px;">LIME</td>
                            <td style="padding: 8px;">Which regions matter most</td>
                            <td style="padding: 8px;">Region-level explanation</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px;">SHAP</td>
                            <td style="padding: 8px;">Exact pixel contributions</td>
                            <td style="padding: 8px;">Quantitative proof</td>
                        </tr>
                    </table>
                </div>
            </div>
        `;
    }

    html += `</div>`;

    // --- Model transparency and limitations (shown only once, after first image) ---
    if (index === 0) {
        html += `
            <div class="explainability-section">
                <h3>⚠️ Important Considerations</h3>
                <div class="considerations-box">
                    <ul>
                        <li><strong>Model Limitations:</strong> This AI model achieved 73% accuracy on test data. It should be used as a screening tool, not for definitive diagnosis.</li>
                        <li><strong>Dataset Context:</strong> Trained on 10,240 brain MRI scans across 4 severity levels.</li>
                        <li><strong>Recommended Action:</strong> Always combine AI predictions with clinical examination, patient history, and additional diagnostic tests.</li>
                        <li><strong>False Positives/Negatives:</strong> The model may misclassify cases, especially in borderline or atypical presentations.</li>
                    </ul>
                </div>
            </div>

            <!-- MODEL TRANSPARENCY CARDS -->
            <div class="explainability-section">
                <h3>📋 Model Transparency</h3>
                <div class="model-info-grid">
                    <div class="info-card">
                        <div class="info-label">Architecture</div>
                        <div class="info-value">MobileNetV2</div>
                    </div>
                    <div class="info-card">
                        <div class="info-label">Input Size</div>
                        <div class="info-value">128×128 pixels</div>
                    </div>
                    <div class="info-card">
                        <div class="info-label">Training Samples</div>
                        <div class="info-value">8,192 images</div>
                    </div>
                    <div class="info-card">
                        <div class="info-label">Test Accuracy</div>
                        <div class="info-value">73%</div>
                    </div>
                </div>
            </div>

            <!-- CONFUSION MATRIX -->
            <div class="explainability-section" id="confusionMatrixSection">
                <h3>📊 Model Confusion Matrix</h3>
                <p class="explanation-text">
                    <strong>What is a Confusion Matrix?</strong> It shows how well the model performed on the test dataset.
                    Each row represents the actual class, each column represents the predicted class.
                    The diagonal values (top-left to bottom-right) show correct predictions — higher diagonal values mean better performance.
                    Off-diagonal values show misclassifications.
                </p>
                <div id="confusionMatrixContainer" style="text-align: center; margin-top: 15px;">
                    <p style="color: rgba(255,255,255,0.5); font-size: 0.9rem;">Loading confusion matrix...</p>
                </div>
            </div>
        `;

        // Load confusion matrix from metrics endpoint
        fetch('http://localhost:5001/metrics')
            .then(res => res.json())
            .then(data => {
                const container = document.getElementById('confusionMatrixContainer');
                if (data.confusion_matrix) {
                    container.innerHTML = `
                        <img src="${data.confusion_matrix}" alt="Confusion Matrix"
                             style="max-width: 600px; width: 100%; border-radius: 10px; border: 2px solid rgba(0,234,255,0.3);">
                        <p style="margin-top: 12px; font-size: 0.85rem; color: rgba(255,255,255,0.6);">
                            Overall Test Accuracy: <strong style="color:#00eaff;">${data.accuracy}</strong> &nbsp;|&nbsp;
                            Precision: <strong style="color:#00eaff;">${data.precision}</strong> &nbsp;|&nbsp;
                            Recall: <strong style="color:#00eaff;">${data.recall}</strong> &nbsp;|&nbsp;
                            F1 Score: <strong style="color:#00eaff;">${data.f1_score}</strong>
                        </p>
                    `;
                } else {
                    container.innerHTML = `<p style="color: rgba(255,255,255,0.4);">Confusion matrix not available.</p>`;
                }
            })
            .catch(() => {
                const container = document.getElementById('confusionMatrixContainer');
                if (container) container.innerHTML = `<p style="color: rgba(255,255,255,0.4);">Could not load confusion matrix.</p>`;
            });
    }

    return html;
}

// =============================================================================
// PREDICTION HISTORY MODAL
// =============================================================================

/**
 * Fetches prediction history from the server and displays it in a modal.
 * Shows patient details, uploaded images, and prediction results.
 */
async function showHistory() {
    try {
        // Fetch all history entries from the backend
        const response = await fetch("http://localhost:5001/history");
        const history = await response.json();

        const modal = document.getElementById("historyModal");
        const content = document.getElementById("historyContent");

        if (history.length === 0) {
            // Show empty state message
            content.innerHTML = '<p style="text-align: center; color: #fff; padding: 40px;">No prediction history available yet.</p>';
        } else {
            // Reverse to show most recent entries first
            content.innerHTML = history.reverse().map(entry => `
                <div class="history-entry">
                    <div class="history-header">
                        <div>
                            <h3>${entry.patient.patientName}</h3>
                            <p>ID: ${entry.patient.patientId || 'N/A'} | Age: ${entry.patient.age} | Gender: ${entry.patient.gender}</p>
                            <p style="font-size: 0.85rem; opacity: 0.7;">Date: ${entry.patient.timestamp}</p>
                        </div>
                        <!-- Delete button for this history entry -->
                        <button onclick="deleteHistory('${entry.id}')" class="delete-btn">🗑️ Delete</button>
                    </div>
                    <!-- Show thumbnail and prediction for each image in this entry -->
                    <div class="history-results">
                        ${entry.results.map((result, idx) => `
                            <div class="history-result-item">
                                <img src="${result.original_image}" alt="${result.filename}">
                                <div class="history-result-info">
                                    <p><strong>${result.filename}</strong></p>
                                    <p>Prediction: <span style="color: #00eaff;">${result.prediction}</span></p>
                                    <p>Confidence: <span style="color: #00eaff;">${result.confidence}</span></p>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                    ${entry.patient.notes ? `<p style="margin-top: 10px;"><strong>Notes:</strong> ${entry.patient.notes}</p>` : ''}
                </div>
            `).join('');
        }

        // Show the modal
        modal.classList.remove("hidden");
        modal.style.display = "flex";

    } catch (error) {
        alert("Error loading history: " + error.message);
    }
}

/**
 * Closes the history modal by hiding it.
 */
function closeHistory() {
    const modal = document.getElementById("historyModal");
    if (modal) {
        modal.classList.add("hidden");
        modal.style.display = "none";
    }
}

/**
 * Deletes a specific history entry by ID after user confirmation.
 * Refreshes the history modal after deletion.
 *
 * @param {string} id - The history entry ID (timestamp string)
 */
async function deleteHistory(id) {
    if (!confirm("Are you sure you want to delete this entry?")) return;

    try {
        // Send DELETE request to the backend
        await fetch(`http://localhost:5001/history/${id}`, { method: 'DELETE' });

        // Refresh the history modal to reflect the deletion
        showHistory();
    } catch (error) {
        alert("Error deleting entry: " + error.message);
    }
}

// =============================================================================
// EVENT LISTENERS — DOM READY
// =============================================================================

window.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById("historyModal");
    if (modal) {
        // Close history modal when user clicks on the dark backdrop (outside the modal box)
        modal.addEventListener("click", (e) => {
            if (e.target.id === "historyModal") {
                closeHistory();
            }
        });
    }
});

console.log("Script initialization complete!");
