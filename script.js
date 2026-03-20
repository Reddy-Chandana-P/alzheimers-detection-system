console.log("Script loaded successfully!");

const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const form = document.getElementById("uploadForm");
const resultBox = document.getElementById("result");

// Real-time validation feedback
const inputs = form.querySelectorAll('input[required], select[required]');
inputs.forEach(input => {
    input.addEventListener('blur', () => {
        if (!input.validity.valid) {
            input.style.borderColor = '#ff4444';
        }
    });
    
    input.addEventListener('input', () => {
        if (input.validity.valid) {
            input.style.borderColor = 'rgba(0, 234, 255, 0.3)';
        }
    });
});

// Phone number formatting
const contactInput = document.getElementById('contactNumber');
if (contactInput) {
    contactInput.addEventListener('input', (e) => {
        let value = e.target.value.replace(/\D/g, '');
        if (value.length > 15) value = value.slice(0, 15);
        e.target.value = value;
    });
}

// Age validation
const ageInput = document.getElementById('age');
if (ageInput) {
    ageInput.addEventListener('input', (e) => {
        const value = parseInt(e.target.value);
        if (value < 1) e.target.value = 1;
        if (value > 120) e.target.value = 120;
    });
}

// Character counter for notes
const notesInput = document.getElementById('notes');
if (notesInput) {
    const counter = document.createElement('div');
    counter.style.cssText = 'text-align: right; font-size: 0.85rem; color: #00eaff; margin-top: -8px; margin-bottom: 10px;';
    notesInput.parentNode.appendChild(counter);
    
    notesInput.addEventListener('input', () => {
        const remaining = 500 - notesInput.value.length;
        counter.textContent = `${remaining} characters remaining`;
        counter.style.color = remaining < 50 ? '#ff4444' : '#00eaff';
    });
}

imageInput.addEventListener("change", () => {
    preview.innerHTML = "";
    const files = [...imageInput.files];
    
    if (files.length === 0) return;
    
    if (files.length > 5) {
        alert("Maximum 5 images allowed");
        imageInput.value = "";
        return;
    }
    
    files.forEach(file => {
        if (file.size > 10 * 1024 * 1024) {
            alert(`File ${file.name} is too large. Maximum size is 10MB.`);
            return;
        }
        
        if (!file.type.startsWith('image/')) {
            alert(`File ${file.name} is not a valid image.`);
            return;
        }
        
        const img = document.createElement("img");
        img.src = URL.createObjectURL(file);
        img.title = file.name;
        preview.appendChild(img);
    });
});

// Form submission
form.onsubmit = async function(e) {
    e.preventDefault();
    e.stopPropagation();
    
    console.log("Form submission started");

    const requiredFields = form.querySelectorAll('[required]');
    let isValid = true;
    
    requiredFields.forEach(field => {
        if (!field.validity.valid) {
            field.style.borderColor = '#ff4444';
            isValid = false;
        }
    });
    
    if (!isValid) {
        alert("Please fill in all required fields correctly.");
        return false;
    }

    const formData = new FormData();
    const imageFiles = imageInput.files;
    
    if (imageFiles.length === 0) {
        alert("Please select at least one image!");
        return false;
    }

    for (let i = 0; i < imageFiles.length; i++) {
        formData.append("image", imageFiles[i]);
    }
    
    const advancedEnabled = document.getElementById("advancedExplain").checked;
    if (advancedEnabled) {
        formData.append("advanced", "true");
    }
    
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

    const button = form.querySelector("button[type='submit']");
    const originalText = button.innerHTML;
    button.innerHTML = `⏳ Analyzing ${imageFiles.length} image(s)...`;
    button.disabled = true;

    try {
        console.log("Sending request to server...");
        
        const response = await fetch("http://localhost:5001/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Prediction failed");
        }

        const data = await response.json();
        console.log("Data received:", data);

        const multipleResults = document.getElementById("multipleResults");
        multipleResults.innerHTML = '';
        
        data.results.forEach((result, index) => {
            multipleResults.innerHTML += createResultHTML(result, index, data.results.length > 1);
        });

        console.log("Results added to page");
        
        resultBox.classList.remove("hidden");
        resultBox.style.display = "block";
        
        console.log("Result box displayed");
        
        setTimeout(() => {
            resultBox.scrollIntoView({ behavior: "smooth" });
        }, 200);

    } catch (error) {
        console.error("Error:", error);
        alert("Error: " + error.message + "\n\nMake sure the Flask server is running!");
    } finally {
        button.innerHTML = originalText;
        button.disabled = false;
    }
    
    return false;
};

function createResultHTML(result, index, isMultiple) {
    const insights = {
        'No Impairment': `
            <p><strong>Clinical Significance:</strong> The AI model indicates no significant signs of cognitive impairment in the analyzed brain scan.</p>
            <p><strong>Typical Indicators:</strong> Normal brain structure, preserved hippocampal volume, no significant atrophy.</p>
            <p><strong>Recommendation:</strong> Continue routine monitoring. Consider baseline cognitive assessments for future comparison.</p>
        `,
        'Very Mild Impairment': `
            <p><strong>Clinical Significance:</strong> The model detected subtle changes that may indicate very early-stage cognitive decline.</p>
            <p><strong>Typical Indicators:</strong> Minimal hippocampal atrophy, slight ventricular enlargement, early white matter changes.</p>
            <p><strong>Recommendation:</strong> Conduct comprehensive neuropsychological testing. Monitor progression with follow-up scans in 6-12 months.</p>
        `,
        'Mild Impairment': `
            <p><strong>Clinical Significance:</strong> The model identified moderate structural changes consistent with mild cognitive impairment.</p>
            <p><strong>Typical Indicators:</strong> Noticeable hippocampal atrophy, cortical thinning in temporal regions, ventricular enlargement.</p>
            <p><strong>Recommendation:</strong> Comprehensive clinical evaluation recommended. Consider biomarker testing and detailed cognitive assessment.</p>
        `,
        'Moderate Impairment': `
            <p><strong>Clinical Significance:</strong> The model detected significant structural changes indicating moderate cognitive impairment.</p>
            <p><strong>Typical Indicators:</strong> Substantial hippocampal and cortical atrophy, pronounced ventricular enlargement, widespread brain volume loss.</p>
            <p><strong>Recommendation:</strong> Immediate comprehensive evaluation. Discuss treatment options and care planning with patient and family.</p>
        `
    };

    const imageTitle = isMultiple ? `<h3 style="color: #00eaff; margin-bottom: 15px;">Image ${index + 1}: ${result.filename}</h3>` : '';
    
    let html = `
        <div class="result-item" style="margin-bottom: 40px; padding: 25px; background: rgba(0, 234, 255, 0.05); border-radius: 15px; border: 1px solid rgba(0, 234, 255, 0.2);">
            ${imageTitle}
            
            <div class="result-summary">
                <div class="prediction-box">
                    <h3>Primary Diagnosis</h3>
                    <p class="prediction-label"><strong>Status:</strong> <span style="color: #00eaff; font-weight: bold; font-size: 1.3rem;">${result.prediction}</span></p>
                    <p class="confidence-label"><strong>Confidence:</strong> <span style="color: #00eaff; font-weight: bold; font-size: 1.3rem;">${result.confidence}</span></p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${result.confidence}">${result.confidence}</div>
                    </div>
                </div>
            </div>

            <div class="explainability-section">
                <h3>📊 Confidence Distribution Across All Classes</h3>
                <p class="explanation-text">Understanding how the AI evaluated each possible diagnosis:</p>
                <div class="predictions-grid">
    `;
    
    for (const [className, confidence] of Object.entries(result.all_predictions)) {
        const isHighest = className === result.prediction;
        html += `
            <div class="prediction-item" style="${isHighest ? 'border: 2px solid #00eaff; background: rgba(0, 234, 255, 0.15);' : ''}">
                <div class="prediction-item-label">${className}</div>
                <div class="prediction-item-value">${confidence}</div>
                <div class="prediction-item-bar">
                    <div class="prediction-item-fill" style="width: ${confidence}"></div>
                </div>
            </div>
        `;
    }
    
    html += `
                </div>
            </div>

            <div class="explainability-section">
                <h3>🧠 Visual Explanation (Grad-CAM)</h3>
                <p class="explanation-text">
                    <strong>What you're seeing:</strong> The heatmap overlay shows which brain regions most influenced the AI's decision. 
                    Warmer colors (red/yellow) indicate areas the model focused on, while cooler colors (blue/purple) show less relevant regions.
                </p>
                <div class="gradcam-container">
                    <img src="${result.gradcam}" alt="Grad-CAM Visualization" style="max-width: 400px; width: 100%; border-radius: 10px; border: 3px solid #00eaff; box-shadow: 0 4px 20px rgba(0, 234, 255, 0.3);">
                    <div class="heatmap-legend">
                        <div class="legend-item">
                            <div class="legend-color" style="background: linear-gradient(to right, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000); height: 20px; border-radius: 10px; margin-bottom: 5px;"></div>
                            <div class="legend-labels" style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #fff; opacity: 0.8;">
                                <span>Low Influence</span>
                                <span>High Influence</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="explainability-section">
                <h3>💡 Clinical Interpretation Guide</h3>
                <div class="clinical-insights">
                    ${insights[result.prediction] || '<p>Clinical interpretation not available for this classification.</p>'}
                </div>
            </div>
    `;
    
    if (result.lime && result.shap) {
        html += `
            <div class="explainability-section">
                <h3>🔬 Advanced Explainability Methods</h3>
                
                <div style="margin: 20px 0;">
                    <h4 style="color: #00eaff; margin-bottom: 10px;">LIME (Local Interpretable Model-agnostic Explanations)</h4>
                    <p class="explanation-text">${result.lime.text}</p>
                    <div style="text-align: center; margin-top: 15px;">
                        <img src="${result.lime.image}" alt="LIME Explanation" style="max-width: 500px; width: 100%; border-radius: 10px; border: 2px solid #00eaff;">
                    </div>
                    <p class="explanation-text" style="margin-top: 15px; font-size: 0.9rem; opacity: 0.8;">
                        <strong>How LIME works:</strong> LIME creates interpretable explanations by approximating the model locally with an interpretable model. 
                        It perturbs the input image and observes how predictions change, identifying which regions (superpixels) are most important for the decision.
                    </p>
                </div>

                <div style="margin: 30px 0;">
                    <h4 style="color: #00eaff; margin-bottom: 10px;">SHAP (SHapley Additive exPlanations)</h4>
                    <p class="explanation-text">${result.shap.text}</p>
                    <div style="text-align: center; margin-top: 15px;">
                        <img src="${result.shap.image}" alt="SHAP Explanation" style="max-width: 600px; width: 100%; border-radius: 10px; border: 2px solid #00eaff;">
                    </div>
                    <p class="explanation-text" style="margin-top: 15px; font-size: 0.9rem; opacity: 0.8;">
                        <strong>How SHAP works:</strong> SHAP uses game theory (Shapley values) to assign each feature an importance value for a particular prediction. 
                        It provides a unified measure of feature importance by considering all possible feature combinations.
                    </p>
                </div>
            </div>
        `;
    }
    
    html += `
        </div>
    `;
    
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
        `;
    }
    
    return html;
}

// History functions
async function showHistory() {
    try {
        const response = await fetch("http://localhost:5001/history");
        const history = await response.json();
        
        const modal = document.getElementById("historyModal");
        const content = document.getElementById("historyContent");
        
        if (history.length === 0) {
            content.innerHTML = '<p style="text-align: center; color: #fff; padding: 40px;">No prediction history available yet.</p>';
        } else {
            content.innerHTML = history.reverse().map(entry => `
                <div class="history-entry">
                    <div class="history-header">
                        <div>
                            <h3>${entry.patient.patientName}</h3>
                            <p>ID: ${entry.patient.patientId || 'N/A'} | Age: ${entry.patient.age} | Gender: ${entry.patient.gender}</p>
                            <p style="font-size: 0.85rem; opacity: 0.7;">Date: ${entry.patient.timestamp}</p>
                        </div>
                        <button onclick="deleteHistory('${entry.id}')" class="delete-btn">🗑️ Delete</button>
                    </div>
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
        
        modal.classList.remove("hidden");
        modal.style.display = "flex";
    } catch (error) {
        alert("Error loading history: " + error.message);
    }
}

function closeHistory() {
    const modal = document.getElementById("historyModal");
    if (modal) {
        modal.classList.add("hidden");
        modal.style.display = "none";
    }
}

async function deleteHistory(id) {
    if (!confirm("Are you sure you want to delete this entry?")) return;
    
    try {
        await fetch(`http://localhost:5001/history/${id}`, { method: 'DELETE' });
        showHistory();
    } catch (error) {
        alert("Error deleting entry: " + error.message);
    }
}

window.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById("historyModal");
    if (modal) {
        modal.addEventListener("click", (e) => {
            if (e.target.id === "historyModal") {
                closeHistory();
            }
        });
    }
});

console.log("Script initialization complete!");
