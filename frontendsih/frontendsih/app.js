/**
 * OnionVision AI - Frontend Logic
 * Smart India Hackathon (SIH) Project
 * Handles image selection, drag-and-drop, sample presets,
 * realistic model inference simulation, and dynamic parameter grading.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const dropzonePrompt = document.getElementById('dropzone-prompt');
  const previewWrapper = document.getElementById('preview-wrapper');
  const imagePreview = document.getElementById('image-preview');
  const previewFilename = document.getElementById('preview-filename');
  const previewFilesize = document.getElementById('preview-filesize');
  const btnClearImage = document.getElementById('btn-clear-image');
  const btnAnalyze = document.getElementById('btn-analyze');
  const analysisStatus = document.getElementById('analysis-status');
  const statusText = document.getElementById('status-text');
  const statusIndicator = document.getElementById('status-indicator');

  // Sample Buttons
  const sampleButtons = document.querySelectorAll('.btn-sample');

  // Result Elements
  const resultClassName = document.getElementById('result-class-name');
  const resultSubgrade = document.getElementById('result-subgrade');
  const resultConfidence = document.getElementById('result-confidence');
  const confidenceBadge = document.getElementById('confidence-badge');
  const verdictTag = document.getElementById('verdict-tag');
  const verdictTimestamp = document.getElementById('verdict-timestamp');

  // Parameter Elements
  const paramSkinVal = document.getElementById('param-skin-val');
  const paramSkinBar = document.getElementById('param-skin-bar');
  const paramSkinDesc = document.getElementById('param-skin-desc');

  const paramSproutVal = document.getElementById('param-sprout-val');
  const paramSproutBar = document.getElementById('param-sprout-bar');
  const paramSproutDesc = document.getElementById('param-sprout-desc');

  const paramMoldVal = document.getElementById('param-mold-val');
  const paramMoldBar = document.getElementById('param-mold-bar');
  const paramMoldDesc = document.getElementById('param-mold-desc');

  const paramLifeVal = document.getElementById('param-life-val');
  const paramLifeBar = document.getElementById('param-life-bar');
  const paramLifeDesc = document.getElementById('param-life-desc');

  const recommendationDesc = document.getElementById('recommendation-desc');
  const btnPrintReport = document.getElementById('btn-print-report');
  const btnResetTest = document.getElementById('btn-reset-test');

  // Preset Dataset
  const sampleData = {
    'grade-a': {
      imagePath: 'assets/sample-grade-a.jpg',
      filename: 'sample_grade_a_nasik_red.jpg',
      filesize: '185 KB',
      className: 'Good Quality',
      subgrade: 'Grade A (Export Standard)',
      confidence: '96.4%',
      confidenceLevel: 'High Confidence',
      verdictClass: 'tag-pass',
      verdictText: 'Passes APMC Mandi Benchmark',
      skin: { val: '96%', bar: '96%', fillClass: '', desc: 'Firm outer dry scales, intact protective tunic' },
      sprout: { val: 'None (0%)', bar: '4%', fillClass: 'param-fill-green', desc: 'Neck tightly closed, no dormant shoot emergence' },
      mold: { val: '0.2% (Clean)', bar: '3%', fillClass: 'param-fill-green', desc: 'No Aspergillus niger or basal rot detected' },
      life: { val: '60 - 75 Days', bar: '88%', fillClass: '', desc: 'Suitable for long-term ventilated cold storage' },
      recommendation: 'Approved for premium domestic distribution and export packaging. No quarantine needed.'
    },
    'sprouted': {
      imagePath: 'assets/sample-sprouted.jpg',
      filename: 'sample_sprouted_defect.jpg',
      filesize: '192 KB',
      className: 'Defective: Sprouted',
      subgrade: 'Grade C (Industrial / Culinary Use Only)',
      confidence: '93.8%',
      confidenceLevel: 'High Confidence',
      verdictClass: 'tag-warn',
      verdictText: 'Storage Risk: Immediate Cull / Processing',
      skin: { val: '72%', bar: '72%', fillClass: 'param-fill-amber', desc: 'Neck loosening with internal tissue softening' },
      sprout: { val: '46% (High)', bar: '78%', fillClass: 'param-fill-amber', desc: 'Active vegetative shoot detected at apical neck' },
      mold: { val: '2.4% (Trace)', bar: '15%', fillClass: 'param-fill-amber', desc: 'Superficial dampness observed near root plate' },
      life: { val: '5 - 10 Days', bar: '18%', fillClass: 'param-fill-amber', desc: 'High metabolic respiration; prompt usage required' },
      recommendation: 'Divert immediately to dehydrated flake/powder processing or local culinary markets.'
    },
    'mold': {
      imagePath: 'assets/sample-mold.jpg',
      filename: 'sample_fungal_mold_reject.jpg',
      filesize: '210 KB',
      className: 'Defective: Fungal Decay',
      subgrade: 'Rejected (Non-Marketable Lot)',
      confidence: '97.2%',
      confidenceLevel: 'High Confidence',
      verdictClass: 'tag-fail',
      verdictText: 'Quarantine / Isolate from Lot',
      skin: { val: '44%', bar: '44%', fillClass: 'param-fill-red', desc: 'Outer tunic breakdown and epidermal maceration' },
      sprout: { val: '14% (Secondary)', bar: '22%', fillClass: 'param-fill-red', desc: 'Pathological soft rot breakdown' },
      mold: { val: '38.5% (Severe)', bar: '85%', fillClass: 'param-fill-red', desc: 'Black mold colony (Aspergillus niger) identified' },
      life: { val: '0 Days (Cull)', bar: '5%', fillClass: 'param-fill-red', desc: 'High risk of airborne cross-infection to adjacent crates' },
      recommendation: 'Quarantine and destroy immediately. Sanitize storage crates to prevent fungal spore contagion.'
    }
  };

  // State
  let currentMode = 'grade-a'; // 'sample' or 'custom'
  let hasImage = false;
  let currentFile = null;

  // Initialize with Grade A sample
  loadSample('grade-a');

  // Drag & Drop Handlers
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('drag-over');
    }, false);
  });


  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('drag-over');
    }, false);
  });


  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files.length > 0) {
      handleCustomFile(files[0]);
    }
  });


  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleCustomFile(e.target.files[0]);
    }
  });


  // Clear / Remove Image
  btnClearImage.addEventListener('click', (e) => {
    e.stopPropagation();
    resetImagePreview();
  });


  // Sample Buttons Handler
  sampleButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const sampleKey = btn.getAttribute('data-sample');
      if (sampleKey) {
        sampleButtons.forEach(b => b.classList.remove('active-sample'));
        btn.classList.add('active-sample');
        loadSample(sampleKey);
      }
    });

  });


  function loadSample(sampleKey) {
    const data = sampleData[sampleKey];
    if (!data) return;

    currentMode = sampleKey;
    hasImage = true;

    // Update Preview
    imagePreview.src = data.imagePath;
    previewFilename.textContent = data.filename;
    previewFilesize.textContent = data.filesize;

    dropzonePrompt.classList.add('hidden');
    previewWrapper.classList.remove('hidden');

    // Run direct display update
    renderPredictionResult(data);
  }

  function handleCustomFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file (JPEG, PNG, or WEBP).');
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      imagePreview.src = event.target.result;
      previewFilename.textContent = file.name;
      previewFilesize.textContent = formatBytes(file.size);

      dropzonePrompt.classList.add('hidden');
      previewWrapper.classList.remove('hidden');

      hasImage = true;
      currentMode = 'custom';
      currentFile = file;

      // Unselect preset samples
      sampleButtons.forEach(b => b.classList.remove('active-sample'));

      // Wait for user to click Analyze Image
    };
    reader.readAsDataURL(file);
  }

  function resetImagePreview() {
    fileInput.value = '';
    imagePreview.src = '';
    previewFilename.textContent = '';
    previewFilesize.textContent = '';

    previewWrapper.classList.add('hidden');
    dropzonePrompt.classList.remove('hidden');

    sampleButtons.forEach(b => b.classList.remove('active-sample'));
    hasImage = false;
    currentMode = null;
    currentFile = null;
  }

  function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  // Analyze Button Click
  btnAnalyze.addEventListener('click', async () => {
    if (!hasImage) {
      alert("Please upload an image first.");
      return;
    }

    if (currentMode === 'custom' && currentFile) {
      await runRealInference(currentFile);
    } else {
      // If it's a sample, fetch the image as a blob and send to API
      try {
        const response = await fetch(imagePreview.src);
        const blob = await response.blob();
        const file = new File([blob], sampleData[currentMode].filename, { type: 'image/jpeg' });

        await runRealInference(file);
      } catch (err) {
        alert("Error loading sample image for analysis.");
      }
    }
  });



  async function runRealInference(file) {
    btnAnalyze.disabled = true;
    analysisStatus.classList.remove('hidden');
    statusIndicator.textContent = 'Processing with AI...';
    statusIndicator.className = 'badge-status';
    
    // Reset steps
    const steps = ['upload', 'model', 'infer', 'eval', 'done'];
    steps.forEach(s => {
        const el = document.getElementById(`step-${s}`);
        if(el) { el.style.color = 'var(--text-secondary)'; el.querySelector('.step-icon').textContent = '○'; }
    });

    
    const setStep = (step) => {
        const el = document.getElementById(`step-${step}`);
        if(el) { el.style.color = 'var(--primary)'; el.querySelector('.step-icon').textContent = '✓'; }
    }
    
    setStep('upload');

    const formData = new FormData();
    formData.append('image', file);
    formData.append('conf_threshold', '0.25');

    try {
      setStep('model');
      
      const response = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        body: formData
      });

      
      setStep('infer');
      const result = await response.json();
      
      if (!response.ok || !result.success) {
        throw new Error(result.error || "Analysis failed");
      }
      
      setStep('eval');
      
      // Update image with bounding boxes
      if (result.annotated_image_base64) {
        imagePreview.src = 'data:image/jpeg;base64,' + result.annotated_image_base64;
      }
      
      processModelResults(result.onions, result.metadata);
      
      setStep('done');
      statusIndicator.textContent = 'Analysis Complete';
      statusIndicator.className = 'badge-status badge-ready';
      
      // Show export button
      const btnExport = document.getElementById('btn-export');
      if (btnExport) btnExport.style.display = 'inline-block';
      
    } catch (error) {
      console.error(error);
      statusIndicator.textContent = 'Error';
      statusIndicator.className = 'badge-status badge-error';
      alert("Error: " + error.message);
    } finally {
      btnAnalyze.disabled = false;
      setTimeout(() => { analysisStatus.classList.add('hidden'); }, 2000);
    }
  }

  function processModelResults(onions, metadata) {
    if (!onions || onions.length === 0) {
      document.getElementById('result-class-name').textContent = "No Onion Detected";
      document.getElementById('result-subgrade').textContent = "N/A";
      document.getElementById('result-confidence').textContent = "N/A";
      document.getElementById('verdict-tag').textContent = "Upload a clearer image";
      document.getElementById('verdict-tag').className = "verdict-tag tag-warn";
      document.getElementById('meta-objects').textContent = "0";
      
      if (metadata) {
          document.getElementById('meta-model').textContent = metadata.model || 'Unknown';
          document.getElementById('meta-time').textContent = (metadata.inference_time_ms || 0) + ' ms';
      }
      
      document.getElementById('why-result-list').innerHTML = "<li>No recognizable onion was detected in this image.</li>";
      document.getElementById('multi-onion-container').style.display = 'none';
      return;
    }

    let worstGrade = 'Class 1';
    let worstConfidence = 0;
    let hasSprout = false;
    let hasRotten = false;
    let hasDoubleSplit = false;
    let currentMaxSeverity = 0;
    const gradeHierarchy = { 'Reject': 4, 'Class 2': 3, 'Class 1': 2, 'Extra Class': 1 };
    
    let defectList = [];
    let reasons = [];

    const batchList = document.getElementById('batch-list');
    batchList.innerHTML = '';

    onions.forEach(onion => {
      let gClass = onion.grade ? onion.grade.class : 'Grade Not Detected';
      let gConf = onion.grade ? Math.round(onion.grade.confidence * 100) : 0;
      
      let sev = gradeHierarchy[gClass] || 0;
      if (sev > currentMaxSeverity) {
          currentMaxSeverity = sev;
          worstGrade = gClass;
          worstConfidence = onion.grade ? onion.grade.confidence : 0;
      }
      
      let oDefects = [];
      onion.defects.forEach(d => {
         if (d.class === 'sprout') hasSprout = true;
         if (d.class === 'rotten') hasRotten = true;
         if (d.class === 'double_split') hasDoubleSplit = true;
         if (!defectList.includes(d.class)) defectList.push(d.class);
         oDefects.push(`${d.class} (${Math.round(d.confidence*100)}%)`);
      });

      
      // Add to batch list UI
      let defectText = oDefects.length > 0 ? oDefects.join(', ') : 'None';
      batchList.innerHTML += `<div style="padding:10px; background:var(--bg-card); border:1px solid var(--border-color); border-radius:4px;">
        <strong>Onion ${onion.onion_id}</strong> &rarr; ${gClass} (${gConf}%) <br>
        <span style="color:var(--text-muted); font-size:0.85rem;">Defects: ${defectText}</span>
      </div>`;
    });

    
    if (currentMaxSeverity === 0) worstGrade = 'Grade Not Detected';

    let verdictText = 'Cleared for standard use';
    let verdictClass = 'tag-pass';
    if (worstGrade === 'Reject' || hasRotten) {
        verdictText = 'Rejected / Quarantine';
        verdictClass = 'tag-fail';
    } else if (worstGrade === 'Class 2' || hasSprout) {
        verdictText = 'Substandard / Culinary Use';
        verdictClass = 'tag-warn';
    }
    
    let subgrade = defectList.length > 0 ? "Defects: " + defectList.join(', ') : "No defects found";
    let confDisplay = worstConfidence > 0 ? (worstConfidence * 100).toFixed(1) + '%' : 'N/A';
    
    // Skin
    let skinObj = { val: 'Not predicted', bar: '0%', fillClass: '', desc: 'General skin integrity is not evaluated by current model.' };
    if (hasDoubleSplit) skinObj = { val: 'Double Split Detected', bar: '100%', fillClass: 'param-fill-amber', desc: 'Model detected a double split deformity.' };
    
    // Sprout
    let sproutObj = { val: 'None (0%)', bar: '0%', fillClass: 'param-fill-green', desc: 'No active apical sprout shoot detected (Model)' };
    if (hasSprout) sproutObj = { val: 'Sprout Detected', bar: '100%', fillClass: 'param-fill-amber', desc: 'Active vegetative shoot detected (Model prediction)' };
    
    // Mold
    let moldObj = { val: 'Clean (0%)', bar: '0%', fillClass: 'param-fill-green', desc: 'No pathological fungal colonies present (Model)' };
    if (hasRotten) moldObj = { val: 'Rot / Mold Detected', bar: '100%', fillClass: 'param-fill-red', desc: 'Pathological fungal decay present (Model prediction)' };
    
    // Shelf Life (Calculated Rule)
    let lifeObj = { val: 'Unknown', bar: '0%', fillClass: '', desc: 'Not predicted directly' };
    if (worstGrade === 'Reject' || hasRotten) {
        lifeObj = { val: '0 Days (Cull)', bar: '5%', fillClass: 'param-fill-red', desc: 'Derived: Rotten or Reject grade detected' };
    } else if (hasSprout) {
        lifeObj = { val: '5 - 10 Days', bar: '20%', fillClass: 'param-fill-amber', desc: 'Derived: Sprouting reduces shelf life significantly' };
    } else if (worstGrade === 'Extra Class' || worstGrade === 'Class 1') {
        lifeObj = { val: '50 - 65 Days', bar: '80%', fillClass: 'param-fill-green', desc: 'Derived: High quality grade with no defects' };
    }
    
    // Generate Reasons
    if (worstGrade === 'Reject') reasons.push("The grading model flagged the batch as Reject.");
    else if (worstGrade === 'Extra Class' || worstGrade === 'Class 1') reasons.push(`High-quality grade (${worstGrade}) detected.`);
    
    if (hasRotten) reasons.push("Fungal rot (rotten) was explicitly detected.");
    if (hasSprout) reasons.push("Sprouting (sprout) was explicitly detected.");
    if (hasDoubleSplit) reasons.push("A double split structural deformity was detected.");
    if (defectList.length === 0) reasons.push("No explicit defects were found by the defect model.");
    
    document.getElementById('why-result-list').innerHTML = reasons.map(r => `<li>${r}</li>`).join('');
    
    document.getElementById('meta-objects').textContent = onions.length;
    document.getElementById('batch-count').textContent = onions.length;
    document.getElementById('multi-onion-container').style.display = onions.length > 0 ? 'block' : 'none';
    
    if (metadata) {
        document.getElementById('meta-model').textContent = metadata.model || 'YOLO11s';
        document.getElementById('meta-time').textContent = (metadata.inference_time_ms || 0) + ' ms';
        document.getElementById('info-arch').textContent = metadata.model || 'YOLO11s';
        document.getElementById('info-task').textContent = metadata.task || 'Defect & Grading';
        document.getElementById('info-size').textContent = metadata.image_size || 'Unknown';
    }

    let recText = `Overall Batch Analysis (${onions.length} onion(s) detected). Please refer to the annotated image.`;
    if (worstGrade === 'Reject' || hasRotten) {
        recText = 'Quarantine and sort immediately. Rotten or severely defective onions present in batch.';
    } else if (worstGrade === 'Class 2' || hasSprout) {
        recText = 'Divert to local culinary markets or processing. Not suitable for long-term storage.';
    } else if (worstGrade === 'Extra Class' || worstGrade === 'Class 1') {
        recText = 'Approved for premium domestic distribution and export packaging. No quarantine needed.';
    } else if (currentMaxSeverity === 0) {
        recText = 'Unable to confidently grade the batch. Manual inspection recommended.';
    }

    let targetData = {
      className: worstGrade,
      subgrade: subgrade,
      confidence: confDisplay,
      confidenceLevel: (worstConfidence > 0.7 ? 'High Confidence' : 'Low Confidence'),
      verdictClass: verdictClass,
      verdictText: verdictText,
      skin: skinObj,
      sprout: sproutObj,
      mold: moldObj,
      life: lifeObj,
      recommendation: recText
    };
    
    renderPredictionResult(targetData);
  }

  function renderPredictionResult(data) {
    if (resultClassName) resultClassName.textContent = data.className || '';
    if (resultSubgrade) resultSubgrade.textContent = data.subgrade || '';
    if (resultConfidence) resultConfidence.textContent = data.confidence || '';
    if (confidenceBadge) confidenceBadge.textContent = data.confidenceLevel || 'High Confidence';
    
    if (verdictTag) {
        verdictTag.textContent = data.verdictText || '';
        verdictTag.className = `verdict-tag ${data.verdictClass || ''}`;
    }
    
    const now = new Date();
    if (verdictTimestamp) verdictTimestamp.textContent = `Analyzed ${now.toLocaleTimeString()}`;
    
    // Skin
    if (data.skin) {
        if (paramSkinVal) paramSkinVal.textContent = data.skin.val || '';
        if (paramSkinBar) {
            paramSkinBar.style.width = data.skin.bar || '0%';
            paramSkinBar.className = `param-bar-fill ${data.skin.fillClass || ''}`;
        }
        if (paramSkinDesc) paramSkinDesc.textContent = data.skin.desc || '';
    }
    
    // Sprout
    if (data.sprout) {
        if (paramSproutVal) paramSproutVal.textContent = data.sprout.val || '';
        if (paramSproutBar) {
            paramSproutBar.style.width = data.sprout.bar || '0%';
            paramSproutBar.className = `param-bar-fill ${data.sprout.fillClass || ''}`;
        }
        if (paramSproutDesc) paramSproutDesc.textContent = data.sprout.desc || '';
    }
    
    // Mold
    if (data.mold) {
        if (paramMoldVal) paramMoldVal.textContent = data.mold.val || '';
        if (paramMoldBar) {
            paramMoldBar.style.width = data.mold.bar || '0%';
            paramMoldBar.className = `param-bar-fill ${data.mold.fillClass || ''}`;
        }
        if (paramMoldDesc) paramMoldDesc.textContent = data.mold.desc || '';
    }
    
    // Life
    if (data.life) {
        if (paramLifeVal) paramLifeVal.textContent = data.life.val || '';
        if (paramLifeBar) {
            paramLifeBar.style.width = data.life.bar || '0%';
            paramLifeBar.className = `param-bar-fill ${data.life.fillClass || ''}`;
        }
        if (paramLifeDesc) paramLifeDesc.textContent = data.life.desc || '';
    }
    
    if (recommendationDesc) recommendationDesc.textContent = data.recommendation || '';
  }

  // Export Report Button
  const btnExport = document.getElementById('btn-export');
  if (btnExport) {
      btnExport.addEventListener('click', () => {
          window.print();
      });
  }
});
