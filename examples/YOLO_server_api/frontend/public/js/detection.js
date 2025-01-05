import { checkAccess, authHeaders, showAppropriateLinks } from './common.js';

const API_URL = '/api'; // Base path for the backend API
let originalImageWidth = 0;
let originalImageHeight = 0;

/** Entry point after the DOM has loaded. */
document.addEventListener('DOMContentLoaded', init);

/** Initialisation function to separate setup logic from the DOMContentLoaded callback. */
function init() {
  checkAccess([]);
  showAppropriateLinks();

  const logoutBtn = document.getElementById('logout-btn');
  const form = document.getElementById('detection-form');
  const detectionError = document.getElementById('detection-error');
  const detectionResult = document.getElementById('detection-result');
  const fileDropArea = document.getElementById('file-drop-area');
  const imageInput = document.getElementById('image-input');
  const removeImageBtn = document.getElementById('remove-image-btn');
  const chooseFileBtn = document.querySelector('.choose-file-btn');

  setupLogoutButton(logoutBtn);
  setupRemoveImageButton(removeImageBtn);
  setupChooseFileButton(chooseFileBtn, imageInput);
  setupFileDrop(fileDropArea, imageInput);
  setupFileInputChange(imageInput, fileDropArea, removeImageBtn);
  // Pass fewer arguments by wrapping them in an object
  setupFormSubmission({
    form,
    imageInput,
    detectionError,
    detectionResult
  });
}

/** Set up the logout button event. */
function setupLogoutButton(logoutBtn) {
  if (!logoutBtn) return;
  logoutBtn.addEventListener('click', () => {
    window.location.href = '/login.html';
  });
}

/** Set up the remove image button event. */
function setupRemoveImageButton(removeImageBtn) {
  removeImageBtn.addEventListener('click', removeImage);
}

/** Set up the choose file button to trigger file input. */
function setupChooseFileButton(chooseFileBtn, imageInput) {
  if (!chooseFileBtn) return;
  chooseFileBtn.addEventListener('click', (e) => {
    e.preventDefault();
    imageInput.click();
  });
}

/** Set up drag-and-drop file handling. */
function setupFileDrop(fileDropArea, imageInput) {
  fileDropArea.addEventListener('dragover', handleDragOver);
  fileDropArea.addEventListener('dragleave', handleDragLeave);
  fileDropArea.addEventListener('drop', (e) => handleFileDrop(e, fileDropArea, imageInput));
}

/** Handle drag over event. */
function handleDragOver(e) {
  e.preventDefault();
  e.currentTarget.classList.add('dragover');
}

/** Handle drag leave event. */
function handleDragLeave(e) {
  e.currentTarget.classList.remove('dragover');
}

/** Handle file drop event. */
function handleFileDrop(e, fileDropArea, imageInput) {
  e.preventDefault();
  fileDropArea.classList.remove('dragover');
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) {
    imageInput.files = e.dataTransfer.files;
    const removeImageBtn = document.getElementById('remove-image-btn');
    showImagePreview(file, fileDropArea, removeImageBtn);
  }
}

/** Set up file input change event. */
function setupFileInputChange(imageInput, fileDropArea, removeImageBtn) {
  imageInput.addEventListener('change', (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) {
      showImagePreview(file, fileDropArea, removeImageBtn);
    }
  });
}

/**
 * Set up form submission for detection.
 * Reduced to a single parameter object to avoid multiple arguments.
 */
function setupFormSubmission({
  form,
  imageInput,
  detectionError,
  detectionResult
}) {
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    clearMessages(detectionError, detectionResult);

    const model = document.getElementById('model-select').value;
    const file = imageInput.files[0];

    if (!file) {
      detectionError.textContent = 'Please select an image.';
      return;
    }

    await performDetection({
      file,
      model,
      detectionError,
      detectionResult
    });
  });
}

/** Clear error and result messages. */
function clearMessages(detectionError, detectionResult) {
  detectionError.textContent = '';
  detectionResult.textContent = '';
}

/** Remove the currently uploaded image and clear related content. */
function removeImage() {
  const canvas = document.getElementById('image-canvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const imageInput = document.getElementById('image-input');
  imageInput.value = '';

  const detectionResult = document.getElementById('detection-result');
  const detectionError = document.getElementById('detection-error');
  detectionResult.textContent = '';
  detectionError.textContent = '';

  const removeImageBtn = document.getElementById('remove-image-btn');
  removeImageBtn.style.display = 'none';
}

/** Display a preview of the uploaded image on the canvas. */
function showImagePreview(file, fileDropArea, removeImageBtn) {
  const reader = new FileReader();
  reader.onload = () => loadImagePreview(reader.result, fileDropArea, removeImageBtn);
  reader.readAsDataURL(file);
}

/** Load image preview once file is read. */
function loadImagePreview(imageSrc, fileDropArea, removeImageBtn) {
  const img = new Image();
  img.onload = () => {
    originalImageWidth = img.width;
    originalImageHeight = img.height;
    drawScaledImage(img, fileDropArea);
    removeImageBtn.style.display = 'inline-block';
  };
  img.src = imageSrc;
}

/** Draw the scaled image on the canvas. */
function drawScaledImage(img, fileDropArea) {
  const canvas = document.getElementById('image-canvas');
  const ctx = canvas.getContext('2d');

  const { width, height } = scaleDimension({
    width: img.width,
    height: img.height,
    maxWidth: fileDropArea.clientWidth - 40,
    maxHeight: fileDropArea.clientHeight - 40
  });
  canvas.width = width;
  canvas.height = height;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, width, height);
}

/**
 * Scale dimensions to maintain aspect ratio within maxWidth and maxHeight.
 * Wrapped into an object to reduce the function argument count.
 */
function scaleDimension({ width, height, maxWidth, maxHeight }) {
  let scaledWidth = width;
  let scaledHeight = height;

  if (scaledWidth > maxWidth) {
    const ratio = maxWidth / scaledWidth;
    scaledWidth = maxWidth;
    scaledHeight *= ratio;
  }
  if (scaledHeight > maxHeight) {
    const ratio = maxHeight / scaledHeight;
    scaledHeight = maxHeight;
    scaledWidth *= ratio;
  }

  return { width: scaledWidth, height: scaledHeight };
}

/**
 * Perform the detection request to the backend.
 * Merged arguments into a single object for readability.
 */
async function performDetection({ file, model, detectionError, detectionResult }) {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('model', model);

  try {
    const response = await fetch(`${API_URL}/detect`, {
      method: 'POST',
      headers: authHeaders(),
      body: formData
    });

    if (!response.ok) {
      const data = await response.json();
      detectionError.textContent = data.detail || 'Detection failed.';
      return;
    }

    const results = await response.json();
    drawDetectionResults(results);
    displayObjectCounts({ results, detectionResult });
  } catch (err) {
    console.error(err);
    detectionError.textContent = 'Error performing detection.';
  }
}

/** Draw detection results on the canvas. */
function drawDetectionResults(results) {
  const canvas = document.getElementById('image-canvas');
  const ctx = canvas.getContext('2d');

  results.forEach((detection) => {
    drawSingleDetection({ ctx, detection, canvas });
  });
}

/** Draw a single detection result. */
function drawSingleDetection({ ctx, detection, canvas }) {
  const names = [
    'Hardhat',
    'Mask',
    'NO-Hardhat',
    'NO-Mask',
    'NO-Safety Vest',
    'Person',
    'Safety Cone',
    'Safety Vest',
    'machinery',
    'vehicle'
  ];
  const colors = {
    'Hardhat': 'green',
    'Safety Vest': 'green',
    'machinery': 'yellow',
    'vehicle': 'yellow',
    'NO-Hardhat': 'red',
    'NO-Safety Vest': 'red',
    'Person': 'orange',
    'Safety Cone': 'pink'
  };

  const [x1, y1, x2, y2, , classId] = detection;
  const label = names[classId];
  const color = colors[label] || 'blue';

  const { scaledX1, scaledY1, scaledX2, scaledY2 } = scaleCoordinates({
    x1,
    y1,
    x2,
    y2,
    canvas
  });

  drawBoundingBox({
    ctx,
    coords: { x1: scaledX1, y1: scaledY1, x2: scaledX2, y2: scaledY2 },
    color
  });
  drawLabel({
    ctx,
    label,
    x: scaledX1,
    y: scaledY1,
    color
  });
}

/** Scale coordinates to fit the canvas. */
function scaleCoordinates({ x1, y1, x2, y2, canvas }) {
  const scaleX = canvas.width / originalImageWidth;
  const scaleY = canvas.height / originalImageHeight;

  return {
    scaledX1: x1 * scaleX,
    scaledY1: y1 * scaleY,
    scaledX2: x2 * scaleX,
    scaledY2: y2 * scaleY
  };
}

/** Draw bounding box around the detected object. */
function drawBoundingBox({ ctx, coords, color }) {
  const { x1, y1, x2, y2 } = coords;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
}

/** Draw label above the bounding box. */
function drawLabel({ ctx, label, x, y, color }) {
  ctx.fillStyle = color;
  ctx.fillRect(x, y - 20, ctx.measureText(label).width + 10, 20);

  ctx.fillStyle = 'black';
  ctx.font = '14px Arial';
  ctx.fillText(label, x + 5, y - 5);
}

/** Display counts of detected objects. */
function displayObjectCounts({ results, detectionResult }) {
  const names = [
    'Hardhat',
    'Mask',
    'NO-Hardhat',
    'NO-Mask',
    'NO-Safety Vest',
    'Person',
    'Safety Cone',
    'Safety Vest',
    'machinery',
    'vehicle'
  ];
  const counts = initializeCounts(names);

  countDetections(results, names, counts);
  showCounts({ detectionResult, counts });
}

/** Initialise counts for each label. */
function initializeCounts(names) {
  const counts = {};
  names.forEach(name => counts[name] = 0);
  return counts;
}

/** Count detections per label. */
function countDetections(results, names, counts) {
  results.forEach(([, , , , , classId]) => {
    const label = names[classId];
    if (label) {
      counts[label] += 1;
    }
  });
}

/** Show counts in the detection result area. */
function showCounts({ detectionResult, counts }) {
  detectionResult.textContent = Object.entries(counts)
    .filter(([_, count]) => count > 0)
    .map(([name, count]) => `${name}: ${count}`)
    .join('\n');
}
