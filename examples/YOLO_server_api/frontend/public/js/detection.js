import { checkAccess, authHeaders, showAppropriateLinks } from './common.js';
const API_URL = '/api'; // Base path for the backend API

// Wait until the DOM content is fully loaded
document.addEventListener('DOMContentLoaded', () => {
  // Check the user's access and show links based on their role
  checkAccess([]);
  showAppropriateLinks();

  // Select key DOM elements
  const logoutBtn = document.getElementById('logout-btn');
  const form = document.getElementById('detection-form');
  const detectionError = document.getElementById('detection-error');
  const detectionResult = document.getElementById('detection-result');
  const fileDropArea = document.getElementById('file-drop-area');
  const imageInput = document.getElementById('image-input');
  const removeImageBtn = document.getElementById('remove-image-btn');

  // Variables to store the original dimensions of the uploaded image
  let originalImageWidth = 0;
  let originalImageHeight = 0;

  // Function to remove the currently uploaded image and clear related content
  function removeImage() {
    const canvas = document.getElementById('image-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas
    imageInput.value = ''; // Reset the file input
    detectionResult.textContent = ''; // Clear the detection results
    detectionError.textContent = ''; // Clear any error messages
    removeImageBtn.style.display = 'none'; // Hide the "Remove Image" button
  }

  // Attach an event listener to the "Remove Image" button
  removeImageBtn.addEventListener('click', () => {
    removeImage();
  });

  // Allow users to click the "Choose File" button to open the file selector
  const chooseFileBtn = document.querySelector('.choose-file-btn');
  chooseFileBtn.addEventListener('click', (e) => {
    e.preventDefault(); // Prevent default behaviour
    imageInput.click(); // Trigger the file input
  });

  // Handle drag-and-drop events for the file drop area
  fileDropArea.addEventListener('dragover', (e) => {
    e.preventDefault(); // Prevent the default drag behaviour
    fileDropArea.classList.add('dragover'); // Highlight the drop area
  });

  fileDropArea.addEventListener('dragleave', () => {
    fileDropArea.classList.remove('dragover'); // Remove the highlight
  });

  fileDropArea.addEventListener('drop', (e) => {
    e.preventDefault(); // Prevent the default drop behaviour
    fileDropArea.classList.remove('dragover'); // Remove the highlight
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      imageInput.files = e.dataTransfer.files; // Set the dropped file to the input
      showImagePreview(file); // Display the preview
    }
  });

  // Handle file input changes
  imageInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      showImagePreview(e.target.files[0]); // Display the preview
    }
  });

  // Function to display a preview of the uploaded image on the canvas
  function showImagePreview(file) {
    const reader = new FileReader();
    reader.onload = () => {
      const canvas = document.getElementById('image-canvas');
      const ctx = canvas.getContext('2d');
      const img = new Image();
      img.onload = () => {
        originalImageWidth = img.width;
        originalImageHeight = img.height;

        // Scale the image to fit within the drop area
        const maxWidth = fileDropArea.clientWidth - 40; // Allow for padding
        const maxHeight = fileDropArea.clientHeight - 40;
        let { width, height } = img;

        // Adjust width and height to maintain the aspect ratio
        if (width > maxWidth) {
          const ratio = maxWidth / width;
          width = maxWidth;
          height *= ratio;
        }
        if (height > maxHeight) {
          const ratio = maxHeight / height;
          height = maxHeight;
          width *= ratio;
        }

        // Set canvas dimensions and draw the image
        canvas.width = width;
        canvas.height = height;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, width, height);
        removeImageBtn.style.display = 'inline-block'; // Show the "Remove Image" button
      };
      img.src = reader.result;
    };
    reader.readAsDataURL(file); // Read the file as a data URL
  }

  // Handle form submission for detection requests
  form.addEventListener('submit', async (e) => {
    e.preventDefault(); // Prevent the default form submission
    detectionError.textContent = ''; // Clear any previous errors
    detectionResult.textContent = ''; // Clear any previous results

    const model = document.getElementById('model-select').value; // Get the selected model
    const file = imageInput.files[0]; // Get the uploaded file
    if (!file) {
      detectionError.textContent = 'Please select an image.'; // Display an error if no file is selected
      return;
    }

    const formData = new FormData(); // Create a FormData object
    formData.append('image', file); // Append the image file
    formData.append('model', model); // Append the selected model

    try {
      const response = await fetch(`${API_URL}/detect`, {
        method: 'POST',
        headers: authHeaders(), // Add authentication headers
        body: formData // Send the form data
      });

      if (!response.ok) {
        const data = await response.json();
        detectionError.textContent = data.detail || 'Detection failed.'; // Display an error message
        return;
      }

      const results = await response.json();
      drawDetectionResults(results); // Draw the detection results on the image
      displayObjectCounts(results); // Display the counts of detected objects
    } catch (err) {
      console.error(err);
      detectionError.textContent = 'Error performing detection.'; // Display a generic error message
    }
  });

  // Draw detection results on the canvas
  function drawDetectionResults(results) {
    const canvas = document.getElementById('image-canvas');
    const ctx = canvas.getContext('2d');

    const names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'];
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

    results.forEach(([x1, y1, x2, y2, confidence, classId]) => {
      const label = names[classId];
      const color = colors[label] || 'blue';

      // Scale the bounding box coordinates to fit the canvas
      const scaleX = canvas.width / originalImageWidth;
      const scaleY = canvas.height / originalImageHeight;
      const scaledX1 = x1 * scaleX;
      const scaledY1 = y1 * scaleY;
      const scaledX2 = x2 * scaleX;
      const scaledY2 = y2 * scaleY;

      // Draw the bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(scaledX1, scaledY1, scaledX2 - scaledX1, scaledY2 - scaledY1);

      // Draw the label background
      ctx.fillStyle = color;
      ctx.fillRect(scaledX1, scaledY1 - 20, ctx.measureText(label).width + 10, 20);

      // Draw the label text
      ctx.fillStyle = 'black';
      ctx.font = '14px Arial';
      ctx.fillText(label, scaledX1 + 5, scaledY1 - 5); // Display the object label
    });
  }

  // Display the counts of detected objects
  function displayObjectCounts(results) {
    const counts = {};
    const names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'];
    names.forEach(name => counts[name] = 0); // Initialise counts for all object names

    results.forEach(([x1, y1, x2, y2, confidence, classId]) => {
      const label = names[classId];
      if (label) counts[label] += 1; // Increment the count for the detected object
    });

    // Display the object counts
    detectionResult.textContent = Object.entries(counts)
      .filter(([_, count]) => count > 0) // Only show objects that were detected
      .map(([name, count]) => `${name}: ${count}`)
      .join('\n');
  }
});
