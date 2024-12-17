import { checkAccess, authHeaders, clearToken } from './common.js';
const API_URL = '/api'; // Base path for the backend API

// Wait for the DOM content to be fully loaded
document.addEventListener('DOMContentLoaded', () => {
  // Restrict access to users with the roles 'admin' or 'model_manager'
  checkAccess(['admin', 'model_manager']);

  // Handle the model file update form submission
  const modelFileUpdateForm = document.getElementById('model-file-update-form');
  const modelUpdateError = document.getElementById('model-update-error');
  modelFileUpdateForm.addEventListener('submit', async (e) => {
    e.preventDefault(); // Prevent the default form submission behaviour
    modelUpdateError.textContent = ''; // Clear any previous error messages

    // Retrieve the model name and file from the form
    const model = document.getElementById('model-name').value.trim();
    const fileInput = document.getElementById('model-file');
    const file = fileInput.files[0];
    if (!file) {
      modelUpdateError.textContent = 'Please select a model file.'; // Show an error if no file is selected
      return;
    }

    // Create a FormData object to hold the form data
    const formData = new FormData();
    formData.append('model', model);
    formData.append('file', file);

    try {
      // Send a POST request to the API to update the model file
      const res = await fetch(`${API_URL}/model_file_update`, {
        method: 'POST',
        headers: authHeaders(), // Include the authentication headers
        body: formData // Send the FormData object as the request body
      });

      if (!res.ok) {
        const data = await res.json();
        modelUpdateError.textContent = data.detail || 'Failed to update model file.'; // Show an error message
        return;
      }

      // Show a success message upon successful model file update
      alert('Model file updated successfully.');
    } catch (err) {
      // Handle any unexpected errors
      modelUpdateError.textContent = 'Error updating model file.';
    }
  });

  // Handle the form for retrieving a new model file
  const getNewModelForm = document.getElementById('get-new-model-form');
  const getNewModelError = document.getElementById('get-new-model-error');
  const modelFileContent = document.getElementById('model-file-content');
  getNewModelForm.addEventListener('submit', async (e) => {
    e.preventDefault(); // Prevent the default form submission behaviour
    getNewModelError.textContent = ''; // Clear any previous error messages
    modelFileContent.textContent = ''; // Clear any previous content

    // Retrieve the model name from the form
    const model = document.getElementById('get-model-name').value.trim();
    const lastUpdateTime = '1970-01-01'; // Automatically set the default last update time

    try {
      // Send a POST request to the API to retrieve a new model file
      const res = await fetch(`${API_URL}/get_new_model`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...authHeaders() // Include the authentication headers
        },
        body: JSON.stringify({ model, last_update_time: lastUpdateTime }) // Send the model name and last update time as JSON
      });

      if (!res.ok) {
        const data = await res.json();
        getNewModelError.textContent = data.detail || 'Failed to retrieve new model.'; // Show an error message
        return;
      }

      // Parse the response data
      const data = await res.json();
      if (data.model_file) {
        // Display the Base64-encoded model file content
        modelFileContent.textContent = `Base64 Model File: ${data.model_file}`;
      } else {
        // Display any returned message if no file is included
        modelFileContent.textContent = data.message;
      }
    } catch (err) {
      // Handle any unexpected errors
      getNewModelError.textContent = 'Error retrieving model file.';
    }
  });
});
