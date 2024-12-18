import { checkAccess, authHeaders, clearToken } from './common.js';
const API_URL = '/api'; // Base path for the backend API

document.addEventListener('DOMContentLoaded', () => {
  checkAccess(['admin', 'model_manager']); // Restrict access to specific roles
  setupModelFileUpdate();
  setupGetNewModel();
});

/** Setup for model file update form submission */
function setupModelFileUpdate() {
  const modelFileUpdateForm = document.getElementById('model-file-update-form');
  const modelUpdateError = document.getElementById('model-update-error');

  if (!modelFileUpdateForm) return;

  modelFileUpdateForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    modelUpdateError.textContent = ''; // Clear any previous error messages

    const formData = gatherModelFileUpdateData(modelUpdateError);
    if (!formData) return;

    try {
      await sendModelFileUpdateRequest(formData, modelUpdateError);
    } catch (err) {
      modelUpdateError.textContent = 'Error updating model file.';
    }
  });
}

/** Gather data for model file update */
function gatherModelFileUpdateData(errorElement) {
  const model = document.getElementById('model-name').value.trim();
  const fileInput = document.getElementById('model-file');
  const file = fileInput.files[0];

  if (!file) {
    errorElement.textContent = 'Please select a model file.';
    return null;
  }

  const formData = new FormData();
  formData.append('model', model);
  formData.append('file', file);
  return formData;
}

/** Send the model file update request */
async function sendModelFileUpdateRequest(formData, errorElement) {
  const response = await fetch(`${API_URL}/model_file_update`, {
    method: 'POST',
    headers: authHeaders(),
    body: formData,
  });

  if (!response.ok) {
    const data = await response.json();
    errorElement.textContent = data.detail || 'Failed to update model file.';
    return;
  }

  alert('Model file updated successfully.');
}

/** Setup for retrieving a new model file */
function setupGetNewModel() {
  const getNewModelForm = document.getElementById('get-new-model-form');
  const getNewModelError = document.getElementById('get-new-model-error');
  const modelFileContent = document.getElementById('model-file-content');

  if (!getNewModelForm) return;

  getNewModelForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    clearGetNewModelMessages(getNewModelError, modelFileContent);

    const requestData = gatherGetNewModelData();
    if (!requestData) return;

    try {
      await sendGetNewModelRequest(requestData, getNewModelError, modelFileContent);
    } catch (err) {
      getNewModelError.textContent = 'Error retrieving model file.';
    }
  });
}

/** Clear messages for the "Get New Model" form */
function clearGetNewModelMessages(errorElement, contentElement) {
  errorElement.textContent = '';
  contentElement.textContent = '';
}

/** Gather data for retrieving a new model file */
function gatherGetNewModelData() {
  const model = document.getElementById('get-model-name').value.trim();
  const lastUpdateTime = '1970-01-01'; // Default last update time
  return { model, last_update_time: lastUpdateTime };
}

/** Send the request to retrieve a new model file */
async function sendGetNewModelRequest(requestData, errorElement, contentElement) {
  const response = await fetch(`${API_URL}/get_new_model`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(),
    },
    body: JSON.stringify(requestData),
  });

  if (!response.ok) {
    const data = await response.json();
    errorElement.textContent = data.detail || 'Failed to retrieve new model.';
    return;
  }

  const data = await response.json();
  displayModelFileContent(data, contentElement);
}

/** Display the retrieved model file content */
function displayModelFileContent(data, contentElement) {
  if (data.model_file) {
    contentElement.textContent = `Base64 Model File: ${data.model_file}`;
  } else {
    contentElement.textContent = data.message || 'No model file available.';
  }
}
