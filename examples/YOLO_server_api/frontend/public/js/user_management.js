import { checkAccess, authHeaders, clearToken } from './common.js';

const API_URL = '/api'; // Base path for the backend API

document.addEventListener('DOMContentLoaded', () => {
  checkAccess(['admin']); // Restrict access to admin users
  setupLogoutButton();
  setupFormHandlers();
});

/**
 * Bind the Logout button functionality.
 */
function setupLogoutButton() {
  const logoutBtn = document.getElementById('logout-btn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', () => {
      clearToken(); // Clear the stored token
      window.location.href = '/login.html'; // Redirect the user to the login page
    });
  }
}

/**
 * Set up all form submission handlers.
 */
function setupFormHandlers() {
  setupFormHandler('add-user-form', handleAddUser, 'add-user-error');
  setupFormHandler('delete-user-form', handleDeleteUser, 'delete-user-error');
  setupFormHandler('update-username-form', handleUpdateUsername, 'update-username-error');
  setupFormHandler('update-password-form', handleUpdatePassword, 'update-password-error');
  setupFormHandler('set-active-status-form', handleSetActiveStatus, 'set-active-status-error');
}

/**
 * Generic function to set up form submission handlers.
 */
function setupFormHandler(formId, handlerFunction, errorElementId) {
  // Retrieve the form and error element references
  const form = document.getElementById(formId);
  const errorElement = document.getElementById(errorElementId);

  // Return early if the form does not exist to reduce nesting
  if (!form) return;

  // Attach the submit event listener
  form.addEventListener('submit', handleSubmit);

  /**
   * Handle the form submission.
   * Using a separate function helps keep nesting shallow.
   */
  async function handleSubmit(e) {
    e.preventDefault();
    errorElement.textContent = ''; // Clear any previous error messages

    try {
      await handlerFunction();
    } catch (err) {
      errorElement.textContent = 'An error occurred while processing the form.';
    }
  }
}

/**
 * Handle the Add User form submission.
 */
async function handleAddUser() {
  const username = document.getElementById('add-username').value.trim();
  const password = document.getElementById('add-password').value.trim();
  const role = document.getElementById('add-role').value;

  await sendRequest(`${API_URL}/add_user`, 'POST', { username, password, role });
  alert('User added successfully.');
}

/**
 * Handle the Delete User form submission.
 */
async function handleDeleteUser() {
  const username = document.getElementById('delete-username').value.trim();
  await sendRequest(`${API_URL}/delete_user`, 'POST', { username });
  alert('User deleted successfully.');
}

/**
 * Handle the Update Username form submission.
 */
async function handleUpdateUsername() {
  const old_username = document.getElementById('old-username').value.trim();
  const new_username = document.getElementById('new-username').value.trim();
  await sendRequest(`${API_URL}/update_username`, 'PUT', { old_username, new_username });
  alert('Username updated successfully.');
}

/**
 * Handle the Update Password form submission.
 */
async function handleUpdatePassword() {
  const username = document.getElementById('update-password-username').value.trim();
  const new_password = document.getElementById('new-password').value.trim();
  await sendRequest(`${API_URL}/update_password`, 'PUT', { username, new_password });
  alert('Password updated successfully.');
}

/**
 * Handle the Set Active Status form submission.
 */
async function handleSetActiveStatus() {
  const username = document.getElementById('active-status-username').value.trim();
  const is_active = document.getElementById('is-active').checked;
  await sendRequest(`${API_URL}/set_user_active_status`, 'PUT', { username, is_active });
  alert('User active status updated successfully.');
}

/**
 * Helper function to send API requests.
 */
async function sendRequest(url, method, body) {
  const res = await fetch(url, {
    method,
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(),
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const data = await res.json();
    throw new Error(data.detail || 'Request failed.');
  }
}
