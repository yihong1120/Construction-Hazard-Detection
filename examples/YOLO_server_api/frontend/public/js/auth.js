import { setToken, clearToken } from './common.js';
const API_URL = '/api'; // Base path for the backend API

// Wait until the DOM content is fully loaded
document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('login-form');
  const errorMsg = document.getElementById('login-error');

  form.addEventListener('submit', (e) => handleLoginFormSubmit(e, errorMsg));
});

/**
 * Handles the login form submission.
 *
 * @param {Event} e - The form submit event.
 * @param {HTMLElement} errorMsg - The error message element.
 */
async function handleLoginFormSubmit(e, errorMsg) {
  e.preventDefault(); // Prevent the default form submission behaviour

  const { username, password } = getFormCredentials();
  if (!username || !password) {
    displayError(errorMsg, 'Username and password are required.');
    return;
  }

  try {
    await login(username, password);
    redirectToHomePage();
  } catch (err) {
    displayError(errorMsg, err.message || 'Error logging in.');
  }
}

/**
 * Retrieves the username and password from the form inputs.
 *
 * @returns {Object} An object containing username and password.
 */
function getFormCredentials() {
  const username = document.getElementById('username').value.trim();
  const password = document.getElementById('password').value.trim();
  return { username, password };
}

/**
 * Logs in the user by sending a POST request to the API.
 *
 * @param {string} username - The username.
 * @param {string} password - The password.
 * @throws Will throw an error if the login fails.
 */
async function login(username, password) {
  clearToken(); // Clear any existing tokens before making a new login request

  const response = await fetch(`${API_URL}/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password })
  });

  if (!response.ok) {
    const data = await response.json();
    throw new Error(data.detail || 'Login failed.');
  }

  const data = await response.json();
  setToken(data.access_token); // Store the token in localStorage
}

/**
 * Redirects the user to the homepage.
 */
function redirectToHomePage() {
  window.location.href = './index.html';
}

/**
 * Displays an error message in the specified element.
 *
 * @param {HTMLElement} element - The element to display the error message in.
 * @param {string} message - The error message to display.
 */
function displayError(element, message) {
  element.textContent = message;
}
