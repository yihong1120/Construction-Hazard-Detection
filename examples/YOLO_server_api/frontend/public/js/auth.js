import { setToken, clearToken } from './common.js';
const API_URL = '/api'; // Base path for the backend API

// Wait until the DOM content is fully loaded
document.addEventListener('DOMContentLoaded', () => {
  // Select the login form and error message elements
  const form = document.getElementById('login-form');
  const errorMsg = document.getElementById('login-error');

  // Attach an event listener to handle form submission
  form.addEventListener('submit', async (e) => {
    e.preventDefault(); // Prevent the default form submission behaviour

    // Retrieve the username and password from the form inputs
    const username = document.getElementById('username').value.trim();
    const password = document.getElementById('password').value.trim();

    try {
      // Clear any existing tokens before making a new login request
      clearToken();

      // Send a POST request to the API with the username and password
      const response = await fetch(`${API_URL}/token`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }, // Set the request header for JSON
        body: JSON.stringify({ username, password }) // Convert credentials to a JSON string
      });

      // Check if the response indicates an unsuccessful login
      if (!response.ok) {
        const data = await response.json();
        errorMsg.textContent = data.detail || 'Login failed'; // Display the error message
        return; // Exit the function
      }

      // Parse the response JSON to retrieve the token
      const data = await response.json();
      setToken(data.access_token); // Store the token in localStorage

      // Redirect to the homepage after successful login
      window.location.href = './index.html';
    } catch (err) {
      // Handle network or other unexpected errors
      errorMsg.textContent = 'Error logging in.';
    }
  });
});
