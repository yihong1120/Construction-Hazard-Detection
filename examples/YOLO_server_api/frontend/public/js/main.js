import { getUsernameFromToken, getUserRoleFromToken, getToken, clearToken, checkAccess } from './common.js';

// Wait for the DOM content to be fully loaded
document.addEventListener('DOMContentLoaded', () => {
  // Check access permissions
  // No specific role is required, meaning any logged-in user can access this page
  checkAccess([]);

  // Handle the Logout button functionality
  const logoutBtn = document.getElementById('logout-btn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', () => {
      clearToken(); // Clear the stored token
      window.location.href = './login.html'; // Redirect the user to the login page
    });
  }

  // Retrieve the user's token from localStorage
  const token = getToken();

  // Display user information or a message if the user is not logged in
  const userInfoDiv = document.getElementById('user-info');
  if (token) {
    // If a token exists, decode it to retrieve the username and role
    const username = getUsernameFromToken();
    const role = getUserRoleFromToken();

    // Display the user's username and role
    userInfoDiv.textContent = `Logged in as ${username} (Role: ${role})`;
  } else {
    // Display a message indicating the user is not logged in
    userInfoDiv.textContent = 'You are not logged in.';
  }
});
