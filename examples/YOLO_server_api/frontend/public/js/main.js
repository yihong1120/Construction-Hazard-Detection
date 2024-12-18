import { getUsernameFromToken, getUserRoleFromToken, getToken, clearToken, checkAccess } from './common.js';

document.addEventListener('DOMContentLoaded', () => {
  // Initialize the page functionalities
  initAccessCheck();
  initLogoutButton();
  displayUserInfo();
});

/**
 * Check user access permissions.
 */
function initAccessCheck() {
  // No specific role is required, meaning any logged-in user can access this page
  checkAccess([]);
}

/**
 * Set up the logout button functionality.
 */
function initLogoutButton() {
  const logoutBtn = document.getElementById('logout-btn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', handleLogout);
  }
}

/**
 * Handle logout functionality.
 */
function handleLogout() {
  clearToken(); // Clear the stored token
  window.location.href = './login.html'; // Redirect the user to the login page
}

/**
 * Display the user's information or a message if the user is not logged in.
 */
function displayUserInfo() {
  const token = getToken();
  const userInfoDiv = document.getElementById('user-info');

  if (token) {
    const username = getUsernameFromToken();
    const role = getUserRoleFromToken();
    userInfoDiv.textContent = `Logged in as ${username} (Role: ${role})`;
  } else {
    userInfoDiv.textContent = 'You are not logged in.';
  }
}
