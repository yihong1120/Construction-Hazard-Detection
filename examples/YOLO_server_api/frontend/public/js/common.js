// Retrieve the Token from localStorage
export function getToken() {
  return localStorage.getItem('access_token');
}

// Store the Token in localStorage
export function setToken(token) {
  localStorage.setItem('access_token', token);
}

// Remove the Token from localStorage
export function clearToken() {
  localStorage.removeItem('access_token');
}

// Check if the Token has expired
function isTokenExpired(token) {
  if (!token) return true; // Token is invalid or does not exist
  try {
    const payload = JSON.parse(atob(token.split('.')[1])); // Decode the Token's payload
    const now = Math.floor(Date.now() / 1000); // Current timestamp in seconds
    return payload.exp && payload.exp < now; // Compare the expiration time with the current time
  } catch (e) {
    logError('Error parsing token:', e); // Use custom logging
    return true; // Assume expired if there's a parsing error
  }
}

// Retrieve the user role from the Token
export function getUserRoleFromToken() {
  const token = getToken();
  if (!token) return null; // Return null if no Token exists
  if (isTokenExpired(token)) return null; // Return null if the Token is expired
  try {
    const payload = JSON.parse(atob(token.split('.')[1])); // Decode the Token's payload
    // Ensure the payload structure is correct and retrieve the role
    return payload?.subject?.role || null;
  } catch (e) {
    logError('Error parsing token payload:', e); // Use custom logging
    return null;
  }
}

// Retrieve the username from the Token
export function getUsernameFromToken() {
  const token = getToken();
  if (!token) return null; // Return null if no Token exists
  if (isTokenExpired(token)) return null; // Return null if the Token is expired
  try {
    const payload = JSON.parse(atob(token.split('.')[1])); // Decode the Token's payload
    // Ensure the payload structure is correct and retrieve the username
    return payload?.subject?.username || null;
  } catch (e) {
    logError('Error parsing token payload:', e); // Use custom logging
    return null;
  }
}

// Return the headers required for authorised requests
export function authHeaders() {
  const token = getToken();
  if (!token || isTokenExpired(token)) {
    return {}; // Return empty headers if the Token is invalid or expired
  }
  return { Authorization: `Bearer ${token}` }; // Return the Bearer token header
}

// Check the user's access permissions
export function checkAccess(requiredRoles = []) {
  const token = getToken();
  if (!token || isTokenExpired(token)) {
    // Redirect to the login page if the Token is missing or expired
    window.location.href = './login.html';
    return;
  }

  const role = getUserRoleFromToken();
  if (requiredRoles.length > 0 && !requiredRoles.includes(role)) {
    // Redirect to the homepage if the user lacks the required role
    window.location.href = './index.html';
  }
}

// Display or hide links based on the user's role
export function showAppropriateLinks() {
  const role = getUserRoleFromToken();
  if (!role) return; // No role, no adjustments needed

  // Display admin-specific links
  if (role === 'admin') {
    document.querySelectorAll('.admin-only').forEach((el) => {
      el.style.display = 'inline-block';
    });
    document.querySelectorAll('.model-manager-only').forEach((el) => {
      el.style.display = 'inline-block';
    });
  }
  // Display model_manager-specific links
  else if (role === 'model_manager') {
    document.querySelectorAll('.model-manager-only').forEach((el) => {
      el.style.display = 'inline-block';
    });
  }
  // Hide privileged links for other roles or guests
  else {
    document.querySelectorAll('.admin-only, .model-manager-only').forEach((el) => {
      el.style.display = 'none';
    });
  }
}

/**
 * Custom error logging function.
 * Logs messages only in development environments.
 * @param {string} _message - The error message.
 * @param {Error} [_error] - The optional error object.
 */
function logError(_message, _error) {
  // Example: Send info to external logging service or remove for production
  // console.error(_message, _error);
}
