import { checkAccess, authHeaders, clearToken } from './common.js';
const API_URL = '/api'; // Base path for the backend API, ensure the server has the corresponding routes

// Wait for the DOM content to be fully loaded
document.addEventListener('DOMContentLoaded', () => {
  // Restrict access to users with the 'admin' role
  checkAccess(['admin']);

  // Bind the Logout button functionality
  const logoutBtn = document.getElementById('logout-btn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', () => {
      clearToken(); // Clear the stored token
      window.location.href = '/login.html'; // Redirect the user to the login page
    });
  }

  // Handle the Add User form submission
  const addUserForm = document.getElementById('add-user-form');
  const addUserError = document.getElementById('add-user-error');
  addUserForm.addEventListener('submit', async (e) => {
    e.preventDefault(); // Prevent the default form submission behaviour
    addUserError.textContent = ''; // Clear any previous error messages

    // Retrieve the input values from the form
    const username = document.getElementById('add-username').value.trim();
    const password = document.getElementById('add-password').value.trim();
    const role = document.getElementById('add-role').value;

    try {
      // Send a POST request to add a new user
      const res = await fetch(`${API_URL}/add_user`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...authHeaders() // Include the authentication headers
        },
        body: JSON.stringify({ username, password, role }) // Send the user data as JSON
      });

      if (!res.ok) {
        const data = await res.json();
        addUserError.textContent = data.detail || 'Failed to add user.'; // Show an error message
        return;
      }

      alert('User added successfully.'); // Show a success message
    } catch (err) {
      addUserError.textContent = 'Error adding user.'; // Show a generic error message
    }
  });

  // Handle the Delete User form submission
  const deleteUserForm = document.getElementById('delete-user-form');
  const deleteUserError = document.getElementById('delete-user-error');
  deleteUserForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    deleteUserError.textContent = ''; // Clear any previous error messages

    // Retrieve the username to delete
    const username = document.getElementById('delete-username').value.trim();

    try {
      // Send a POST request to delete the user
      const res = await fetch(`${API_URL}/delete_user`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...authHeaders() // Include the authentication headers
        },
        body: JSON.stringify({ username }) // Send the username as JSON
      });

      if (!res.ok) {
        const data = await res.json();
        deleteUserError.textContent = data.detail || 'Failed to delete user.'; // Show an error message
        return;
      }

      alert('User deleted successfully.'); // Show a success message
    } catch (err) {
      deleteUserError.textContent = 'Error deleting user.'; // Show a generic error message
    }
  });

  // Handle the Update Username form submission
  const updateUsernameForm = document.getElementById('update-username-form');
  const updateUsernameError = document.getElementById('update-username-error');
  updateUsernameForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    updateUsernameError.textContent = ''; // Clear any previous error messages

    // Retrieve the old and new usernames
    const old_username = document.getElementById('old-username').value.trim();
    const new_username = document.getElementById('new-username').value.trim();

    try {
      // Send a PUT request to update the username
      const res = await fetch(`${API_URL}/update_username`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          ...authHeaders() // Include the authentication headers
        },
        body: JSON.stringify({ old_username, new_username }) // Send the old and new usernames as JSON
      });

      if (!res.ok) {
        const data = await res.json();
        updateUsernameError.textContent = data.detail || 'Failed to update username.'; // Show an error message
        return;
      }

      alert('Username updated successfully.'); // Show a success message
    } catch (err) {
      updateUsernameError.textContent = 'Error updating username.'; // Show a generic error message
    }
  });

  // Handle the Update Password form submission
  const updatePasswordForm = document.getElementById('update-password-form');
  const updatePasswordError = document.getElementById('update-password-error');
  updatePasswordForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    updatePasswordError.textContent = ''; // Clear any previous error messages

    // Retrieve the username and new password
    const username = document.getElementById('update-password-username').value.trim();
    const new_password = document.getElementById('new-password').value.trim();

    try {
      // Send a PUT request to update the user's password
      const res = await fetch(`${API_URL}/update_password`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          ...authHeaders() // Include the authentication headers
        },
        body: JSON.stringify({ username, new_password }) // Send the username and new password as JSON
      });

      if (!res.ok) {
        const data = await res.json();
        updatePasswordError.textContent = data.detail || 'Failed to update password.'; // Show an error message
        return;
      }

      alert('Password updated successfully.'); // Show a success message
    } catch (err) {
      updatePasswordError.textContent = 'Error updating password.'; // Show a generic error message
    }
  });

  // Handle the Set Active Status form submission
  const setActiveStatusForm = document.getElementById('set-active-status-form');
  const setActiveStatusError = document.getElementById('set-active-status-error');
  setActiveStatusForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    setActiveStatusError.textContent = ''; // Clear any previous error messages

    // Retrieve the username and active status
    const username = document.getElementById('active-status-username').value.trim();
    const is_active = document.getElementById('is-active').checked;

    try {
      // Send a PUT request to update the user's active status
      const res = await fetch(`${API_URL}/set_user_active_status`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          ...authHeaders() // Include the authentication headers
        },
        body: JSON.stringify({ username, is_active }) // Send the username and active status as JSON
      });

      if (!res.ok) {
        const data = await res.json();
        setActiveStatusError.textContent = data.detail || 'Failed to update active status.'; // Show an error message
        return;
      }

      alert('User active status updated successfully.'); // Show a success message
    } catch (err) {
      setActiveStatusError.textContent = 'Error updating active status.'; // Show a generic error message
    }
  });
});
