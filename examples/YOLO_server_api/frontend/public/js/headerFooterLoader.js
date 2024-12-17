// Wait for the DOM content to be fully loaded
document.addEventListener('DOMContentLoaded', () => {
  // Select the containers for the header and footer
  const headerContainer = document.getElementById('header-container');
  const footerContainer = document.getElementById('footer-container');

  // Load the Header
  if (headerContainer) {
    fetch('/header.html') // Fetch the header HTML file
      .then(response => response.text()) // Convert the response to text
      .then(html => {
        headerContainer.innerHTML = html; // Insert the header content into the container

        // Bind the Logout button event
        const logoutBtn = document.getElementById('logout-btn');
        if (logoutBtn) {
          // Dynamically import the common.js module to access clearToken
          import('/js/common.js').then(module => {
            const { clearToken } = module;
            logoutBtn.addEventListener('click', () => {
              clearToken(); // Clear the stored token
              window.location.href = '/login.html'; // Redirect to the login page
            });
          });
        }

        // Bind the Menu Toggle button for mobile navigation
        const menuToggle = document.getElementById('menu-toggle');
        const navLinks = document.getElementById('nav-links');
        if (menuToggle && navLinks) {
          // Toggle the visibility of navigation links when the button is clicked
          menuToggle.addEventListener('click', () => {
            navLinks.classList.toggle('expanded'); // Add or remove the 'expanded' class
          });
        }
      })
      .catch(err => console.error('Error loading header:', err)); // Log any errors that occur while fetching the header
  }

  // Load the Footer
  if (footerContainer) {
    fetch('/footer.html') // Fetch the footer HTML file
      .then(response => response.text()) // Convert the response to text
      .then(html => {
        footerContainer.innerHTML = html; // Insert the footer content into the container

        // Dynamically set the current year in the footer
        const yearSpan = document.getElementById('current-year');
        if (yearSpan) {
          yearSpan.textContent = new Date().getFullYear(); // Get and display the current year
        }
      })
      .catch(err => console.error('Error loading footer:', err)); // Log any errors that occur while fetching the footer
  }
});
