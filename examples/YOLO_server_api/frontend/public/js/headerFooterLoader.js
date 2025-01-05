document.addEventListener('DOMContentLoaded', async () => {
  const headerContainer = document.getElementById('header-container');
  const footerContainer = document.getElementById('footer-container');

  if (headerContainer) {
    await loadHeader(headerContainer);
  }

  if (footerContainer) {
    await loadFooter(footerContainer);
  }
});

/**
 * Load the header content into the header container.
 * @param {HTMLElement} headerContainer - The container to load the header into.
 */
async function loadHeader(headerContainer) {
  try {
    const response = await fetch('/header.html'); // Fetch the header HTML content
    const html = await response.text(); // Parse the response as text
    headerContainer.innerHTML = html; // Inject the HTML into the container
    await initialiseHeader(); // Initialise header-specific behaviours
  } catch (err) {
    // Removed logError call
    // Optionally handle the error here, e.g., display a message to the user
  }
}

/**
 * Initialise header-specific behaviours.
 */
async function initialiseHeader() {
  await bindLogoutEvent(); // Bind the logout button event
  bindMenuToggleEvent(); // Bind the menu toggle event
}

/**
 * Bind the logout button event.
 */
async function bindLogoutEvent() {
  const logoutBtn = document.getElementById('logout-btn'); // Reference the logout button
  if (!logoutBtn) return;

  try {
    const module = await import('/js/common.js'); // Dynamically import the `common.js` module
    const { clearToken } = module; // Extract the clearToken function
    logoutBtn.addEventListener('click', () => {
      clearToken(); // Clear authentication tokens
      window.location.href = '/login.html'; // Redirect to the login page
    });
  } catch (err) {
    // Removed logError call
    // Optionally handle the error here, e.g., display a message to the user
  }
}

/**
 * Bind the menu toggle event for mobile navigation.
 */
function bindMenuToggleEvent() {
  const menuToggle = document.getElementById('menu-toggle'); // Reference the menu toggle button
  const navLinks = document.getElementById('nav-links'); // Reference the navigation links container
  if (menuToggle && navLinks) {
    menuToggle.addEventListener('click', () => {
      navLinks.classList.toggle('expanded'); // Toggle the expanded class
    });
  }
}

/**
 * Load the footer content into the footer container.
 * @param {HTMLElement} footerContainer - The container to load the footer into.
 */
async function loadFooter(footerContainer) {
  try {
    const response = await fetch('/footer.html'); // Fetch the footer HTML content
    const html = await response.text(); // Parse the response as text
    footerContainer.innerHTML = html; // Inject the HTML into the container
    updateFooterYear(); // Update the footer with the current year
  } catch (err) {
    // Removed logError call
    // Optionally handle the error here, e.g., display a message to the user
  }
}

/**
 * Update the footer with the current year.
 */
function updateFooterYear() {
  const yearSpan = document.getElementById('current-year'); // Reference the year span
  if (yearSpan) {
    yearSpan.textContent = new Date().getFullYear(); // Set the text content to the current year
  }
}
