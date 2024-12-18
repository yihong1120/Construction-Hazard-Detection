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

async function loadHeader(headerContainer) {
  try {
    const response = await fetch('/header.html');
    const html = await response.text();
    headerContainer.innerHTML = html;
    bindHeaderEvents();
  } catch (err) {
    console.error('Error loading header:', err);
  }
}

function bindHeaderEvents() {
  const logoutBtn = document.getElementById('logout-btn');
  if (logoutBtn) {
    import('/js/common.js').then(module => {
      const { clearToken } = module;
      logoutBtn.addEventListener('click', () => {
        clearToken();
        window.location.href = '/login.html';
      });
    });
  }

  const menuToggle = document.getElementById('menu-toggle');
  const navLinks = document.getElementById('nav-links');
  if (menuToggle && navLinks) {
    menuToggle.addEventListener('click', () => {
      navLinks.classList.toggle('expanded');
    });
  }
}

async function loadFooter(footerContainer) {
  try {
    const response = await fetch('/footer.html');
    const html = await response.text();
    footerContainer.innerHTML = html;
    updateFooterYear();
  } catch (err) {
    console.error('Error loading footer:', err);
  }
}

function updateFooterYear() {
  const yearSpan = document.getElementById('current-year');
  if (yearSpan) {
    yearSpan.textContent = new Date().getFullYear();
  }
}
