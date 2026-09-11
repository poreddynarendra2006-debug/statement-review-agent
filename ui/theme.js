/**
 * FinSight AI - Global Theme Controller (Dark / Light Mode)
 */

(function () {
  // Apply saved theme immediately before DOM paint to prevent flash
  const savedTheme = localStorage.getItem('finsight_theme') || 'dark';
  if (savedTheme === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
  } else {
    document.documentElement.removeAttribute('data-theme');
  }
})();

document.addEventListener('DOMContentLoaded', () => {
  setupThemeToggle();
});

function setupThemeToggle() {
  const toggleBtn = document.getElementById('themeToggleBtn');
  if (!toggleBtn) return;

  updateToggleIcon();

  toggleBtn.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';

    if (newTheme === 'light') {
      document.documentElement.setAttribute('data-theme', 'light');
    } else {
      document.documentElement.removeAttribute('data-theme');
    }

    localStorage.setItem('finsight_theme', newTheme);
    updateToggleIcon();

    // Dispatch global event for Chart.js updates
    window.dispatchEvent(new CustomEvent('themeChanged', { detail: { theme: newTheme } }));
  });
}

function updateToggleIcon() {
  const toggleBtn = document.getElementById('themeToggleBtn');
  if (!toggleBtn) return;

  const currentTheme = document.documentElement.getAttribute('data-theme');
  if (currentTheme === 'light') {
    toggleBtn.innerHTML = `<i class="fa-solid fa-sun" style="color: #f59e0b;"></i>`;
    toggleBtn.setAttribute('title', 'Switch to Dark Mode');
  } else {
    toggleBtn.innerHTML = `<i class="fa-solid fa-moon" style="color: #3b82f6;"></i>`;
    toggleBtn.setAttribute('title', 'Switch to Light Mode');
  }
}
