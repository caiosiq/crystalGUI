/* shared/alerts.js */

function showAlert(type, message) {
  // Create alert element
  const alertDiv = document.createElement('div');
  alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
  alertDiv.innerHTML = `
    ${message}
    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
  `;
  
  // Find or create alerts container
  let alertsContainer = document.getElementById('alerts-container');
  if (!alertsContainer) {
    alertsContainer = document.createElement('div');
    alertsContainer.id = 'alerts-container';
    alertsContainer.style.position = 'fixed';
    alertsContainer.style.top = '20px';
    alertsContainer.style.right = '20px';
    alertsContainer.style.zIndex = '9999';
    alertsContainer.style.maxWidth = '400px';
    document.body.appendChild(alertsContainer);
  }
  
  // Add alert to container
  alertsContainer.appendChild(alertDiv);
  
  // Auto-dismiss after 5 seconds
  setTimeout(() => {
    if (alertDiv.parentNode) {
      alertDiv.remove();
    }
  }, 5000);
}
