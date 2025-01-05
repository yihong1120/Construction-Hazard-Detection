// Constants
// A list of supported "no warnings" messages in various languages
const NO_WARNINGS_MESSAGES = [
    'No warning',             // English
    '無警告',                  // Traditional Chinese
    '无警告',                  // Simplified Chinese
    "Pas d'avertissement",    // French
    'Không có cảnh báo',       // Vietnamese
    'Tidak ada peringatan',    // Indonesian
    'ไม่มีคำเตือน'             // Thai
  ];

  // WebSocket connection instance
  let socket;

  // A simple logger replacement (or remove it entirely if not needed)
  function logInformation(message) {
    // For production, consider removing or sending to a logging service
    // console.log(`[INFO] ${message}`);
  }
  function logError(message) {
    // For production, consider removing or sending to an error tracking service
    // console.error(`[ERROR] ${message}`);
  }

  document.addEventListener('DOMContentLoaded', () => {
    // Retrieve URL parameters for label and key
    const urlParameters = new URLSearchParams(window.location.search);
    const label = urlParameters.get('label');
    const key = urlParameters.get('key');

    // Redirect to index page if either label or key is missing
    if (!label || !key) {
      logError('Label or key parameter is missing');
      window.location.href = 'index.html';
      return;
    }

    // Retrieve DOM elements for dynamic updates
    const domReferences = {
      cameraTitle: document.getElementById('camera-title'),
      streamImage: document.getElementById('stream-image'),
      loadingIndicator: document.getElementById('loading-indicator'),
      streamMeta: document.getElementById('stream-meta'),
      warningsList: document.getElementById('warnings-ul')
    };

    // Set the camera title to include the label and key
    domReferences.cameraTitle.textContent = `${label} - ${key}`;

    // Initialise the WebSocket connection for real-time updates
    initialiseWebSocket({ label, key, domReferences });

    // Ensure the WebSocket is properly closed when the page is unloaded
    window.addEventListener('beforeunload', () => {
      if (socket) socket.close();
    });
  });

  /**
   * Initialise WebSocket with an object parameter
   */
  function initialiseWebSocket({ label, key, domReferences }) {
    // Determine the appropriate WebSocket protocol
    const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
    const webSocketUrl = `${protocol}${window.location.host}/api/ws/stream/${encodeURIComponent(label)}/${encodeURIComponent(key)}`;

    // Create a new WebSocket connection
    socket = new WebSocket(webSocketUrl);

    // Define WebSocket event handlers
    socket.onopen = () => handleWebSocketOpen();
    socket.onmessage = (event) => handleWebSocketMessage(event, domReferences);
    socket.onerror = (error) => handleWebSocketError(error);
    socket.onclose = () => handleWebSocketClose();
  }

  function handleWebSocketOpen() {
    logInformation('WebSocket connected!');

    // Keep the WebSocket connection alive by sending a ping every 30 seconds
    setInterval(() => {
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000);
  }

  /**
   * Handle incoming WebSocket messages
   */
  function handleWebSocketMessage(event, domReferences) {
    const data = JSON.parse(event.data);

    // Handle error messages from the server
    if (data.error) {
      logError(data.error);
      window.location.href = 'index.html';
      return;
    }

    // Update the live stream image if provided
    if (data.image) {
      updateStreamImage({
        ...domReferences,
        imageData: data.image
      });
    }

    // Update the warnings list if warnings are provided
    if (data.warnings) {
      updateWarningsList(domReferences.warningsList, data.warnings);
    }
  }

  function handleWebSocketError(error) {
    logError(`WebSocket error: ${error}`);
    window.location.href = 'index.html';
  }

  function handleWebSocketClose() {
    logInformation('WebSocket closed');
  }

  /**
   * Update the live stream image
   */
  function updateStreamImage({ streamImage, loadingIndicator, streamMeta, imageData }) {
    // Hide the loading indicator and display the live stream image
    loadingIndicator.style.display = 'none';
    streamImage.style.display = 'block';

    // Set the image source to the received base64 data
    streamImage.src = `data:image/png;base64,${imageData}`;

    // Update the metadata with the current timestamp
    streamMeta.textContent = `Last updated: ${new Date().toLocaleString()}`;
  }

  /**
   * Update the warnings list
   */
  function updateWarningsList(warningsList, warningsData) {
    const warnings = warningsData.split('\n');
    warningsList.innerHTML = '';

    // Check if there are no warnings
    if (warnings.length === 1 && NO_WARNINGS_MESSAGES.includes(warnings[0])) {
      warningsList.className = 'no-warnings';
      appendWarningItem(warningsList, warnings[0], ['no-warning']);
    } else {
      warningsList.className = 'warnings';
      warnings.forEach((warning) => appendWarningItem(warningsList, warning));
    }
  }

  /**
   * Append a single warning to the list
   */
  function appendWarningItem(warningsList, warningText, additionalClasses = []) {
    const paragraph = document.createElement('p');
    paragraph.textContent = warningText;
    paragraph.classList.add('warning-item', ...additionalClasses);
    warningsList.appendChild(paragraph);
  }
