/* ----------------------------------
   Constants
------------------------------------- */
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

/* ----------------------------------
 Global Variables & Element Selectors
------------------------------------- */
// WebSocket connection instance
let socket;

/* ----------------------------------
 Utility Functions
------------------------------------- */
/**
* Custom logging function to avoid direct console usage
*
* @param {string} message - The message to log.
*/
function logInformation(message) {
  // For production, consider removing or sending to a logging service
  // console.log(`[INFO] ${message}`);
}

/**
* Custom error logging function to avoid direct console usage
*
* @param {string} message - The error message to log.
*/
function logError(message) {
  // For production, consider removing or sending to an error tracking service
  // console.error(`[ERROR] ${message}`);
}

/* ----------------------------------
 Initialization
------------------------------------- */
document.addEventListener('DOMContentLoaded', () => {
  initialisePage();
  setupUnloadHandler();
});

/**
* Initialise the page by validating parameters and setting up WebSocket.
*/
function initialisePage() {
  const { label, key } = getURLParameters();

  // Redirect to index page if either label or key is missing
  if (!label || !key) {
      logError('Label or key parameter is missing');
      redirectToIndex();
      return;
  }

  const domReferences = getDOMReferences();
  setCameraTitle(domReferences.cameraTitle, label, key);
  initialiseWebSocket({ label, key, domReferences });
}

/**
* Retrieve and validate URL parameters.
*
* @returns {Object} An object containing label and key.
*/
function getURLParameters() {
  const urlParameters = new URLSearchParams(window.location.search);
  const label = urlParameters.get('label');
  const key = urlParameters.get('key');
  return { label, key };
}

/**
* Retrieve necessary DOM elements for dynamic updates.
*
* @returns {Object} An object containing DOM references.
*/
function getDOMReferences() {
  return {
      cameraTitle: document.getElementById('camera-title'),
      streamImage: document.getElementById('stream-image'),
      loadingIndicator: document.getElementById('loading-indicator'),
      streamMeta: document.getElementById('stream-meta'),
      warningsList: document.getElementById('warnings-ul')
  };
}

/**
* Set the camera title to include the label and key.
*
* @param {HTMLElement} cameraTitle - The DOM element for camera title.
* @param {string} label - The label parameter.
* @param {string} key - The key parameter.
*/
function setCameraTitle(cameraTitle, label, key) {
  cameraTitle.textContent = `${label} - ${key}`;
}

/**
* Ensure the WebSocket is properly closed when the page is unloaded.
*/
function setupUnloadHandler() {
  window.addEventListener('beforeunload', () => {
      closeWebSocket();
  });
}

/**
* Redirect the user to the index page.
*/
function redirectToIndex() {
  window.location.href = 'index.html';
}

/* ----------------------------------
 WebSocket Handling
------------------------------------- */
/**
* Initialise WebSocket with an object parameter.
*
* @param {Object} params - The parameters for WebSocket initialization.
* @param {string} params.label - The label used to establish the WebSocket connection.
* @param {string} params.key - The key used to establish the WebSocket connection.
* @param {Object} params.domReferences - The DOM elements to update based on WebSocket messages.
*/
function initialiseWebSocket({ label, key, domReferences }) {
  const protocol = getWebSocketProtocol();
  const webSocketUrl = `${protocol}${window.location.host}/api/ws/stream/${encodeURIComponent(label)}/${encodeURIComponent(key)}`;

  // Create a new WebSocket connection
  socket = new WebSocket(webSocketUrl);

  // Define WebSocket event handlers
  socket.onopen = handleWebSocketOpen;
  socket.onmessage = (event) => handleWebSocketMessage(event, domReferences);
  socket.onerror = handleWebSocketError;
  socket.onclose = handleWebSocketClose;
}

/**
* Determine the appropriate WebSocket protocol based on the page's protocol.
*
* @returns {string} The WebSocket protocol ('wss://' or 'ws://').
*/
function getWebSocketProtocol() {
  return window.location.protocol === 'https:' ? 'wss://' : 'ws://';
}

/**
* Handle WebSocket connection establishment.
*/
function handleWebSocketOpen() {
  logInformation('WebSocket connected!');
  setupWebSocketHeartbeat();
}

/**
* Set up a heartbeat mechanism to keep the WebSocket connection alive.
*/
function setupWebSocketHeartbeat() {
  setInterval(() => {
      if (socket && socket.readyState === WebSocket.OPEN) {
          socket.send(JSON.stringify({ type: 'ping' }));
      }
  }, 30000); // 30 seconds
}

/**
* Handle incoming WebSocket messages.
*
* @param {MessageEvent} event - The WebSocket message event.
* @param {Object} domReferences - The DOM elements to update.
*/
function handleWebSocketMessage(event, domReferences) {
  try {
      const data = JSON.parse(event.data);
      processWebSocketData(data, domReferences);
  } catch (error) {
      logError(`Failed to parse WebSocket message: ${error}`);
  }
}

/**
* Process the data received from the WebSocket.
*
* @param {Object} data - The data received from the WebSocket server.
* @param {Object} domReferences - The DOM elements to update.
*/
function processWebSocketData(data, domReferences) {
  if (data.error) {
      handleServerError(data.error);
      return;
  }

  if (data.image) {
      updateStreamImage({ ...domReferences, imageData: data.image });
  }

  if (data.warnings) {
      updateWarningsList(domReferences.warningsList, data.warnings);
  }
}

/**
* Handle error messages from the server.
*
* @param {string} errorMessage - The error message from the server.
*/
function handleServerError(errorMessage) {
  logError(errorMessage);
  redirectToIndex();
}

/**
* Handle WebSocket errors.
*
* @param {Event} error - The WebSocket error event.
*/
function handleWebSocketError(error) {
  logError(`WebSocket error: ${error}`);
  redirectToIndex();
}

/**
* Handle WebSocket closure.
*/
function handleWebSocketClose() {
  logInformation('WebSocket closed');
}

/**
* Close the WebSocket connection gracefully.
*/
function closeWebSocket() {
  if (socket) {
      socket.close();
  }
}

/* ----------------------------------
 DOM Updates
------------------------------------- */
/**
* Update the live stream image.
*
* @param {Object} params - Parameters for updating the stream image.
* @param {HTMLElement} params.streamImage - The image element to update.
* @param {HTMLElement} params.loadingIndicator - The loading indicator element.
* @param {HTMLElement} params.streamMeta - The metadata element to update.
* @param {string} params.imageData - The base64-encoded image data.
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
* Update the warnings list.
*
* @param {HTMLElement} warningsList - The unordered list element to update.
* @param {string} warningsData - The warnings data as a newline-separated string.
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
* Append a single warning to the list.
*
* @param {HTMLElement} warningsList - The unordered list element.
* @param {string} warningText - The warning text to append.
* @param {Array} additionalClasses - Additional CSS classes to add to the warning item.
*/
function appendWarningItem(warningsList, warningText, additionalClasses = []) {
  const paragraph = document.createElement('p');
  paragraph.textContent = warningText;
  paragraph.classList.add('warning-item', ...additionalClasses);
  warningsList.appendChild(paragraph);
}
