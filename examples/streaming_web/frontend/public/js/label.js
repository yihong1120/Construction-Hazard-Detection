let socket; // Define the WebSocket globally to manage the connection throughout the script

// Execute when the document's DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    const labelTitle = document.getElementById('label-title');
    const loadingMessage = document.getElementById('loading-message');
    const urlParams = new URLSearchParams(window.location.search);
    const label = urlParams.get('label');

    if (!validateLabel(label)) return;

    labelTitle.textContent = label;
    loadingMessage.style.display = 'block'; // Show the loading message
    initializeWebSocket(label);

    window.addEventListener('beforeunload', closeWebSocket);
});

/**
 * Validate the label parameter and redirect if invalid.
 *
 * @param {string|null} label - The label parameter from the URL.
 * @returns {boolean} Whether the label is valid.
 */
function validateLabel(label) {
    if (!label) {
        logError('Label parameter is missing in the URL');
        window.location.href = 'index.html'; // Redirect to index.html
        return false;
    }
    return true;
}

/**
 * Initialise the WebSocket connection for live updates.
 *
 * @param {string} label - The label used to establish the WebSocket connection.
 */
function initializeWebSocket(label) {
    const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
    socket = new WebSocket(`${protocol}${window.location.host}/api/ws/labels/${encodeURIComponent(label)}`);

    socket.onopen = setupWebSocketHeartbeat;
    socket.onmessage = (event) => handleUpdate(JSON.parse(event.data), label);
    socket.onerror = handleWebSocketError;
    socket.onclose = handleWebSocketClose;
}

/**
 * Set up a heartbeat mechanism to keep the WebSocket connection alive.
 */
function setupWebSocketHeartbeat() {
    logInfo('WebSocket connected');
    setInterval(() => {
        if (socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type: 'ping' }));
        }
    }, 30000);
}

/**
 * Close the WebSocket connection gracefully.
 */
function closeWebSocket() {
    if (socket) socket.close();
}

/**
 * Handle WebSocket errors by logging the error and redirecting to the index page.
 */
function handleWebSocketError() {
    logError('WebSocket error occurred');
    window.location.href = 'index.html';
}

/**
 * Handle WebSocket closure by logging a message.
 */
function handleWebSocketClose() {
    logInfo('WebSocket connection closed');
}

/**
 * Handle updates received from the WebSocket server.
 *
 * @param {Object} data - The data received from the WebSocket server.
 * @param {string} currentLabel - The label currently being displayed.
 */
function handleUpdate(data, currentLabel) {
    const loadingMessage = document.getElementById('loading-message');
    if (data.label === currentLabel && data.images?.length > 0) {
        renderCameraGrid(data.images);
        loadingMessage.style.display = 'none';
    } else {
        logInfo('No data for the current label, redirecting to index.html');
        window.location.href = 'index.html';
    }
}

/**
 * Render the camera grid with new images.
 *
 * @param {Array} images - An array of image data, each containing a key and base64-encoded image.
 */
function renderCameraGrid(images) {
    const cameraGrid = document.getElementById('camera-grid');
    cameraGrid.innerHTML = ''; // Clear existing content in the grid
    images.forEach(({ key, image }) => {
        const cameraDiv = createCameraDiv(key, image);
        cameraGrid.appendChild(cameraDiv);
    });
}

/**
 * Create a camera div for the given key and image.
 *
 * @param {string} key - The unique key for the camera.
 * @param {string} image - The base64-encoded image data.
 * @returns {HTMLElement} The camera div element.
 */
function createCameraDiv(key, image) {
    const cameraDiv = createElementWithClass('div', 'camera');
    cameraDiv.dataset.key = key;

    const title = createElementWithText('h2', key);
    const img = setupCameraImage(key, image);

    cameraDiv.appendChild(title);
    cameraDiv.appendChild(img);

    cameraDiv.addEventListener('click', () => redirectToCameraPage(key));
    return cameraDiv;
}

/**
 * Create an element with a specific class name.
 *
 * @param {string} tagName - The HTML tag name.
 * @param {string} className - The class name to assign.
 * @returns {HTMLElement} The created element.
 */
function createElementWithClass(tagName, className) {
    const element = document.createElement(tagName);
    element.className = className;
    return element;
}

/**
 * Create an element with text content.
 *
 * @param {string} tagName - The HTML tag name.
 * @param {string} text - The text content to assign.
 * @returns {HTMLElement} The created element.
 */
function createElementWithText(tagName, text) {
    const element = document.createElement(tagName);
    element.textContent = text;
    return element;
}

/**
 * Set up the camera image element.
 *
 * @param {string} key - The unique key for the camera.
 * @param {string} image - The base64-encoded image data.
 * @returns {HTMLImageElement} The image element.
 */
function setupCameraImage(key, image) {
    const img = document.createElement('img');
    img.src = `data:image/png;base64,${image}`;
    img.alt = `${key} image`;
    return img;
}

/**
 * Redirect the user to the camera page for the specified key.
 *
 * @param {string} key - The unique key for the camera.
 */
function redirectToCameraPage(key) {
    const urlParams = new URLSearchParams(window.location.search);
    const label = urlParams.get('label');
    window.location.href = `/camera.html?label=${encodeURIComponent(label)}&key=${encodeURIComponent(key)}`;
}

/**
 * Custom logging functions to replace console statements.
 */

function logInfo(message) {
    // Uncomment this for development or logging services
    // console.log(`[INFO] ${message}`);
}

function logError(message) {
    // Uncomment this for development or logging services
    // console.error(`[ERROR] ${message}`);
}
