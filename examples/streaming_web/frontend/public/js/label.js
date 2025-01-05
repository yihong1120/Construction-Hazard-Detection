let socket; // Define the WebSocket globally to manage the connection throughout the script

// Execute when the document's DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    const labelTitle = document.getElementById('label-title'); // Reference to the label title element
    const loadingMessage = document.getElementById('loading-message'); // Reference to the loading message element
    const urlParams = new URLSearchParams(window.location.search); // Extract query parameters from the URL
    const label = urlParams.get('label'); // Retrieve the 'label' parameter

    // Validate and initialise the label
    if (!validateLabel(label)) return;

    // Set the page's title to the label name
    labelTitle.textContent = label;
    loadingMessage.style.display = 'block'; // Show the loading message
    initializeWebSocket(label);

    // Ensure the WebSocket connection is closed when the page is unloaded
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
    socket.onerror = () => handleWebSocketError();
    socket.onclose = () => handleWebSocketClose();
}

/**
 * Set up a heartbeat mechanism to keep the WebSocket connection alive.
 */
function setupWebSocketHeartbeat() {
    logInfo('WebSocket connected');
    setInterval(() => {
        if (socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type: 'ping' })); // Send a ping message every 30 seconds
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
    window.location.href = 'index.html'; // Redirect to index.html on error
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
        loadingMessage.style.display = 'none'; // Hide the loading message
    } else {
        logInfo('No data for the current label, redirecting to index.html');
        window.location.href = 'index.html'; // Redirect to index.html if no images
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
    const cameraDiv = document.createElement('div');
    cameraDiv.className = 'camera'; // Add a class for styling
    cameraDiv.dataset.key = key;

    const title = document.createElement('h2');
    title.textContent = key;

    const img = document.createElement('img');
    img.src = `data:image/png;base64,${image}`;
    img.alt = `${key} image`;

    cameraDiv.appendChild(title);
    cameraDiv.appendChild(img);

    cameraDiv.addEventListener('click', () => redirectToCameraPage(key));
    return cameraDiv;
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
    // Replace with logging service or comment out for production
    // console.log(message);
}

function logError(message) {
    // Replace with logging service or comment out for production
    // console.error(message);
}
