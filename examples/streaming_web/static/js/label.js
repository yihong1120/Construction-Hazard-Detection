$(document).ready(() => {
    initializeWebSocket();
});

/**
 * Initialize the WebSocket connection and set up event handlers.
 */
function initializeWebSocket() {
    const protocol = getWebSocketProtocol();
    if (!isSocketIODefined()) return;

    const socket = createWebSocketConnection(protocol);

    const currentPageLabel = getCurrentPageLabel();

    setupSocketEventHandlers(socket, currentPageLabel);
}

/**
 * Get the appropriate WebSocket protocol based on the current page protocol.
 * @returns {string} The WebSocket protocol ('ws://' or 'wss://').
 */
function getWebSocketProtocol() {
    return window.location.protocol === 'https:' ? 'wss://' : 'ws://';
}

/**
 * Check if Socket.IO is defined.
 * @returns {boolean} True if Socket.IO is defined, false otherwise.
 */
function isSocketIODefined() {
    if (typeof io === 'undefined') {
        showError('Socket.IO is not defined. Please ensure it is included in your HTML.');
        return false;
    }
    return true;
}

/**
 * Create a WebSocket connection with reconnection strategy.
 * @param {string} protocol - The WebSocket protocol ('ws://' or 'wss://').
 * @returns {Object} The WebSocket connection instance.
 */
function createWebSocketConnection(protocol) {
    return io.connect(protocol + document.domain + ':' + location.port, {
        transports: ['websocket'],
        reconnectionAttempts: 5,   // Maximum of 5 reconnection attempts
        reconnectionDelay: 2000    // Reconnection interval of 2000 milliseconds
    });
}

/**
 * Get the label of the current page.
 * @returns {string} The label of the current page.
 */
function getCurrentPageLabel() {
    return $('h1').text();  // Assuming the <h1> tag contains the current label name
}

/**
 * Set up WebSocket event handlers.
 * @param {Object} socket - The WebSocket connection instance.
 * @param {string} currentPageLabel - The label of the current page.
 */
function setupSocketEventHandlers(socket, currentPageLabel) {
    socket.on('connect', () => {
        debugLog('WebSocket connected!');
    });

    socket.on('connect_error', (error) => {
        debugLog('WebSocket connection error:', error);
    });

    socket.on('reconnect_attempt', () => {
        debugLog('Attempting to reconnect...');
    });

    socket.on('update', (data) => {
        handleUpdate(data, currentPageLabel);
    });
}

/**
 * Handle WebSocket updates
 * @param {Object} data - The received data
 * @param {string} currentPageLabel - The label of the current page
 */
function handleUpdate(data, currentPageLabel) {
    // Check if the received data is applicable to the current page's label
    if (data.label === currentPageLabel) {
        debugLog('Received update for current label:', data.label);
        updateCameraGrid(data);
    } else {
        debugLog('Received update for different label:', data.label);
    }
}

/**
 * Update the camera grid
 * @param {Object} data - The data containing images and names
 */
function updateCameraGrid(data) {
    const fragment = document.createDocumentFragment();
    data.images.forEach((image, index) => {
        const cameraData = {
            image: image,
            imageName: data.image_names[index],
            label: data.label
        };
        const cameraDiv = createCameraDiv(cameraData);
        fragment.appendChild(cameraDiv);
    });
    $('.camera-grid').empty().append(fragment);
}

/**
 * Create a camera div element
 * @param {Object} cameraData - The data for creating the camera div
 * @param {string} cameraData.image - The image data
 * @param {string} cameraData.imageName - The image name
 * @param {string} cameraData.label - The label name
 * @returns {HTMLElement} - The div element containing the image and title
 */
function createCameraDiv({ image, imageName, label }) {
    const cameraDiv = $('<div>').addClass('camera');
    const title = $('<h2>').text(imageName);
    const img = $('<img>').attr('src', `data:image/png;base64,${image}`).attr('alt', `${label} image`);
    cameraDiv.append(title).append(img);
    return cameraDiv[0];
}

/**
 * Log messages for debugging purposes
 * @param  {...any} messages - The messages to log
 */
function debugLog(...messages) {
    if (isDevelopmentEnvironment()) {
        logToConsole(...messages);
    }
}

/**
 * Show error messages.
 * @param {string} message - The error message to display.
 */
function showError(message) {
    if (isDevelopmentEnvironment()) {
        logErrorToConsole(message);
    }
}

/**
 * Log messages to the console.
 * @param  {...any} messages - The messages to log.
 */
function logToConsole(...messages) {
    if (typeof console !== 'undefined') {
        console.log(...messages);
    }
}

/**
 * Log error messages to the console.
 * @param {string} message - The error message to log.
 */
function logErrorToConsole(message) {
    if (typeof console !== 'undefined') {
        console.error(message);
    }
}

/**
 * Check if the current environment is development.
 * @returns {boolean} True if the current environment is development, false otherwise.
 */
function isDevelopmentEnvironment() {
    return typeof process !== 'undefined' && process.env && process.env.NODE_ENV === 'development';
}
