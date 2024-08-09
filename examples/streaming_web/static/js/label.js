$(document).ready(() => {
    // Automatically detect the current page protocol to decide between ws and wss
    const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
    // Create WebSocket connection and configure reconnection strategy
    const socket = io.connect(protocol + document.domain + ':' + location.port, {
        transports: ['websocket'],
        reconnectionAttempts: 5,   // Maximum of 5 reconnection attempts
        reconnectionDelay: 2000    // Reconnection interval of 2000 milliseconds
    });

    // Get the label of the current page
    const currentPageLabel = $('h1').text();  // Assuming the <h1> tag contains the current label name

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
});

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
    if (process.env.NODE_ENV === 'development') {
        console.log(...messages);
    }
}