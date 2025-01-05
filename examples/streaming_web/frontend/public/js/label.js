let socket; // Define the WebSocket globally to manage the connection throughout the script

// Execute when the document's DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    const labelTitle = document.getElementById('label-title'); // Reference to the label title element
    const cameraGrid = document.getElementById('camera-grid'); // Reference to the camera grid container
    const loadingMessage = document.getElementById('loading-message'); // Reference to the loading message element
    const urlParams = new URLSearchParams(window.location.search); // Extract query parameters from the URL
    const label = urlParams.get('label'); // Retrieve the 'label' parameter

    // If the label is missing from the URL, log an error and terminate further execution
    if (!label) {
        console.error('Label parameter is missing in the URL');
        window.location.href = 'index.html'; // Redirect to index.html
        return;
    }

    // Set the page's title to the label name
    labelTitle.textContent = label;
    loadingMessage.style.display = 'block'; // Show the loading message
    initializeWebSocket(label);

    // Ensure the WebSocket connection is closed when the page is unloaded
    window.addEventListener('beforeunload', () => {
        if (socket) {
            socket.close(); // Close the WebSocket connection gracefully
        }
    });
});

/**
 * Initialise the WebSocket connection for live updates.
 *
 * @param {string} label - The label used to establish the WebSocket connection.
 */
function initializeWebSocket(label) {
    // Determine the appropriate WebSocket protocol based on the page's protocol
    const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
    socket = new WebSocket(`${protocol}${window.location.host}/api/ws/labels/${encodeURIComponent(label)}`);

    // Handle WebSocket connection establishment
    socket.onopen = setupWebSocketHeartbeat;

    // Handle incoming messages from the WebSocket server
    socket.onmessage = (event) => {
        const data = JSON.parse(event.data); // Parse the received JSON data
        handleUpdate(data, label); // Process the update based on the current label
    };

    // Handle WebSocket errors
    socket.onerror = handleWebSocketError;

    // Handle WebSocket closure
    socket.onclose = handleWebSocketClose;
}

/**
 * Set up a heartbeat mechanism to keep the WebSocket connection alive.
 */
function setupWebSocketHeartbeat() {
    console.log('WebSocket connected!');
    setInterval(() => {
        if (socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type: 'ping' })); // Send a ping message every 30 seconds
        }
    }, 30000);
}

/**
 * Handle WebSocket errors by logging the error and redirecting to the index page.
 *
 * @param {Event} error - The WebSocket error event.
 */
function handleWebSocketError(error) {
    console.error('WebSocket error:', error);
    window.location.href = 'index.html'; // Redirect to index.html on error
}

/**
 * Handle WebSocket closure by logging a message to the console.
 */
function handleWebSocketClose() {
    console.log('WebSocket closed');
}

/**
 * Handle updates received from the WebSocket server.
 *
 * @param {Object} data - The data received from the WebSocket server.
 * @param {string} currentLabel - The label currently being displayed.
 */
function handleUpdate(data, currentLabel) {
    const loadingMessage = document.getElementById('loading-message');
    if (data.label === currentLabel && data.images.length > 0) {
        console.log('Received update for current label:', data.label);
        renderCameraGrid(data.images); // Update the camera grid with new images
        loadingMessage.style.display = 'none'; // Hide the loading message
    } else {
        console.log('No data for the current label, redirecting to index.html');
        window.location.href = 'index.html'; // Redirect to index.html if no images
    }
}

/**
 * Render the camera grid with new images.
 *
 * @param {Array} images - An array of image data, each containing a key and base64-encoded image.
 */
function renderCameraGrid(images) {
    const cameraGrid = document.getElementById('camera-grid'); // Reference to the camera grid container
    cameraGrid.innerHTML = ''; // Clear existing content in the grid
    images.forEach(({ key, image }) => {
        const cameraDiv = createOrUpdateCameraDiv(key, image); // Create or update the camera div
        cameraGrid.appendChild(cameraDiv); // Append the camera div to the grid
    });
}

/**
 * Create or update a camera div for the given key and image.
 *
 * @param {string} key - The unique key for the camera.
 * @param {string} image - The base64-encoded image data.
 * @returns {HTMLElement} The camera div element.
 */
function createOrUpdateCameraDiv(key, image) {
    const existingCameraDiv = document.querySelector(`.camera[data-key="${key}"]`);
    if (existingCameraDiv) {
        const img = existingCameraDiv.querySelector('img');
        img.src = `data:image/png;base64,${image}`; // Update the image source
        return existingCameraDiv;
    } else {
        const cameraDiv = document.createElement('div');
        cameraDiv.className = 'camera'; // Add a class for styling
        cameraDiv.dataset.key = key; // Set a custom data attribute with the key

        const title = document.createElement('h2');
        title.textContent = key; // Display the key directly

        const img = document.createElement('img');
        img.src = `data:image/png;base64,${image}`; // Set the base64-encoded image as the source
        img.alt = `${key} image`; // Add an alternative text for accessibility

        cameraDiv.appendChild(title); // Append the title to the div
        cameraDiv.appendChild(img); // Append the image to the div

        cameraDiv.addEventListener('click', () => {
            redirectToCameraPage(key); // Redirect to the camera page on click
        });

        return cameraDiv;
    }
}

/**
 * Redirect the user to the camera page for the specified key.
 *
 * @param {string} key - The unique key for the camera.
 */
function redirectToCameraPage(key) {
    const urlParams = new URLSearchParams(window.location.search);
    const label = urlParams.get('label'); // Retrieve the 'label' parameter
    window.location.href = `/camera.html?label=${encodeURIComponent(label)}&key=${encodeURIComponent(key)}`; // Redirect to the camera page
}
