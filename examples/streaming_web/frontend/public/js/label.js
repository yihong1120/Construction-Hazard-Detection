let socket; // Define the WebSocket globally to manage the connection throughout the script

// Execute when the document's DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    const labelTitle = document.getElementById('label-title'); // Reference to the label title element
    const cameraGrid = document.getElementById('camera-grid'); // Reference to the camera grid container
    const urlParams = new URLSearchParams(window.location.search); // Extract query parameters from the URL
    const label = urlParams.get('label'); // Retrieve the 'label' parameter

    // If the label is missing from the URL, log an error and terminate further execution
    if (!label) {
        console.error('Label parameter is missing in the URL');
        return;
    }

    // Set the page's title to the label name
    labelTitle.textContent = label;

    // Initialise the WebSocket connection for the given label
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
    socket.onopen = () => {
        console.log('WebSocket connected!');

        // Set up a heartbeat mechanism to keep the connection alive
        setInterval(() => {
            if (socket.readyState === WebSocket.OPEN) {
                socket.send(JSON.stringify({ type: 'ping' })); // Send a ping message every 30 seconds
            }
        }, 30000);
    };

    // Handle incoming messages from the WebSocket server
    socket.onmessage = (event) => {
        const data = JSON.parse(event.data); // Parse the received JSON data
        handleUpdate(data, label); // Process the update based on the current label
    };

    // Handle WebSocket errors
    socket.onerror = (error) => console.error('WebSocket error:', error);

    // Handle WebSocket closure
    socket.onclose = () => console.log('WebSocket closed');
}

/**
 * Handle updates received from the WebSocket server.
 *
 * @param {Object} data - The data received from the WebSocket server.
 * @param {string} currentLabel - The label currently being displayed.
 */
function handleUpdate(data, currentLabel) {
    if (data.label === currentLabel) {
        console.log('Received update for current label:', data.label);
        updateCameraGrid(data.images); // Update the camera grid with new images
    } else {
        console.log('Received update for different label:', data.label);
    }
}

/**
 * Update the camera grid with new images.
 *
 * @param {Array} images - An array of image data, each containing a key and base64-encoded image.
 */
function updateCameraGrid(images) {
    const cameraGrid = document.getElementById('camera-grid'); // Reference to the camera grid container
    images.forEach(({ key, image }) => {
        // Check if a camera div for the given key already exists
        const existingCameraDiv = document.querySelector(`.camera[data-key="${key}"]`);
        if (existingCameraDiv) {
            // Update the existing image source
            const img = existingCameraDiv.querySelector('img');
            img.src = `data:image/png;base64,${image}`;
        } else {
            // Create a new camera div if it doesn't exist
            const cameraDiv = document.createElement('div');
            cameraDiv.className = 'camera'; // Add a class for styling
            cameraDiv.dataset.key = key; // Set a custom data attribute with the key

            // Create a title for the camera
            const title = document.createElement('h2');
            title.textContent = key; // Display the key directly

            // Create an image element for the camera
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${image}`; // Set the base64-encoded image as the source
            img.alt = `${key} image`; // Add an alternative text for accessibility

            // Append the title and image to the camera div
            cameraDiv.appendChild(title);
            cameraDiv.appendChild(img);

            // Append the camera div to the grid
            cameraGrid.appendChild(cameraDiv);
        }
    });
}
