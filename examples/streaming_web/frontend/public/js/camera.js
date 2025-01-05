// Constants
// A list of supported "no warnings" messages in various languages
const NO_WARNINGS_MESSAGES = [
    'No warning', // English
    '無警告', // Traditional Chinese
    '无警告', // Simplified Chinese
    'Pas d\'avertissement', // French
    'Không có cảnh báo', // Vietnamese
    'Tidak ada peringatan', // Indonesian
    'ไม่มีคำเตือน' // Thai
];

// WebSocket connection instance
let socket;

document.addEventListener('DOMContentLoaded', () => {
    // Retrieve URL parameters for label and key
    const urlParams = new URLSearchParams(window.location.search);
    const label = urlParams.get('label');
    const key = urlParams.get('key');

    // Redirect to index page if either label or key is missing
    if (!label || !key) {
        console.error('Label or key parameter is missing');
        window.location.href = 'index.html';
        return;
    }

    // Retrieve DOM elements for dynamic updates
    const cameraTitle = document.getElementById('camera-title');
    const streamImage = document.getElementById('stream-image');
    const loadingIndicator = document.getElementById('loading-indicator');
    const streamMeta = document.getElementById('stream-meta');
    const warningsList = document.getElementById('warnings-ul');

    // Set the camera title to include the label and key
    cameraTitle.textContent = `${label} - ${key}`;

    // Initialise the WebSocket connection for real-time updates
    initializeWebSocket(label, key, streamImage, loadingIndicator, streamMeta, warningsList);

    // Ensure the WebSocket is properly closed when the page is unloaded
    window.addEventListener('beforeunload', () => {
        if (socket) socket.close();
    });
});

function initializeWebSocket(label, key, streamImage, loadingIndicator, streamMeta, warningsList) {
    // Determine the appropriate WebSocket protocol based on the current page protocol
    const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
    const wsUrl = `${protocol}${window.location.host}/api/ws/stream/${encodeURIComponent(label)}/${encodeURIComponent(key)}`;

    // Create a new WebSocket connection
    socket = new WebSocket(wsUrl);

    // Define WebSocket event handlers
    socket.onopen = handleWebSocketOpen;
    socket.onmessage = (event) => handleWebSocketMessage(event, streamImage, loadingIndicator, streamMeta, warningsList);
    socket.onerror = handleWebSocketError;
    socket.onclose = handleWebSocketClose;
}

function handleWebSocketOpen() {
    console.log('WebSocket connected!');
    // Keep the WebSocket connection alive by sending a ping message every 30 seconds
    setInterval(() => {
        if (socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ type: 'ping' }));
        }
    }, 30000);
}

function handleWebSocketMessage(event, streamImage, loadingIndicator, streamMeta, warningsList) {
    // Parse the incoming message as JSON
    const data = JSON.parse(event.data);

    // Handle error messages from the server
    if (data.error) {
        console.error(data.error);
        window.location.href = 'index.html';
        return;
    }

    // Update the live stream image if provided
    if (data.image) {
        updateStreamImage(streamImage, loadingIndicator, streamMeta, data.image);
    }

    // Update the warnings list if warnings are provided
    if (data.warnings) {
        updateWarningsList(warningsList, data.warnings);
    }
}

function handleWebSocketError(error) {
    // Log WebSocket errors and redirect to the index page
    console.error('WebSocket error:', error);
    window.location.href = 'index.html';
}

function handleWebSocketClose() {
    // Log when the WebSocket connection is closed
    console.log('WebSocket closed');
}

function updateStreamImage(streamImage, loadingIndicator, streamMeta, imageData) {
    // Hide the loading indicator and display the live stream image
    loadingIndicator.style.display = 'none';
    streamImage.style.display = 'block';
    // Set the image source to the received base64 data
    streamImage.src = `data:image/png;base64,${imageData}`;
    // Update the metadata with the current timestamp
    streamMeta.textContent = `Last updated: ${new Date().toLocaleString()}`;
}

function updateWarningsList(warningsList, warningsData) {
    // Split warnings data into an array of warnings
    const warnings = warningsData.split('\n');
    // Clear the existing warnings list
    warningsList.innerHTML = '';

    // Check if there are no warnings and update the list accordingly
    if (warnings.length === 1 && NO_WARNINGS_MESSAGES.includes(warnings[0])) {
        warningsList.className = 'no-warnings';
        appendWarningItem(warningsList, warnings[0], ['no-warning']);
    } else {
        warningsList.className = 'warnings';
        warnings.forEach(warning => appendWarningItem(warningsList, warning));
    }
}

function appendWarningItem(warningsList, warningText, additionalClasses = []) {
    // Create a new paragraph element for the warning
    const p = document.createElement('p');
    p.textContent = warningText;
    // Add classes for styling
    p.classList.add('warning-item', ...additionalClasses);
    // Append the warning to the warnings list
    warningsList.appendChild(p);
}
