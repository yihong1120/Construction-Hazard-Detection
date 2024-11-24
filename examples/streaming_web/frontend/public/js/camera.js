let socket;

// Wait for the DOM content to fully load
document.addEventListener('DOMContentLoaded', () => {
    // Extract URL parameters to retrieve the label and key
    const urlParams = new URLSearchParams(window.location.search);
    const label = urlParams.get('label'); // Retrieve the 'label' parameter
    const key = urlParams.get('key'); // Retrieve the 'key' parameter

    // If label or key is missing, redirect to the index page
    if (!label || !key) {
        console.error('Label or key parameter is missing'); // Log an error
        window.location.href = 'index.html'; // Redirect to the index page
        return; // Exit the function
    }

    // Retrieve DOM elements for dynamic updates
    const cameraTitle = document.getElementById('camera-title'); // Camera title element
    const streamImage = document.getElementById('stream-image'); // Live stream image
    const loadingIndicator = document.getElementById('loading-indicator'); // Loading indicator
    const streamMeta = document.getElementById('stream-meta'); // Metadata (e.g., last updated timestamp)
    const warningsList = document.getElementById('warnings-ul'); // Warning messages list

    // Set the camera title to include the label and key
    cameraTitle.textContent = `${label} - ${key}`;

    // Initialise the WebSocket connection for real-time updates
    initializeWebSocket(label, key, streamImage, loadingIndicator, streamMeta, warningsList);

    // Ensure the WebSocket is properly closed when the page is unloaded
    window.addEventListener('beforeunload', () => {
        if (socket) {
            socket.close(); // Close the WebSocket connection
        }
    });
});

function initializeWebSocket(label, key, streamImage, loadingIndicator, streamMeta, warningsList) {
    // Determine the correct WebSocket protocol based on the current page protocol
    const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';

    // Construct the WebSocket URL with the label and key parameters
    socket = new WebSocket(`${protocol}${window.location.host}/api/ws/stream/${encodeURIComponent(label)}/${encodeURIComponent(key)}`);

    // Define a list of supported "no warnings" strings in various languages
    const noWarningsMessages = [
        'No warnings', // English
        '無警告', // Traditional Chinese
        '无警告', // Simplified Chinese
        'Pas d\'avertissement', // French
        'Không có cảnh báo', // Vietnamese
        'Tidak ada peringatan', // Indonesian
        'ไม่มีคำเตือน' // Thai
    ];

    // Handle the WebSocket 'open' event
    socket.onopen = () => {
        console.log('WebSocket connected!'); // Log the connection
        // Send a 'ping' message every 30 seconds to keep the connection alive
        setInterval(() => {
            if (socket.readyState === WebSocket.OPEN) {
                socket.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    };

    // Handle incoming messages from the WebSocket
    socket.onmessage = (event) => {
        const data = JSON.parse(event.data); // Parse the received data as JSON

        // If there is an error in the data, log it and redirect to the index page
        if (data.error) {
            console.error(data.error); // Log the error
            window.location.href = 'index.html'; // Redirect to the index page
        } else {
            // Handle image updates
            if (data.image) {
                loadingIndicator.style.display = 'none'; // Hide the loading indicator
                streamImage.style.display = 'block'; // Display the live stream image
                streamImage.src = `data:image/png;base64,${data.image}`; // Update the image source with the base64 data
                const timestamp = new Date().toLocaleString(); // Get the current timestamp
                streamMeta.textContent = `Last updated: ${timestamp}`; // Update the metadata
            }

            // Handle warning messages
            const warnings = data.warnings ? data.warnings.split('\n') : ['No warnings']; // Split warnings into an array
            warningsList.innerHTML = ''; // Clear any previous warnings

            // Check if there are no warnings
            if (warnings.length === 1 && noWarningsMessages.includes(warnings[0])) {
                warningsList.className = 'no-warnings'; // Apply the green background class for "no warnings"
                const p = document.createElement('p'); // Create a new paragraph element
                p.textContent = warnings[0]; // Set the text content to "no warnings"
                p.classList.add('warning-item', 'no-warning'); // Apply individual styles for green warnings
                warningsList.appendChild(p); // Append the paragraph to the warnings list
            } else {
                // If there are warnings, apply the red background class
                warningsList.className = 'warnings'; // Apply the red background class for warnings
                warnings.forEach((warning) => {
                    const p = document.createElement('p'); // Create a new paragraph for each warning
                    p.textContent = warning; // Set the warning text
                    p.classList.add('warning-item'); // Apply individual styles for red warnings
                    warningsList.appendChild(p); // Append the paragraph to the warnings list
                });
            }
        }
    };

    // Handle WebSocket 'error' events
    socket.onerror = (error) => {
        console.error('WebSocket error:', error); // Log the error
        window.location.href = 'index.html'; // Redirect to the index page
    };

    // Handle WebSocket 'close' events
    socket.onclose = () => {
        console.log('WebSocket closed'); // Log the closure
    };
}
