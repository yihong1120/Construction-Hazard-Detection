const API_URL = '/api'; // Backend API base path

// Execute when the document's DOM is fully loaded
document.addEventListener('DOMContentLoaded', async () => {
    const cameraGrid = document.getElementById('camera-grid'); // Reference to the camera grid container

    try {
        const labels = await fetchLabels(); // Fetch the list of labels from the backend
        renderLabels(cameraGrid, labels); // Render the fetched labels onto the page
    } catch (error) {
        logError('Error fetching labels:', error); // Use a custom logging function to avoid direct console use
    }
});

// Fetch labels from the backend
async function fetchLabels() {
    const response = await fetch(`${API_URL}/labels`); // Make a GET request to fetch labels
    if (!response.ok) throw new Error('Failed to fetch labels'); // Throw an error if the response is not OK
    const data = await response.json(); // Parse the JSON response
    return data.labels || []; // Return the labels or an empty array if none exist
}

// Render the labels onto the page
function renderLabels(cameraGrid, labels) {
    labels.forEach(label => {
        const cameraDiv = createCameraDiv(label); // Create a camera container for each label
        cameraGrid.appendChild(cameraDiv); // Add the camera container to the grid
    });
}

// Create a camera container for a label
function createCameraDiv(label) {
    const cameraDiv = createDivElement('camera'); // Create a container with a class
    const link = createLinkElement(`/label.html?label=${encodeURIComponent(label)}`, [
        createTitleElement(label),
        createDescriptionElement(`View ${label}`)
    ]);
    cameraDiv.appendChild(link); // Add the link to the camera container
    return cameraDiv; // Return the constructed camera container
}

// Helper function to create a div element with a specified class
function createDivElement(className) {
    const div = document.createElement('div');
    div.className = className;
    return div;
}

// Helper function to create a link element with specified href and children
function createLinkElement(href, children = []) {
    const link = document.createElement('a');
    link.href = href;
    children.forEach(child => link.appendChild(child));
    return link;
}

// Helper function to create a title element
function createTitleElement(text) {
    const title = document.createElement('h2');
    title.textContent = text;
    return title;
}

// Helper function to create a description element
function createDescriptionElement(text) {
    const description = document.createElement('p');
    description.textContent = text;
    return description;
}

// Custom logging function to replace direct console usage
function logError(message, error) {
    // Here, you can send errors to a logging service or simply comment out this line for production
    // console.error(`${message} ${error}`);
}
