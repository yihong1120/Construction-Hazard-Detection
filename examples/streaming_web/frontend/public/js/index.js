const API_URL = '/api'; // Backend API base path

// Execute when the document's DOM is fully loaded
document.addEventListener('DOMContentLoaded', async () => {
    const cameraGrid = document.getElementById('camera-grid'); // Reference to the camera grid container

    try {
        const labels = await fetchLabels(); // Fetch the list of labels from the backend
        renderLabels(cameraGrid, labels); // Render the fetched labels onto the page
    } catch (error) {
        console.error('Error fetching labels:', error); // Log an error message if fetching or processing labels fails
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
    const cameraDiv = document.createElement('div'); // Create a container for the label
    cameraDiv.className = 'camera'; // Assign a class for styling

    const link = document.createElement('a'); // Create a clickable link
    link.href = `/label.html?label=${encodeURIComponent(label)}`; // Encode the label to ensure URL safety

    const title = document.createElement('h2'); // Create a title for the label
    title.textContent = label; // Set the label text

    const description = document.createElement('p'); // Create a description under the title
    description.textContent = `View ${label}`; // Set the descriptive text

    link.appendChild(title); // Add the title to the link
    link.appendChild(description); // Add the description to the link
    cameraDiv.appendChild(link); // Add the link to the camera container

    return cameraDiv; // Return the constructed camera container
}
