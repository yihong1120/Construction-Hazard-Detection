const API_URL = '/api'; // Backend API base path

// Execute when the document's DOM is fully loaded
document.addEventListener('DOMContentLoaded', async () => {
    const cameraGrid = document.getElementById('camera-grid'); // Reference to the camera grid container

    try {
        // Fetch the list of labels from the backend
        const response = await fetch(`${API_URL}/labels`);
        if (!response.ok) throw new Error('Failed to fetch labels'); // Throw an error if the response is not OK
        const data = await response.json(); // Parse the JSON response

        // Render the fetched labels onto the page
        const labels = data.labels || []; // Default to an empty array if no labels are returned
        labels.forEach(label => {
            const cameraDiv = document.createElement('div'); // Create a container for each label
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
            cameraGrid.appendChild(cameraDiv); // Add the camera container to the grid
        });
    } catch (error) {
        // Log an error message if fetching or processing labels fails
        console.error('Error fetching labels:', error);
    }
});
