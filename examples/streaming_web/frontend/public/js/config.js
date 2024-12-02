// Select DOM elements
const configContainer = document.getElementById("config-container");
const configView = document.getElementById("config-view");
const editBtn = document.getElementById("edit-btn");
const editor = document.getElementById("editor");
const configEditor = document.getElementById("config-editor");
const saveBtn = document.getElementById("save-btn");
const cancelBtn = document.getElementById("cancel-btn");
const statusDiv = document.getElementById("status");

let isSaving = false; // Prevent multiple save requests
let configLoaded = false; // Track if config is already loaded
let resourceMonitorIntervalId; // For resource monitoring

// Utility function to show loading status
function setLoading(isLoading) {
    statusDiv.textContent = isLoading ? "Loading..." : "";
    statusDiv.style.color = isLoading ? "blue" : "";
}

// Fetch configuration from the server
async function fetchConfig() {
    if (configLoaded) return; // Prevent duplicate fetches
    setLoading(true); // Show loading indicator
    try {
        const response = await fetch("/api/config");
        if (response.ok) {
            const data = await response.json();
            configView.textContent = JSON.stringify(data.config, null, 2);
            configEditor.value = JSON.stringify(data.config, null, 2);
            statusDiv.textContent = "Configuration loaded successfully!";
            statusDiv.style.color = "green";
            configLoaded = true; // Mark config as loaded
        } else {
            throw new Error("Failed to fetch configuration.");
        }
    } catch (error) {
        console.error(error);
        statusDiv.textContent = "Error fetching configuration.";
        statusDiv.style.color = "red";
        // Optionally reset configLoaded to allow retry
        configLoaded = false;
    } finally {
        setLoading(false); // Hide loading indicator
    }
}

// Save updated configuration to the server
async function saveConfig() {
    if (isSaving) {
        console.warn("Save operation is already in progress.");
        return; // Prevent duplicate save operations
    }

    isSaving = true; // Set saving flag
    setLoading(true); // Show loading indicator
    try {
        const config = JSON.parse(configEditor.value);
        const response = await fetch("/api/config", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ config }),
        });
        if (response.ok) {
            statusDiv.textContent = "Configuration updated successfully!";
            statusDiv.style.color = "green";
            configLoaded = false; // Reset loaded flag to allow re-fetch
            await fetchConfig(); // Reload the updated configuration
            toggleEditor(false); // Automatically return to the view-only interface
        } else {
            throw new Error("Failed to save configuration.");
        }
    } catch (error) {
        console.error(error);
        statusDiv.textContent = "Error saving configuration.";
        statusDiv.style.color = "red";
    } finally {
        isSaving = false; // Reset saving flag
        setLoading(false); // Hide loading indicator
    }
}

// Toggle between editor and view-only mode
function toggleEditor(showEditor) {
    if (showEditor) {
        editor.classList.remove("hidden");
        configView.classList.add("hidden");
        editBtn.classList.add("hidden");
    } else {
        editor.classList.add("hidden");
        configView.classList.remove("hidden");
        editBtn.classList.remove("hidden");
    }
}

// Initialize event listeners once
function initializeEventListeners() {
    editBtn.addEventListener("click", () => toggleEditor(true));
    cancelBtn.addEventListener("click", () => toggleEditor(false));
    saveBtn.addEventListener("click", saveConfig);
}

// Clean up and reset state on page unload
function cleanUp() {
    statusDiv.textContent = "";
    configLoaded = false;
    if (resourceMonitorIntervalId) {
        clearInterval(resourceMonitorIntervalId);
    }
}

// Load configuration when the page is ready
document.addEventListener("DOMContentLoaded", () => {
    fetchConfig();
    initializeEventListeners();
    window.addEventListener("beforeunload", cleanUp);
});

// Monitor resource usage (optional, for debugging)
function monitorResourceUsage() {
    resourceMonitorIntervalId = setInterval(() => {
        if (window.performance && window.performance.memory) {
            const memory = window.performance.memory;
            console.log(`Memory usage: ${(memory.usedJSHeapSize / 1024 / 1024).toFixed(2)} MB`);
        }
    }, 5000); // Log every 5 seconds
}

// Uncomment the following line to enable resource monitoring
// monitorResourceUsage();
