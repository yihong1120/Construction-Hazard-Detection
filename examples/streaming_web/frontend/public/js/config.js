const configContainer = document.getElementById("config-container");
const editBtn = document.getElementById("edit-btn");
const addConfigBtn = document.getElementById("add-config-btn");
const saveBtn = document.getElementById("save-btn");
const cancelBtn = document.getElementById("cancel-btn");
const formControls = document.getElementById("form-controls");

let configData = [];
let isEditing = false;

async function fetchConfig() {
    try {
        const response = await fetch("/api/config");
        if (!response.ok) throw new Error("Failed to fetch configuration.");
        const data = await response.json();
        // Transform notifications object to array
        configData = data.config.map(config => ({
            ...config,
            notifications: Object.entries(config.notifications).map(([token, language]) => ({ token, language }))
        }));
        renderConfigForm();
    } catch (error) {
        console.error(error);
    }
}

function formatDetectionItemName(key) {
    // Replace underscores with spaces and capitalize the first letter of each word
    return key.replace(/_/g, ' ').replace(/\b\w/g, char => char.toUpperCase());
}

function updateConfigDataFromForm() {
    const configItems = configContainer.children;
    configData = Array.from(configItems).map((container, index) => {
        const inputs = container.querySelectorAll("input, select");
        const config = { notifications: [], detection_items: {} };
        inputs.forEach((input) => {
            if (input.name === "line_token" || input.name === "language") {
                // Process notification items
                const notifIndex = input.getAttribute("data-notif-index");
                if (!config.notifications[notifIndex]) {
                    config.notifications[notifIndex] = { token: "", language: "en" };
                }
                if (input.name === "line_token") {
                    config.notifications[notifIndex].token = input.value.trim();
                } else if (input.name === "language") {
                    config.notifications[notifIndex].language = input.value;
                }
            } else if (input.name === "no_expire_date") {
                // Process no_expire_date
                config.no_expire_date = input.checked;
                if (input.checked) {
                    config.expire_date = "No Expire Date";
                }
            } else if (input.name === "expire_date") {
                // Process expire_date
                if (!config.expire_date && input.type === "date") {
                    config.expire_date = input.value || new Date().toISOString().split('T')[0];
                }
                // Save the previous expire_date value
                config.previous_expire_date = input.value;
            } else if (input.name.startsWith("detect_")) {
                // Process detection items and detect_with_server
                if (input.name === "detect_with_server") {
                    config.detect_with_server = input.checked;
                } else {
                    config.detection_items[input.name] = input.checked;
                }
            } else if (input.name) {
                // Process other input fields
                config[input.name] = input.value.trim();
            }
        });
        return config;
    });
}

function renderConfigForm() {
    configContainer.innerHTML = ""; // Clear existing config items

    configData.forEach((config, index) => {
        const container = document.createElement("div");
        container.className = "config-item";

        // Process expire_date
        let expireDateValue;
        if (config.expire_date === "No Expire Date") {
            expireDateValue = "";
        } else if (config.expire_date) {
            expireDateValue = config.expire_date;
        } else {
            // If expire_date is not set, set it to today's date
            expireDateValue = new Date().toISOString().split('T')[0];
            config.expire_date = expireDateValue;
        }

        // Set the inner HTML of the container
        container.innerHTML = `
            <div class="config-header">
                <div class="site-stream">
                    <label>
                        Site:
                        <input type="text" name="site" value="${config.site || ''}" ${isEditing ? "" : "disabled"} />
                    </label>
                    <span>-</span>
                    <label>
                        Stream Name:
                        <input type="text" name="stream_name" value="${config.stream_name || ''}" ${isEditing ? "" : "disabled"} />
                    </label>
                </div>
                <button type="button" class="delete-config-btn" style="display: ${isEditing ? "block" : "none"};">
                    <i class="fas fa-trash-alt"></i>
                </button>
            </div>
            <label>
                Video URL: <input type="text" name="video_url" value="${config.video_url || ''}" ${isEditing ? "" : "disabled"} />
            </label>
            <label>
                Model Key:
                <select name="model_key" ${isEditing ? "" : "disabled"}>
                    ${["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]
                        .map((key) => `<option value="${key}" ${config.model_key === key ? "selected" : ""}>${key}</option>`)
                        .join("")}
                </select>
            </label>
            <label>
                Expiry Date:
                <input type="date" name="expire_date" value="${expireDateValue}" ${isEditing ? "" : "disabled"} ${config.expire_date === "No Expire Date" ? "style='display:none;'" : ""} />
                <input type="text" value="No Expire Date" disabled ${config.expire_date === "No Expire Date" ? "" : "style='display:none;'"}>
                ${isEditing ? `
                <label>
                    <input type="checkbox" name="no_expire_date" ${config.expire_date === "No Expire Date" ? "checked" : ""} />
                    No Expire Date
                </label>
                ` : ""}
            </label>
            <label>
                <input type="checkbox" name="detect_with_server" ${config.detect_with_server ? "checked" : ""} ${isEditing ? "" : "disabled"} />
                Detect with Server
            </label>
            <fieldset>
                <legend>Detection Items</legend>
                ${Object.entries(config.detection_items).map(([key, value]) => `
                    <label>
                        <input type="checkbox" name="${key}" ${value ? "checked" : ""} ${isEditing ? "" : "disabled"} />
                        ${formatDetectionItemName(key)}
                    </label>
                `).join("")}
            </fieldset>
            <fieldset>
                <legend>Notifications</legend>
                <div class="notifications-container">
                    ${config.notifications.map((notification, notifIndex) => `
                        <div class="notification-item" data-notif-index="${notifIndex}">
                            ${
                                isEditing
                                    ? '<button type="button" class="delete-notification"><i class="fas fa-times"></i></button>'
                                    : ""
                            }
                            <div class="notification-content">
                                <label>
                                    Token: <input type="text" name="line_token" value="${notification.token}" data-notif-index="${notifIndex}" ${isEditing ? "" : "disabled"} />
                                </label>
                                <label>
                                    Language:
                                    <select name="language" data-notif-index="${notifIndex}" ${isEditing ? "" : "disabled"}>
                                        ${["zh-TW", "zh-CN", "en", "fr", "id", "vt", "th"]
                                            .map((l) => `<option value="${l}" ${notification.language === l ? "selected" : ""}>${l}</option>`)
                                            .join("")}
                                    </select>
                                </label>
                            </div>
                        </div>
                    `).join("")}
                </div>
                ${isEditing ? '<button type="button" class="add-notification"><i class="fas fa-plus"></i> Add Notification</button>' : ""}
            </fieldset>
        `;

        // Delete config button event
        const deleteConfigBtn = container.querySelector(".delete-config-btn");
        deleteConfigBtn.addEventListener("click", () => {
            updateConfigDataFromForm();
            configData.splice(index, 1); // Fetch the updated config data
            renderConfigForm(); // Re-render the config form
        });

        // Event listener for deleting a notification
        const notificationsContainer = container.querySelector(".notifications-container");
        notificationsContainer.addEventListener("click", (event) => {
            if (event.target.closest(".delete-notification")) {
                // Find the index of the config
                const configIndex = index;

                // Find the index of the notification
                const notificationItem = event.target.closest(".notification-item");
                const notifIndex = parseInt(notificationItem.getAttribute("data-notif-index"));

                updateConfigDataFromForm();

                // Find the updated config
                const updatedConfig = configData[configIndex];

                // Remove the notification
                updatedConfig.notifications.splice(notifIndex, 1);
                renderConfigForm(); // 重新渲染通知
            }
        });

        // Event listener for adding a notification
        const addNotificationBtn = container.querySelector(".add-notification");
        addNotificationBtn?.addEventListener("click", () => {
            updateConfigDataFromForm();
            configData[index].notifications.push({ token: "", language: "en" }); // 添加空的通知
            renderConfigForm();
        });

        // Event listener for toggling no_expire_date
        const expireDateInput = container.querySelector("input[name='expire_date']");
        const noExpireDateCheckbox = container.querySelector("input[name='no_expire_date']");
        const noExpireDateText = container.querySelector("input[type='text'][value='No Expire Date']");

        if (noExpireDateCheckbox) {
            // Initialise the display of the expire date input and no expire date text
            if (noExpireDateCheckbox.checked) {
                expireDateInput.style.display = "none";
                noExpireDateText.style.display = "";
            } else {
                expireDateInput.style.display = "";
                noExpireDateText.style.display = "none";
            }

            // Add event listener for toggling the expire date input and no expire date text
            noExpireDateCheckbox.addEventListener("change", () => {
                if (noExpireDateCheckbox.checked) {
                    // Save the previous expire date value
                    config.previous_expire_date = expireDateInput.value;
                    expireDateInput.style.display = "none";
                    noExpireDateText.style.display = "";
                } else {
                    expireDateInput.style.display = "";
                    noExpireDateText.style.display = "none";
                    // Restore the previous expire date value
                    expireDateInput.value = config.previous_expire_date || new Date().toISOString().split('T')[0];
                }
            });
        }

        // Add the container to the config container
        configContainer.appendChild(container);
    });
}

function toggleEditMode(enable) {
    isEditing = enable;

    // Re-render the form
    renderConfigForm();

    // Toggle the visibility of the buttons and form controls
    editBtn.classList.toggle("hidden", enable);
    addConfigBtn.classList.toggle("hidden", !enable);
    formControls.classList.toggle("hidden", !enable);

    if (!enable) {
        // If exiting edit mode, fetch the config again
        fetchConfig();
    }
}

async function saveConfig() {
    // 清除之前的错误消息
    document.querySelectorAll(".error-message").forEach(el => el.remove());
    document.querySelectorAll(".error").forEach(el => el.classList.remove("error"));

    let isValid = true;

    try {
        updateConfigDataFromForm(); // Update configData

        const updatedConfig = configData.map((config, index) => {
            const container = configContainer.children[index];

            // Validate required fields
            ["site", "stream_name", "video_url"].forEach(field => {
                if (!config[field]) {
                    isValid = false;
                    const input = container.querySelector(`input[name='${field}']`);
                    input.classList.add("error");

                    // Add an error message if it doesn't exist
                    if (!input.previousElementSibling || !input.previousElementSibling.classList.contains("error-message")) {
                        const errorMessage = document.createElement("div");
                        errorMessage.className = "error-message";
                        errorMessage.textContent = "This field is required.";
                        input.parentNode.insertBefore(errorMessage, input);
                    }
                }
            });

            // Filter out notifications with empty token
            config.notifications = config.notifications.filter(notif => notif.token);

            // Ensure that the expire_date is set if no_expire_date is not checked
            if (!config.no_expire_date && !config.expire_date) {
                config.expire_date = new Date().toISOString().split('T')[0];
            }

            // Remove the previous_expire_date field
            delete config.previous_expire_date;

            return config;
        });

        if (!isValid) {
            // Not to save the configuration if it's invalid
            return;
        }

        // Transform notifications array to object
        const processedConfig = updatedConfig.map(config => {
            const notificationsObj = {};
            config.notifications.forEach(notif => {
                notificationsObj[notif.token] = notif.language;
            });
            return { ...config, notifications: notificationsObj };
        });

        // Send the updated configuration to the server
        const response = await fetch("/api/config", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ config: processedConfig }),
        });

        if (!response.ok) throw new Error("Failed to save configuration.");

        toggleEditMode(false); // Exit edit mode
    } catch (error) {
        console.error(error);
    }
}

// 按钮事件处理
editBtn.addEventListener("click", () => toggleEditMode(true));
cancelBtn.addEventListener("click", () => toggleEditMode(false)); // Exit edit mode
saveBtn.addEventListener("click", saveConfig);
addConfigBtn.addEventListener("click", () => {
    updateConfigDataFromForm();
    // Add a new config item with default values
    const today = new Date().toISOString().split('T')[0]; // Format: YYYY-MM-DD
    configData.push({
        video_url: "",
        site: "",
        stream_name: "",
        model_key: "yolo11n", // Default model key to yolo11n
        notifications: [], // Empty notifications
        expire_date: today,
        no_expire_date: false,
        detect_with_server: false, // Default detect_with_server to false
        detection_items: {
            "detect_no_safety_vest_or_helmet": false,
            "detect_near_machinery_or_vehicle": false,
            "detect_in_restricted_area": false
        }
    });

    // Re-render the form
    renderConfigForm();

    // Keep the form in edit mode
    toggleEditMode(true);
});

document.addEventListener("DOMContentLoaded", fetchConfig);
