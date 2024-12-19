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

    const configItemTemplate = document.getElementById("config-item-template");
    const notificationItemTemplate = document.getElementById("notification-item-template");

    configData.forEach((config, index) => {
        const container = configItemTemplate.content.cloneNode(true);
        const item = container.querySelector(".config-item");

        const siteInput = item.querySelector("input[name='site']");
        const streamNameInput = item.querySelector("input[name='stream_name']");
        const videoUrlInput = item.querySelector("input[name='video_url']");
        const modelKeySelect = item.querySelector("select[name='model_key']");
        const expireDateInput = item.querySelector("input[name='expire_date']");
        const noExpireDateText = item.querySelector("input[type='text'][value='No Expire Date']");
        const detectWithServerCheckbox = item.querySelector("input[name='detect_with_server']");
        const detectionItems = item.querySelectorAll("input[type='checkbox'][name^='detect_']:not([name='detect_with_server'])");
        const addNotificationBtn = item.querySelector(".add-notification");
        const deleteConfigBtn = item.querySelector(".delete-config-btn");
        const expireDateContainer = item.querySelector(".expire-date-container");

        // Set values
        siteInput.value = config.site || '';
        streamNameInput.value = config.stream_name || '';
        videoUrlInput.value = config.video_url || '';
        if (modelKeySelect) {
            modelKeySelect.value = config.model_key || 'yolo11n';
        }

        // Process expire_date
        let expireDateValue;
        if (config.expire_date === "No Expire Date") {
            expireDateValue = "";
        } else if (config.expire_date) {
            expireDateValue = config.expire_date;
        } else {
            expireDateValue = new Date().toISOString().split('T')[0];
            config.expire_date = expireDateValue;
        }

        expireDateInput.value = expireDateValue;

        // Detect with server
        detectWithServerCheckbox.checked = !!config.detect_with_server;

        // Detection Items
        detectionItems.forEach((checkbox) => {
            const key = checkbox.name;
            checkbox.checked = !!config.detection_items[key];
            // Update label text in case keys differ (optional)
            const label = checkbox.parentElement;
            label.lastChild.textContent = formatDetectionItemName(key);
        });

        // Notifications
        const notificationsContainer = item.querySelector(".notifications-container");
        notificationsContainer.innerHTML = "";
        config.notifications.forEach((notification, notifIndex) => {
            const notifEl = notificationItemTemplate.content.cloneNode(true);
            const notifItem = notifEl.querySelector(".notification-item");
            const lineTokenInput = notifItem.querySelector("input[name='line_token']");
            const languageSelect = notifItem.querySelector("select[name='language']");
            const deleteNotifBtn = notifItem.querySelector(".delete-notification");

            lineTokenInput.value = notification.token;
            languageSelect.value = notification.language;
            lineTokenInput.setAttribute("data-notif-index", notifIndex);
            languageSelect.setAttribute("data-notif-index", notifIndex);

            // Show/Hide delete notification button based on edit mode
            if (isEditing) {
                deleteNotifBtn.style.display = "block";
            } else {
                deleteNotifBtn.style.display = "none";
            }

            notificationsContainer.appendChild(notifEl);
        });

        // If in editing mode, add a no_expire_date checkbox
        if (isEditing) {
            const noExpireDateCheckbox = document.createElement("input");
            noExpireDateCheckbox.type = "checkbox";
            noExpireDateCheckbox.name = "no_expire_date";
            noExpireDateCheckbox.checked = config.expire_date === "No Expire Date";
            expireDateContainer.appendChild(document.createElement("br"));
            const noExpireDateLabel = document.createElement("label");
            noExpireDateLabel.appendChild(noExpireDateCheckbox);
            noExpireDateLabel.appendChild(document.createTextNode(" No Expire Date"));
            expireDateContainer.appendChild(noExpireDateLabel);

            // Initialise the display of the expire date input and no expire date text
            if (noExpireDateCheckbox.checked) {
                expireDateInput.style.display = "none";
                noExpireDateText.style.display = "";
            } else {
                expireDateInput.style.display = "";
                noExpireDateText.style.display = "none";
            }

            noExpireDateCheckbox.addEventListener("change", () => {
                if (noExpireDateCheckbox.checked) {
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
        } else {
            // Not in editing mode
            if (config.expire_date === "No Expire Date") {
                expireDateInput.style.display = "none";
                noExpireDateText.style.display = "";
            } else {
                expireDateInput.style.display = "";
                noExpireDateText.style.display = "none";
            }
        }

        // Show/Hide delete config button and add notification button based on edit mode
        if (isEditing) {
            deleteConfigBtn.style.display = "block";
            addNotificationBtn.style.display = "inline-block";
        } else {
            deleteConfigBtn.style.display = "none";
            addNotificationBtn.style.display = "none";
        }

        // Event: delete config
        deleteConfigBtn.addEventListener("click", () => {
            updateConfigDataFromForm();
            configData.splice(index, 1);
            renderConfigForm();
        });

        // Event: delete notification
        notificationsContainer.addEventListener("click", (event) => {
            if (event.target.closest(".delete-notification")) {
                const notificationItem = event.target.closest(".notification-item");
                const notifIndex = parseInt(notificationItem.querySelector("input[name='line_token']").getAttribute("data-notif-index"));
                updateConfigDataFromForm();
                const updatedConfig = configData[index];
                updatedConfig.notifications.splice(notifIndex, 1);
                renderConfigForm();
            }
        });

        // Event: add notification
        addNotificationBtn.addEventListener("click", () => {
            updateConfigDataFromForm();
            configData[index].notifications.push({ token: "", language: "en" });
            renderConfigForm();
        });

        // 根據isEditing狀態設定所有input,select的disabled狀態
        const fields = item.querySelectorAll('input, select');
        fields.forEach(f => {
            if (!isEditing) {
                f.setAttribute('disabled', 'true');
            } else {
                f.removeAttribute('disabled');
            }
        });

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

// 按鈕事件處理
editBtn.addEventListener("click", () => toggleEditMode(true));
cancelBtn.addEventListener("click", () => toggleEditMode(false));
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

    renderConfigForm();
    toggleEditMode(true);
});

document.addEventListener("DOMContentLoaded", fetchConfig);
