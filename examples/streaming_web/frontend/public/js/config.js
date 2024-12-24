const configContainer = document.getElementById("config-container");
const editBtn = document.getElementById("edit-btn");
const addConfigBtn = document.getElementById("add-config-btn");
const saveBtn = document.getElementById("save-btn");
const cancelBtn = document.getElementById("cancel-btn");
const formControls = document.getElementById("form-controls");

let configData = [];
let isEditing = false;

// Fetch today's date in ISO format (YYYY-MM-DD)
function getTodayDate() {
    return new Date().toISOString().split("T")[0];
}

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
            const { name, type } = input;

            // Process Notifications
            if (name === "line_token" || name === "language") {
                const notifIndex = input.getAttribute("data-notif-index");
                if (!config.notifications[notifIndex]) {
                    config.notifications[notifIndex] = { token: "", language: "en" };
                }
                if (name === "line_token") {
                    config.notifications[notifIndex].token = input.value.trim();
                } else {
                    config.notifications[notifIndex].language = input.value;
                }
            }

            // Process Expiry Date
            else if (name === "expire_date") {
                config.expire_date = input.value.trim();
            }

            // Process Checkboxes
            else if (name === "detect_with_server") {
                config.detect_with_server = input.checked;
            }
            else if (name === "store_in_redis") {
                config.store_in_redis = input.checked;
            }
            else if (name.startsWith("detect_")) {
                config.detection_items[name] = input.checked;
            }

            // Process Work Hours
            else if (name === "work_start_hour") {
                config.work_start_hour = parseInt(input.value, 10);
            }
            else if (name === "work_end_hour") {
                config.work_end_hour = parseInt(input.value, 10);
            }

            // Process other fields
            else if (name) {
                // site, stream_name, video_url, model_key, etc.
                config[name] = input.value.trim();
            }
        });

        // Process "No Expire Date"
        const noExpireDateCheckbox = container.querySelector("input[name='no_expire_date']");
        if (noExpireDateCheckbox) {
            if (noExpireDateCheckbox.checked) {
                config.expire_date = "No Expire Date";
            } else {
                // If the expire_date is empty, set it to today
                if (!config.expire_date || config.expire_date === "No Expire Date") {
                    config.expire_date = getTodayDate();
                }
            }
        }

        // Remove empty notifications
        config.notifications = config.notifications.filter(notif => notif.token);

        // Ensure store_in_redis is a boolean
        config.store_in_redis = !!config.store_in_redis;

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

        // Fetch form elements
        const siteInput = item.querySelector("input[name='site']");
        const streamNameInput = item.querySelector("input[name='stream_name']");
        const videoUrlInput = item.querySelector("input[name='video_url']");
        const modelKeySelect = item.querySelector("select[name='model_key']");
        const expireDateInput = item.querySelector("input[name='expire_date']");
        const noExpireDateText = item.querySelector("input[type='text'][value='No Expire Date']");
        const detectWithServerCheckbox = item.querySelector("input[name='detect_with_server']");
        const storeInRedisCheckbox = item.querySelector("input[name='store_in_redis']");
        const detectionItems = item.querySelectorAll("input[type='checkbox'][name^='detect_']:not([name='detect_with_server']):not([name='store_in_redis'])");
        const workStartHourSelect = item.querySelector("select[name='work_start_hour']");
        const workEndHourSelect = item.querySelector("select[name='work_end_hour']");
        const addNotificationBtn = item.querySelector(".add-notification");
        const deleteConfigBtn = item.querySelector(".delete-config-btn");
        const expireDateContainer = item.querySelector(".expire-date-container");

        // Default values
        siteInput.value = config.site || '';
        streamNameInput.value = config.stream_name || '';
        videoUrlInput.value = config.video_url || '';
        modelKeySelect.value = config.model_key || 'yolo11n';

        // Set Work Hours
        workStartHourSelect.value = config.work_start_hour !== undefined ? config.work_start_hour : 7;
        workEndHourSelect.value = config.work_end_hour !== undefined ? config.work_end_hour : 18;

        // Set Expire Date
        if (config.expire_date === "No Expire Date") {
            expireDateInput.value = "";
            expireDateInput.style.display = "none";
            noExpireDateText.style.display = "";
        } else {
            expireDateInput.value = config.expire_date || getTodayDate();
            expireDateInput.style.display = "";
            noExpireDateText.style.display = "none";
        }

        // Set Detect with Server & Store in Redis
        detectWithServerCheckbox.checked = !!config.detect_with_server;
        storeInRedisCheckbox.checked = !!config.store_in_redis;

        // Set Detection Items
        detectionItems.forEach((checkbox) => {
            const key = checkbox.name;
            checkbox.checked = !!config.detection_items[key];
            // Update the label text, replacing underscores with spaces and capitalizing the first letter of each word
            const label = checkbox.parentElement;
            label.lastChild.textContent = formatDetectionItemName(key);
        });

        // Set Notifications
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

            // According to the edit mode, show/hide delete button
            if (isEditing) {
                deleteNotifBtn.style.display = "block";
            } else {
                deleteNotifBtn.style.display = "none";
            }

            notificationsContainer.appendChild(notifEl);
        });

        // If the expire_date is empty, set it to today
        if (isEditing) {
            const noExpireDateCheckbox = document.createElement("input");
            noExpireDateCheckbox.type = "checkbox";
            noExpireDateCheckbox.name = "no_expire_date";
            noExpireDateCheckbox.checked = config.expire_date === "No Expire Date";
            noExpireDateCheckbox.id = `no-expire-date-${index}`;

            const noExpireDateLabel = document.createElement("label");
            noExpireDateLabel.htmlFor = `no-expire-date-${index}`;
            noExpireDateLabel.appendChild(noExpireDateCheckbox);
            noExpireDateLabel.appendChild(document.createTextNode(" No Expire Date"));

            expireDateContainer.appendChild(document.createElement("br"));
            expireDateContainer.appendChild(noExpireDateLabel);

            // Initial show/hide "No Expire Date" text
            if (noExpireDateCheckbox.checked) {
                expireDateInput.style.display = "none";
                noExpireDateText.style.display = "";
            } else {
                expireDateInput.style.display = "";
                noExpireDateText.style.display = "none";
            }

            // Monitor "No Expire Date" checkbox changes
            noExpireDateCheckbox.addEventListener("change", () => {
                if (noExpireDateCheckbox.checked) {
                    expireDateInput.value = "";
                    expireDateInput.style.display = "none";
                    noExpireDateText.style.display = "";
                } else {
                    expireDateInput.style.display = "";
                    noExpireDateText.style.display = "none";
                    // If the expire_date is empty, set it to today
                    if (!config.expire_date || config.expire_date === "No Expire Date") {
                        expireDateInput.value = getTodayDate();
                        config.expire_date = getTodayDate();
                    }
                }
            });
        } else {
            // Show/hide "No Expire Date" text
            if (config.expire_date === "No Expire Date") {
                expireDateInput.style.display = "none";
                noExpireDateText.style.display = "";
            } else {
                expireDateInput.style.display = "";
                noExpireDateText.style.display = "none";
            }
        }

        // According to the edit mode, show/hide buttons
        if (isEditing) {
            deleteConfigBtn.style.display = "block";
            addNotificationBtn.style.display = "inline-block";
        } else {
            deleteConfigBtn.style.display = "none";
            addNotificationBtn.style.display = "none";
        }

        // Event: Delete Config
        deleteConfigBtn.addEventListener("click", () => {
            updateConfigDataFromForm();
            configData.splice(index, 1);
            renderConfigForm();
        });

        // Event: Delete Notification
        notificationsContainer.addEventListener("click", (event) => {
            if (event.target.closest(".delete-notification")) {
                const notificationItem = event.target.closest(".notification-item");
                const notifIndex = parseInt(
                    notificationItem.querySelector("input[name='line_token']").getAttribute("data-notif-index")
                );
                updateConfigDataFromForm();
                const updatedConfig = configData[index];
                updatedConfig.notifications.splice(notifIndex, 1);
                renderConfigForm();
            }
        });

        // Event: Add Notification
        addNotificationBtn.addEventListener("click", () => {
            updateConfigDataFromForm();
            configData[index].notifications.push({ token: "", language: "en" });
            renderConfigForm();
        });

        // According to the edit mode, enable/disable form fields
        const fields = item.querySelectorAll("input, select");
        fields.forEach(f => {
            if (!isEditing) {
                f.setAttribute("disabled", "true");
            } else {
                f.removeAttribute("disabled");
            }
        });

        configContainer.appendChild(container);
    });
}

function toggleEditMode(enable) {
    isEditing = enable;

    // Reset form data if exiting edit mode
    renderConfigForm();

    // Show/hide buttons and form controls
    editBtn.classList.toggle("hidden", enable);
    addConfigBtn.classList.toggle("hidden", !enable);
    formControls.classList.toggle("hidden", !enable);

    if (!enable) {
        // If exiting edit mode, fetch the latest config
        fetchConfig();
    }
}

async function saveConfig() {
    // Erase previous error messages
    document.querySelectorAll(".error-message").forEach(el => el.remove());
    document.querySelectorAll(".error").forEach(el => el.classList.remove("error"));

    let isValid = true;

    try {
        updateConfigDataFromForm(); // Update configData with the latest form values

        const updatedConfig = configData.map((config, index) => {
            const container = configContainer.children[index];

            // Validate required fields: site, stream_name, video_url
            ["site", "stream_name", "video_url"].forEach(field => {
                if (!config[field]) {
                    isValid = false;
                    const input = container.querySelector(`input[name='${field}']`);
                    if (input) {
                        input.classList.add("error");
                        if (
                            !input.previousElementSibling ||
                            !input.previousElementSibling.classList.contains("error-message")
                        ) {
                            const errorMessage = document.createElement("div");
                            errorMessage.className = "error-message";
                            errorMessage.textContent = "This field is required.";
                            input.parentNode.insertBefore(errorMessage, input);
                        }
                    }
                }
            });

            // Validate work_start_hour < work_end_hour
            if (config.work_start_hour >= config.work_end_hour) {
                isValid = false;
                const workStartHourSelect = container.querySelector(`select[name='work_start_hour']`);
                const workEndHourSelect = container.querySelector(`select[name='work_end_hour']`);
                workStartHourSelect.classList.add("error");
                workEndHourSelect.classList.add("error");

                if (
                    !workEndHourSelect.previousElementSibling ||
                    !workEndHourSelect.previousElementSibling.classList.contains("error-message")
                ) {
                    const errorMessage = document.createElement("div");
                    errorMessage.className = "error-message";
                    errorMessage.textContent =
                        "Work Start Hour cannot be greater than or equal to Work End Hour.";
                    workEndHourSelect.parentNode.insertBefore(errorMessage, workEndHourSelect);
                }
            }

            // Validate Expiry Date
            if (config.expire_date !== "No Expire Date" && !config.expire_date) {
                config.expire_date = getTodayDate();
            }

            return config;
        });

        if (!isValid) {
            return; // No need to proceed if there are validation errors
        }

        // Set notifications as an object
        const processedConfig = updatedConfig.map(config => {
            const notificationsObj = {};
            config.notifications.forEach(notif => {
                notificationsObj[notif.token] = notif.language;
            });
            return {
                ...config,
                notifications: notificationsObj,
                // Remove the "No Expire Date" checkbox value
                no_expire_date: undefined
            };
        });

        // Remove undefined fields
        const finalConfig = processedConfig.map(config => {
            const cleanedConfig = { ...config };
            Object.keys(cleanedConfig).forEach(key => {
                if (cleanedConfig[key] === undefined) {
                    delete cleanedConfig[key];
                }
            });
            return cleanedConfig;
        });

        // Conduct the API request to save the configuration
        const response = await fetch("/api/config", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ config: finalConfig })
        });

        if (!response.ok) throw new Error("Failed to save configuration.");

        toggleEditMode(false); // Exit edit mode
    } catch (error) {
        console.error(error);
    }
}

// Process Events
editBtn.addEventListener("click", () => toggleEditMode(true));
cancelBtn.addEventListener("click", () => toggleEditMode(false));
saveBtn.addEventListener("click", saveConfig);
addConfigBtn.addEventListener("click", () => {
    updateConfigDataFromForm();
    // Add a new empty config item
    const today = getTodayDate();
    configData.push({
        site: "",
        stream_name: "",
        video_url: "",
        model_key: "yolo11n",
        expire_date: today, // Default to today
        detect_with_server: false,
        store_in_redis: false,
        work_start_hour: 7,
        work_end_hour: 18,
        notifications: [],
        detection_items: {
            detect_no_safety_vest_or_helmet: false,
            detect_near_machinery_or_vehicle: false,
            detect_in_restricted_area: false
        }
    });

    renderConfigForm();
    toggleEditMode(true);
});

// Automatically fetch the configuration when the page loads
document.addEventListener("DOMContentLoaded", fetchConfig);
