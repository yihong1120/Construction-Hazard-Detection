/* ----------------------------------
   Global Variables & Element Selectors
------------------------------------- */
const configContainer = document.getElementById("config-container");
const editBtn = document.getElementById("edit-btn");
const addConfigBtn = document.getElementById("add-config-btn");
const saveBtn = document.getElementById("save-btn");
const cancelBtn = document.getElementById("cancel-btn");
const formControls = document.getElementById("form-controls");

let configData = [];
let isEditing = false;

/* ----------------------------------
   Utility Functions
------------------------------------- */
function getTodayDate() {
  // Fetch today's date in ISO format (YYYY-MM-DD)
  return new Date().toISOString().split("T")[0];
}

function formatDetectionItemName(key) {
  // Replace underscores with spaces and capitalize the first letter of each word
  return key.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

/* ----------------------------------
   Fetch & Render
------------------------------------- */
async function fetchConfig() {
  try {
    const response = await fetch("/api/config");
    if (!response.ok) throw new Error("Failed to fetch configuration.");
    const data = await response.json();

    // Transform notifications object to array
    configData = data.config.map((cfg) => ({
      ...cfg,
      notifications: Object.entries(cfg.notifications).map(([token, language]) => ({ token, language }))
    }));

    renderConfigForm();
  } catch (error) {
    console.error(error);
  }
}

/**
 * Render the entire configuration form
 */
function renderConfigForm() {
  configContainer.innerHTML = ""; // Clear existing config items

  // Loop through configData and create each config item
  configData.forEach((config, index) => {
    const container = createConfigItem(config, index);
    configContainer.appendChild(container);
  });
}

/**
 * Create a single config item DOM node
 */
function createConfigItem(config, index) {
  const configItemTemplate = document.getElementById("config-item-template");
  const container = configItemTemplate.content.cloneNode(true);
  const item = container.querySelector(".config-item");

  // Initialize form fields
  initFormFields(item, config, index);

  // Handle notifications
  initNotifications(item, config, index);

  // Handle No Expire Date checkbox (only visible in edit mode)
  handleExpireDateEditMode(item, config, index);

  // Show/hide delete & add-notification buttons based on edit mode
  toggleEditButtons(item, index);

  // Enable or disable fields based on isEditing
  toggleFormFields(item);

  return container;
}

/**
 * Initialize form fields with default or existing config values
 */
function initFormFields(item, config, index) {
  const siteInput = item.querySelector("input[name='site']");
  const streamNameInput = item.querySelector("input[name='stream_name']");
  const videoUrlInput = item.querySelector("input[name='video_url']");
  const modelKeySelect = item.querySelector("select[name='model_key']");
  const expireDateInput = item.querySelector("input[name='expire_date']");
  const noExpireDateText = item.querySelector("input[type='text'][value='No Expire Date']");
  const detectWithServerCheckbox = item.querySelector("input[name='detect_with_server']");
  const storeInRedisCheckbox = item.querySelector("input[name='store_in_redis']");
  const detectionItems = item.querySelectorAll(
    "input[type='checkbox'][name^='detect_']:not([name='detect_with_server']):not([name='store_in_redis'])"
  );
  const workStartHourSelect = item.querySelector("select[name='work_start_hour']");
  const workEndHourSelect = item.querySelector("select[name='work_end_hour']");

  // Default values
  siteInput.value = config.site || "";
  streamNameInput.value = config.stream_name || "";
  videoUrlInput.value = config.video_url || "";
  modelKeySelect.value = config.model_key || "yolo11n";

  // Work Hours
  workStartHourSelect.value =
    config.work_start_hour !== undefined ? config.work_start_hour : 7;
  workEndHourSelect.value =
    config.work_end_hour !== undefined ? config.work_end_hour : 18;

  // Expire Date
  if (config.expire_date === "No Expire Date") {
    expireDateInput.value = "";
    expireDateInput.style.display = "none";
    noExpireDateText.style.display = "";
  } else {
    expireDateInput.value = config.expire_date || getTodayDate();
    expireDateInput.style.display = "";
    noExpireDateText.style.display = "none";
  }

  // Detect with Server & Store in Redis
  detectWithServerCheckbox.checked = !!config.detect_with_server;
  storeInRedisCheckbox.checked = !!config.store_in_redis;

  // Detection items
  detectionItems.forEach((checkbox) => {
    const key = checkbox.name;
    checkbox.checked = !!config.detection_items[key];
    // Update label text
    const label = checkbox.parentElement;
    label.lastChild.textContent = formatDetectionItemName(key);
  });
}

/**
 * Initialize the notifications part of a config item
 */
function initNotifications(item, config, index) {
  const notificationItemTemplate = document.getElementById("notification-item-template");
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

    // Store notifIndex in data-attribute for later retrieval
    lineTokenInput.setAttribute("data-notif-index", notifIndex);
    languageSelect.setAttribute("data-notif-index", notifIndex);

    // Show or hide delete button based on edit mode
    deleteNotifBtn.style.display = isEditing ? "block" : "none";

    notificationsContainer.appendChild(notifEl);
  });

  // Add click listener for delete notification
  notificationsContainer.addEventListener("click", (event) => {
    if (event.target.closest(".delete-notification")) {
      const notificationItem = event.target.closest(".notification-item");
      const notifIndex = parseInt(
        notificationItem.querySelector("input[name='line_token']").getAttribute("data-notif-index"),
        10
      );
      updateConfigDataFromForm();
      const updatedConfig = configData[index];
      updatedConfig.notifications.splice(notifIndex, 1);
      renderConfigForm();
    }
  });

  // Handle add notification button
  const addNotificationBtn = item.querySelector(".add-notification");
  addNotificationBtn.addEventListener("click", () => {
    updateConfigDataFromForm();
    configData[index].notifications.push({ token: "", language: "en" });
    renderConfigForm();
  });
}

/**
 * Handle "No Expire Date" checkbox in edit mode
 */
function handleExpireDateEditMode(item, config, index) {
  const expireDateContainer = item.querySelector(".expire-date-container");
  const expireDateInput = item.querySelector("input[name='expire_date']");
  const noExpireDateText = item.querySelector("input[type='text'][value='No Expire Date']");

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

    // Initial state
    if (noExpireDateCheckbox.checked) {
      expireDateInput.style.display = "none";
      noExpireDateText.style.display = "";
    } else {
      expireDateInput.style.display = "";
      noExpireDateText.style.display = "none";
    }

    noExpireDateCheckbox.addEventListener("change", () => {
      if (noExpireDateCheckbox.checked) {
        expireDateInput.value = "";
        expireDateInput.style.display = "none";
        noExpireDateText.style.display = "";
      } else {
        expireDateInput.style.display = "";
        noExpireDateText.style.display = "none";
        // If expire_date is empty, set it to today
        if (!config.expire_date || config.expire_date === "No Expire Date") {
          expireDateInput.value = getTodayDate();
          config.expire_date = getTodayDate();
        }
      }
    });
  } else {
    // Show/hide "No Expire Date" text for non-edit mode
    if (config.expire_date === "No Expire Date") {
      expireDateInput.style.display = "none";
      noExpireDateText.style.display = "";
    } else {
      expireDateInput.style.display = "";
      noExpireDateText.style.display = "none";
    }
  }
}

/**
 * Show or hide the delete config & add notification buttons
 */
function toggleEditButtons(item) {
  const deleteConfigBtn = item.querySelector(".delete-config-btn");
  const addNotificationBtn = item.querySelector(".add-notification");

  if (isEditing) {
    deleteConfigBtn.style.display = "block";
    addNotificationBtn.style.display = "inline-block";
  } else {
    deleteConfigBtn.style.display = "none";
    addNotificationBtn.style.display = "none";
  }

  // Delete config event
  deleteConfigBtn.addEventListener("click", handleDeleteConfig);
}

/**
 * Handle delete config button
 */
function handleDeleteConfig(event) {
  updateConfigDataFromForm();
  // Find which index was clicked
  const item = event.currentTarget.closest(".config-item");
  const containerItems = Array.from(configContainer.children);
  const index = containerItems.indexOf(item);

  if (index >= 0) {
    configData.splice(index, 1);
    renderConfigForm();
  }
}

/**
 * Enable or disable input fields based on edit mode
 */
function toggleFormFields(item) {
  const fields = item.querySelectorAll("input, select");
  fields.forEach((f) => {
    if (!isEditing) {
      f.setAttribute("disabled", "true");
    } else {
      f.removeAttribute("disabled");
    }
  });
}

/* ----------------------------------
   Update & Validation
------------------------------------- */
/**
 * Update configData from the DOM form
 */
function updateConfigDataFromForm() {
  const configItems = configContainer.children;

  configData = Array.from(configItems).map((container) => {
    const inputs = container.querySelectorAll("input, select");
    const cfg = { notifications: [], detection_items: {} };

    inputs.forEach((input) => {
      const { name } = input;

      // Notifications
      if (name === "line_token" || name === "language") {
        const notifIndex = input.getAttribute("data-notif-index");
        if (!cfg.notifications[notifIndex]) {
          cfg.notifications[notifIndex] = { token: "", language: "en" };
        }
        if (name === "line_token") {
          cfg.notifications[notifIndex].token = input.value.trim();
        } else {
          cfg.notifications[notifIndex].language = input.value;
        }
      }
      // Expire Date
      else if (name === "expire_date") {
        cfg.expire_date = input.value.trim();
      }
      // Checkboxes
      else if (name === "detect_with_server") {
        cfg.detect_with_server = input.checked;
      } else if (name === "store_in_redis") {
        cfg.store_in_redis = input.checked;
      } else if (name.startsWith("detect_")) {
        cfg.detection_items[name] = input.checked;
      }
      // Work Hours
      else if (name === "work_start_hour") {
        cfg.work_start_hour = parseInt(input.value, 10);
      } else if (name === "work_end_hour") {
        cfg.work_end_hour = parseInt(input.value, 10);
      }
      // Other fields (site, stream_name, video_url, model_key, etc.)
      else if (name) {
        cfg[name] = input.value.trim();
      }
    });

    // Handle "No Expire Date" checkbox
    const noExpireDateCheckbox = container.querySelector("input[name='no_expire_date']");
    if (noExpireDateCheckbox) {
      if (noExpireDateCheckbox.checked) {
        cfg.expire_date = "No Expire Date";
      } else if (!cfg.expire_date || cfg.expire_date === "No Expire Date") {
        // If the expire_date is empty, set it to today
        cfg.expire_date = getTodayDate();
      }
    }

    // Remove empty notifications
    cfg.notifications = cfg.notifications.filter((notif) => notif.token);

    // Ensure store_in_redis is a boolean
    cfg.store_in_redis = !!cfg.store_in_redis;

    return cfg;
  });
}

/**
 * Validate each config item and return the final array
 */
function validateAndProcessUpdatedConfig() {
  let isValid = true;

  const updatedConfig = configData.map((cfg, idx) => {
    const container = configContainer.children[idx];

    // Validate required fields: site, stream_name, video_url
    ["site", "stream_name", "video_url"].forEach((field) => {
      if (!cfg[field]) {
        isValid = false;
        markFieldError(container, field, "This field is required.");
      }
    });

    // Validate work_start_hour < work_end_hour
    if (cfg.work_start_hour >= cfg.work_end_hour) {
      isValid = false;
      markWorkHourError(container, "Work Start Hour cannot be greater than or equal to Work End Hour.");
    }

    // Validate Expiry Date
    if (cfg.expire_date !== "No Expire Date" && !cfg.expire_date) {
      cfg.expire_date = getTodayDate();
    }

    return cfg;
  });

  return { updatedConfig, isValid };
}

/**
 * Mark a specific field as error
 */
function markFieldError(container, fieldName, message) {
  const input = container.querySelector(`input[name='${fieldName}']`);
  if (input) {
    input.classList.add("error");
    // Avoid duplicate error messages
    if (!input.previousElementSibling || !input.previousElementSibling.classList.contains("error-message")) {
      const errorMessage = document.createElement("div");
      errorMessage.className = "error-message";
      errorMessage.textContent = message;
      input.parentNode.insertBefore(errorMessage, input);
    }
  }
}

/**
 * Mark work hour fields as error
 */
function markWorkHourError(container, message) {
  const workStartHourSelect = container.querySelector("select[name='work_start_hour']");
  const workEndHourSelect = container.querySelector("select[name='work_end_hour']");
  workStartHourSelect.classList.add("error");
  workEndHourSelect.classList.add("error");

  if (
    !workEndHourSelect.previousElementSibling ||
    !workEndHourSelect.previousElementSibling.classList.contains("error-message")
  ) {
    const errorMessage = document.createElement("div");
    errorMessage.className = "error-message";
    errorMessage.textContent = message;
    workEndHourSelect.parentNode.insertBefore(errorMessage, workEndHourSelect);
  }
}

/**
 * Transform configData to final format and POST to backend
 */
async function saveConfig() {
  // Erase previous error messages
  document.querySelectorAll(".error-message").forEach((el) => el.remove());
  document.querySelectorAll(".error").forEach((el) => el.classList.remove("error"));

  try {
    updateConfigDataFromForm();
    const { updatedConfig, isValid } = validateAndProcessUpdatedConfig();

    if (!isValid) {
      return; // Stop if validation fails
    }

    // Convert notifications array to object
    const finalConfig = updatedConfig.map((cfg) => {
      const notificationsObj = {};
      cfg.notifications.forEach((notif) => {
        notificationsObj[notif.token] = notif.language;
      });
      return {
        ...cfg,
        notifications: notificationsObj,
        no_expire_date: undefined // remove the "no_expire_date" from final JSON
      };
    });

    // Remove undefined fields
    finalConfig.forEach((cfg) => {
      Object.keys(cfg).forEach((key) => {
        if (cfg[key] === undefined) {
          delete cfg[key];
        }
      });
    });

    // Save to backend
    const response = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config: finalConfig })
    });

    if (!response.ok) throw new Error("Failed to save configuration.");
    toggleEditMode(false);
  } catch (error) {
    console.error(error);
  }
}

/* ----------------------------------
   Edit Mode Toggle
------------------------------------- */
function toggleEditMode(enable) {
  isEditing = enable;
  renderConfigForm();

  // Show/hide buttons and form controls
  editBtn.classList.toggle("hidden", enable);
  addConfigBtn.classList.toggle("hidden", !enable);
  formControls.classList.toggle("hidden", !enable);

  if (!enable) {
    // If exiting edit mode, re-fetch the latest config
    fetchConfig();
  }
}

/* ----------------------------------
   Event Listeners
------------------------------------- */
editBtn.addEventListener("click", () => toggleEditMode(true));
cancelBtn.addEventListener("click", () => toggleEditMode(false));
saveBtn.addEventListener("click", saveConfig);
addConfigBtn.addEventListener("click", () => {
  updateConfigDataFromForm();
  // Add a new empty config
  configData.push({
    site: "",
    stream_name: "",
    video_url: "",
    model_key: "yolo11n",
    expire_date: getTodayDate(), // default
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

// Automatically fetch config on page load
document.addEventListener("DOMContentLoaded", fetchConfig);
