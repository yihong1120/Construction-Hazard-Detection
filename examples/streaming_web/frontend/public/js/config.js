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
  // Replace underscores with spaces and capitalise each word's first letter
  return key.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

// Custom logging function to avoid direct console usage
function logError(error) {
  // Example: Send error to external logging service or remove for production
  // console.error(error);
}

/* ----------------------------------
   Fetch & Render
------------------------------------- */
async function fetchConfig() {
  try {
    const response = await fetch("/api/config");
    if (!response.ok) {
      throw new Error("Failed to fetch configuration.");
    }
    const data = await response.json();

    // Transform notifications object to array
    configData = data.config.map((cfg) => ({
      ...cfg,
      notifications: Object.entries(cfg.notifications).map(([token, language]) => ({
        token,
        language
      }))
    }));

    renderConfigForm();
  } catch (error) {
    logError(error); // Replaces console.error
  }
}

/**
 * Render the entire configuration form
 */
function renderConfigForm() {
  configContainer.innerHTML = ""; // Clear existing config items

  configData.forEach((config, idx) => {
    const container = createConfigItem(config, idx);
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

  initFormFields(item, config);
  initNotifications(item, config, index);
  handleExpireDateEditMode(item, config);
  toggleEditButtons(item);
  toggleFormFields(item);

  return container;
}

/* ----------------------------------
   Form Field Initialisation
------------------------------------- */
/**
 * Initialise form fields with default or existing config values
 */
function initFormFields(item, config) {
  setBasicFields(item, config);
  setWorkHours(item, config);
  setExpireDateUI(item, config);
  setDetectAndRedisFlags(item, config);
  setDetectionItemsUI(item, config);
}

/**
 * Set basic text/select fields like site, stream_name, video_url, model_key
 */
function setBasicFields(item, config) {
  const siteInput = item.querySelector("input[name='site']");
  const streamNameInput = item.querySelector("input[name='stream_name']");
  const videoUrlInput = item.querySelector("input[name='video_url']");
  const modelKeySelect = item.querySelector("select[name='model_key']");

  siteInput.value = config.site || "";
  streamNameInput.value = config.stream_name || "";
  videoUrlInput.value = config.video_url || "";
  modelKeySelect.value = config.model_key || "yolo11n";
}

/**
 * Set work hours (start/end)
 */
function setWorkHours(item, config) {
  const workStartHourSelect = item.querySelector("select[name='work_start_hour']");
  const workEndHourSelect = item.querySelector("select[name='work_end_hour']");

  workStartHourSelect.value =
    config.work_start_hour !== undefined ? config.work_start_hour : 7;
  workEndHourSelect.value =
    config.work_end_hour !== undefined ? config.work_end_hour : 18;
}

/**
 * Set expire date UI (text input vs. "No Expire Date" text)
 */
function setExpireDateUI(item, config) {
  const expireDateInput = item.querySelector("input[name='expire_date']");
  const noExpireDateText = item.querySelector("input[type='text'][value='No Expire Date']");

  if (config.expire_date === "No Expire Date") {
    expireDateInput.value = "";
    expireDateInput.style.display = "none";
    noExpireDateText.style.display = "";
  } else {
    expireDateInput.value = config.expire_date || getTodayDate();
    expireDateInput.style.display = "";
    noExpireDateText.style.display = "none";
  }
}

/**
 * Set detect_with_server & store_in_redis checkboxes
 */
function setDetectAndRedisFlags(item, config) {
  const detectWithServerCheckbox = item.querySelector("input[name='detect_with_server']");
  const storeInRedisCheckbox = item.querySelector("input[name='store_in_redis']");

  detectWithServerCheckbox.checked = !!config.detect_with_server;
  storeInRedisCheckbox.checked = !!config.store_in_redis;
}

/**
 * Set detection items (checkboxes + label)
 */
function setDetectionItemsUI(item, config) {
  const detectionItems = item.querySelectorAll(
    "input[type='checkbox'][name^='detect_']:not([name='detect_with_server']):not([name='store_in_redis'])"
  );

  detectionItems.forEach((checkbox) => {
    checkbox.checked = !!config.detection_items[checkbox.name];
    // Update label text
    const label = checkbox.parentElement;
    label.lastChild.textContent = formatDetectionItemName(checkbox.name);
  });
}

/* ----------------------------------
   Notifications
------------------------------------- */
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
    lineTokenInput.setAttribute("data-notif-index", notifIndex);
    languageSelect.setAttribute("data-notif-index", notifIndex);

    deleteNotifBtn.style.display = isEditing ? "block" : "none";
    notificationsContainer.appendChild(notifEl);
  });

  setupNotificationEvents(notificationsContainer, index);
}

/**
 * Set up notification event listeners
 *
 * @param {HTMLElement} notificationsContainer - The container for notifications.
 * @param {number} configIndex - The index of the config in configData.
 */
function setupNotificationEvents(notificationsContainer, configIndex) {
  // Delete notification event
  notificationsContainer.addEventListener("click", (event) => {
    if (event.target.closest(".delete-notification")) {
      const notificationItem = event.target.closest(".notification-item");
      const notifIndex = parseInt(
        notificationItem.querySelector("input[name='line_token']").getAttribute("data-notif-index"),
        10
      );
      removeNotification(configIndex, notifIndex);
    }
  });

  // Add notification event
  const addNotificationBtn = notificationsContainer.parentElement.querySelector(".add-notification");
  addNotificationBtn.addEventListener("click", () => {
    addNotification(configIndex);
  });
}

/**
 * Add a new notification
 *
 * @param {number} configIndex - The index of the config in configData.
 */
function addNotification(configIndex) {
  updateConfigDataFromForm();
  configData[configIndex].notifications.push({ token: "", language: "en" });
  renderConfigForm();
}

/**
 * Remove a notification
 *
 * @param {number} configIndex - The index of the config in configData.
 * @param {number} notifIndex - The index of the notification to remove.
 */
function removeNotification(configIndex, notifIndex) {
  updateConfigDataFromForm();
  configData[configIndex].notifications.splice(notifIndex, 1);
  renderConfigForm();
}

/* ----------------------------------
   Expire Date Edit Mode
------------------------------------- */
function handleExpireDateEditMode(item, config) {
  if (isEditing) {
    createNoExpireDateCheckbox(item, config);
  } else {
    handleNonEditExpireDate(item, config);
  }
}

/**
 * Create & handle "No Expire Date" checkbox in edit mode
 */
function createNoExpireDateCheckbox(item, config) {
  const expireDateContainer = item.querySelector(".expire-date-container");
  const expireDateInput = item.querySelector("input[name='expire_date']");
  const noExpireDateText = item.querySelector("input[type='text'][value='No Expire Date']");

  const noExpireDateCheckbox = document.createElement("input");
  noExpireDateCheckbox.type = "checkbox";
  noExpireDateCheckbox.name = "no_expire_date";
  noExpireDateCheckbox.checked = config.expire_date === "No Expire Date";
  noExpireDateCheckbox.id = `no-expire-date-${Math.random()}`; // or other unique ID

  const noExpireDateLabel = document.createElement("label");
  noExpireDateLabel.htmlFor = noExpireDateCheckbox.id;
  noExpireDateLabel.appendChild(noExpireDateCheckbox);
  noExpireDateLabel.appendChild(document.createTextNode(" No Expire Date"));

  expireDateContainer.appendChild(document.createElement("br"));
  expireDateContainer.appendChild(noExpireDateLabel);

  // Initial state
  toggleNoExpireDate(expireDateInput, noExpireDateText, noExpireDateCheckbox.checked);

  noExpireDateCheckbox.addEventListener("change", () => {
    toggleNoExpireDate(expireDateInput, noExpireDateText, noExpireDateCheckbox.checked);
    // If the expire_date is empty, set it to today
    if (!config.expire_date || config.expire_date === "No Expire Date") {
      expireDateInput.value = getTodayDate();
      config.expire_date = getTodayDate();
    }
  });
}

/**
 * Toggle visibility of expire date input and "No Expire Date" text
 *
 * @param {HTMLInputElement} expireDateInput - The expire date input field.
 * @param {HTMLInputElement} noExpireDateText - The "No Expire Date" text input.
 * @param {boolean} isNoExpire - Whether "No Expire Date" is selected.
 */
function toggleNoExpireDate(expireDateInput, noExpireDateText, isNoExpire) {
  if (isNoExpire) {
    expireDateInput.value = "";
    expireDateInput.style.display = "none";
    noExpireDateText.style.display = "";
  } else {
    expireDateInput.style.display = "";
    noExpireDateText.style.display = "none";
  }
}

/**
 * Show/hide "No Expire Date" text for non-edit mode
 */
function handleNonEditExpireDate(item, config) {
  const expireDateInput = item.querySelector("input[name='expire_date']");
  const noExpireDateText = item.querySelector("input[type='text'][value='No Expire Date']");

  toggleNoExpireDate(expireDateInput, noExpireDateText, config.expire_date === "No Expire Date");
}

/* ----------------------------------
   Buttons & Edit Mode
------------------------------------- */
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

  deleteConfigBtn.addEventListener("click", () => handleDeleteConfig(item));
}

/**
 * Handle delete config
 */
function handleDeleteConfig(item) {
  updateConfigDataFromForm();
  const containerItems = Array.from(configContainer.children);
  const idx = containerItems.indexOf(item);

  if (idx >= 0) {
    configData.splice(idx, 1);
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
function updateConfigDataFromForm() {
  const configItems = configContainer.children;

  configData = Array.from(configItems).map((container) =>
    buildSingleConfigFromDOM(container)
  );
}

/**
 * Build a single config object from DOM elements
 */
function buildSingleConfigFromDOM(container) {
  const inputs = container.querySelectorAll("input, select");
  const cfg = {
    notifications: [],
    detection_items: {}
  };

  inputs.forEach((input) => processInputField(input, cfg));
  handleNoExpireCheckbox(container, cfg);

  // Remove empty notifications
  cfg.notifications = cfg.notifications.filter((notif) => notif.token);
  // Ensure store_in_redis is boolean
  cfg.store_in_redis = !!cfg.store_in_redis;

  return cfg;
}

/**
 * Process each input/select field
 */
function processInputField(input, cfg) {
  const { name } = input;
  if (!name) return;

  switch (name) {
    case "line_token":
    case "language":
      updateNotificationField(input, cfg);
      break;
    case "expire_date":
      cfg.expire_date = input.value.trim();
      break;
    case "detect_with_server":
      cfg.detect_with_server = input.checked;
      break;
    case "store_in_redis":
      cfg.store_in_redis = input.checked;
      break;
    default:
      if (name.startsWith("detect_")) {
        cfg.detection_items[name] = input.checked;
      } else if (name === "work_start_hour") {
        cfg.work_start_hour = parseInt(input.value, 10);
      } else if (name === "work_end_hour") {
        cfg.work_end_hour = parseInt(input.value, 10);
      } else {
        // site, stream_name, video_url, model_key, etc.
        cfg[name] = input.value.trim();
      }
  }
}

/**
 * Update notifications array (line_token / language)
 */
function updateNotificationField(input, cfg) {
  const notifIndex = input.getAttribute("data-notif-index");
  if (!cfg.notifications[notifIndex]) {
    cfg.notifications[notifIndex] = { token: "", language: "en" };
  }
  if (input.name === "line_token") {
    cfg.notifications[notifIndex].token = input.value.trim();
  } else {
    cfg.notifications[notifIndex].language = input.value;
  }
}

/**
 * Handle "No Expire Date" checkbox
 */
function handleNoExpireCheckbox(container, cfg) {
  const noExpireDateCheckbox = container.querySelector("input[name='no_expire_date']");
  if (noExpireDateCheckbox) {
    if (noExpireDateCheckbox.checked) {
      cfg.expire_date = "No Expire Date";
    } else if (!cfg.expire_date || cfg.expire_date === "No Expire Date") {
      cfg.expire_date = getTodayDate();
    }
  }
}

/**
 * Validate each config item and return the final array
 */
function validateAndProcessUpdatedConfig() {
  let isValid = true;

  function validateRequiredFields(container, cfg, requiredFields) {
    requiredFields.forEach((field) => {
      if (!cfg[field]) {
        isValid = false;
        markFieldError(container, field, "This field is required.");
      }
    });
  }

  function validateWorkHours(container, cfg) {
    if (cfg.work_start_hour >= cfg.work_end_hour) {
      isValid = false;
      markWorkHourError(container, "Work Start Hour cannot be greater than or equal to Work End Hour.");
    }
  }

  function setDefaultExpireDate(cfg) {
    if (cfg.expire_date !== "No Expire Date" && !cfg.expire_date) {
      cfg.expire_date = getTodayDate();
    }
  }

  configData.forEach((cfg, idx) => {
    const container = configContainer.children[idx];

    validateRequiredFields(container, cfg, ["site", "stream_name", "video_url"]);
    validateWorkHours(container, cfg);
    setDefaultExpireDate(cfg);
  });

  return { updatedConfig: configData, isValid };
}

/**
 * Mark a specific field as error
 */
function markFieldError(container, fieldName, message) {
  const input = container.querySelector(`input[name='${fieldName}']`);
  if (!input) return;

  input.classList.add("error");
  if (!input.previousElementSibling || !input.previousElementSibling.classList.contains("error-message")) {
    const errorMessage = document.createElement("div");
    errorMessage.className = "error-message";
    errorMessage.textContent = message;
    input.parentNode.insertBefore(errorMessage, input);
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

/* ----------------------------------
   Save Config
------------------------------------- */
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

    const finalConfig = convertNotificationsArrayToObj(updatedConfig);
    removeUndefinedFields(finalConfig);

    // Save to backend
    const response = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config: finalConfig })
    });

    if (!response.ok) throw new Error("Failed to save configuration.");
    toggleEditMode(false);
  } catch (error) {
    logError(error); // Replaces console.error
  }
}

/**
 * Convert notifications array to object for each config
 */
function convertNotificationsArrayToObj(configArray) {
  return configArray.map((cfg) => {
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
}

/**
 * Remove undefined fields from final config
 */
function removeUndefinedFields(finalConfig) {
  for (const cfg of finalConfig) {
    for (const key in cfg) {
      if (cfg[key] === undefined) {
        delete cfg[key];
      }
    }
  }
}

/* ----------------------------------
   Edit Mode Toggle
------------------------------------- */
function toggleEditMode(enable) {
  isEditing = enable;
  renderConfigForm();

  editBtn.classList.toggle("hidden", enable);
  addConfigBtn.classList.toggle("hidden", !enable);
  formControls.classList.toggle("hidden", !enable);

  if (!enable) {
    fetchConfig(); // Re-fetch to ensure we have the latest data
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
      detect_no_safety_vest_or_helmet: true,
      detect_near_machinery_or_vehicle: true,
      detect_in_restricted_area: true
    }
  });

  renderConfigForm();
  toggleEditMode(true);
});

// Automatically fetch config on page load
document.addEventListener("DOMContentLoaded", fetchConfig);
