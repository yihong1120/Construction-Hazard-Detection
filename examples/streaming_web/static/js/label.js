$(document).ready(() => {
    initializeWebSocket();
});

/**
 * Initialize the WebSocket connection and set up event handlers.
 */
function initializeWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
    const currentPageLabel = getCurrentPageLabel();

    const socket = new WebSocket(`${protocol}${window.location.host}/ws/label/${currentPageLabel}`);

    socket.onopen = () => {
        console.log('WebSocket connected!');
    };

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleUpdate(data, currentPageLabel);
    };

    socket.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    socket.onclose = () => {
        console.log('WebSocket closed');
    };
}

/**
 * Get the label of the current page.
 * @returns {string} The label of the current page.
 */
function getCurrentPageLabel() {
    return $('h1').text();
}

/**
 * Handle WebSocket updates for multiple images.
 * @param {Object} data - The received data
 * @param {string} currentPageLabel - The label of the current page
 */
function handleUpdate(data, currentPageLabel) {
    if (data.label === currentPageLabel) {
        console.log('Received update for current label:', data.label);
        updateCameraGrid(data.images); // 更新為處理多鏡頭影像
    } else {
        console.log('Received update for different label:', data.label);
    }
}

/**
 * Update the camera grid with new images for multiple cameras.
 * @param {Array} images - The array of images with key and base64 data.
 */
function updateCameraGrid(images) {
    images.forEach((cameraData) => {
        // 檢查是否已經存在相同的 image_name
        const existingCameraDiv = $(`.camera h2:contains(${cameraData.key.split('_').pop()})`).closest('.camera');

        if (existingCameraDiv.length > 0) {
            // 更新現有的圖像
            existingCameraDiv.find('img').attr('src', `data:image/png;base64,${cameraData.image}`);
        } else {
            // 如果沒有相同的 image_name，則創建新的圖區
            const cameraDiv = createCameraDiv(cameraData);
            $('.camera-grid').append(cameraDiv);
        }
    });
}

/**
 * Create a camera div element.
 * @param {Object} cameraData - The data for creating the camera div
 * @returns {HTMLElement} - The div element containing the image and title
 */
function createCameraDiv(cameraData) {
    const cameraDiv = $('<div>').addClass('camera');
    // 只保留 _ 之後的部分作為標題
    const titleText = cameraData.key.split('_').pop();
    const title = $('<h2>').text(titleText);
    const img = $('<img>').attr('src', `data:image/png;base64,${cameraData.image}`).attr('alt', `${cameraData.key} image`);
    cameraDiv.append(title).append(img);
    return cameraDiv[0];
}
