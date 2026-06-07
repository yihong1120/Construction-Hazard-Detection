const HLS_TARGET_LATENCY_SECONDS = 2;
const HLS_MIN_LATENCY_SECONDS = 0.75;
const HLS_MAX_LATENCY_SECONDS = 4;
const HLS_STALL_SEEK_CHECKS = 3;
const HLS_FORWARD_BUFFER_SECONDS = 30;
const HLS_BACK_BUFFER_SECONDS = 30;
const HLS_MAX_BUFFER_SIZE_BYTES = 60 * 1000 * 1000;
const MEDIA_AUTH_COOKIE_NAME = 'hazard_access_token';

// Execute when the document's DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    syncMediaAuthCookie();
    const labelTitle = document.getElementById('label-title');
    const loadingMessage = document.getElementById('loading-message');
    const urlParams = new URLSearchParams(window.location.search);
    const label = urlParams.get('label');

    if (!validateLabel(label)) return;

    labelTitle.textContent = label;
    loadingMessage.style.display = 'block'; // Show the loading message
    initializeHlsOverview(label).catch((error) => {
        logError(`Failed to load streams: ${error}`);
        loadingMessage.textContent = 'No live streams available';
    });

    window.addEventListener('beforeunload', closePlayback);
});

/**
 * Validate the label parameter and redirect if invalid.
 *
 * @param {string|null} label - The label parameter from the URL.
 * @returns {boolean} Whether the label is valid.
 */
function validateLabel(label) {
    if (!label) {
        logError('Label parameter is missing in the URL');
        window.location.href = 'index.html'; // Redirect to index.html
        return false;
    }
    return true;
}

async function initializeHlsOverview(label) {
    const loadingMessage = document.getElementById('loading-message');
    const urlParams = new URLSearchParams(window.location.search);
    const token = getMediaAuthToken();
    const fetchOptions = token
        ? { headers: { Authorization: `Bearer ${token}` } }
        : {};
    const response = await fetch(
        `/api/streams/${encodeURIComponent(label)}`,
        fetchOptions
    );
    if (!response.ok) {
        throw new Error('Failed to fetch streams');
    }

    const data = await response.json();
    const streams = Array.isArray(data.streams) ? data.streams : [];
    if (streams.length === 0) {
        throw new Error('No streams available');
    }

    streams.forEach((stream) => {
        const key = stream.key;
        const streamId = stream.stream_id || '';
        const hlsUrl = stream.playback_url || stream.hls_url;
        const annotatedHlsUrl = stream.annotated_playback_url
            || stream.annotated_hls_url
            || '';
        if (hlsUrl) {
            upsertHlsCameraFrame(key, streamId, hlsUrl, annotatedHlsUrl);
        }
    });
    loadingMessage.style.display = 'none';
}

function upsertHlsCameraFrame(key, streamId, hlsUrl, annotatedHlsUrl = '') {
    const cameraGrid = document.getElementById('camera-grid');
    let cameraDiv = cameraGrid.querySelector(`[data-key="${CSS.escape(key)}"]`);
    if (!cameraDiv) {
        cameraDiv = createCameraDiv(key, streamId);
        cameraGrid.appendChild(cameraDiv);
    }
    if (streamId) cameraDiv.dataset.streamId = streamId;
    cameraDiv.dataset.hlsUrl = hlsUrl;
    if (annotatedHlsUrl) cameraDiv.dataset.annotatedHlsUrl = annotatedHlsUrl;

    const media = cameraDiv.querySelector('img, video');
    const video = media?.tagName === 'VIDEO'
        ? media
        : document.createElement('video');
    if (media && media !== video) media.replaceWith(video);
    video.autoplay = true;
    video.muted = true;
    video.playsInline = true;
    video.controls = false;
    attachHls(video, hlsUrl);
}

function attachHls(video, hlsUrl) {
    const stopPlayback = () => {
        if (video._hlsPlaybackTimer) {
            window.clearInterval(video._hlsPlaybackTimer);
            video._hlsPlaybackTimer = null;
        }
        if (video._hlsPlayer) {
            video._hlsPlayer.destroy();
            video._hlsPlayer = null;
        }
        logError('HLS playback failed');
    };
    if (video.canPlayType('application/vnd.apple.mpegurl')) {
        video.src = hlsUrl;
        video.play().catch(() => {});
        startHlsPlaybackWatchdog(video, stopPlayback);
        return;
    }
    if (!window.Hls || !window.Hls.isSupported()) {
        stopPlayback();
        return;
    }

    const previousPlayer = video._hlsPlayer;
    if (previousPlayer) previousPlayer.destroy();
    if (video._hlsPlaybackTimer) {
        window.clearInterval(video._hlsPlaybackTimer);
        video._hlsPlaybackTimer = null;
    }
    const player = new window.Hls({
        lowLatencyMode: true,
        liveSyncDuration: HLS_TARGET_LATENCY_SECONDS,
        liveMaxLatencyDuration: HLS_MAX_LATENCY_SECONDS,
        maxBufferLength: HLS_FORWARD_BUFFER_SECONDS,
        maxMaxBufferLength: HLS_FORWARD_BUFFER_SECONDS,
        maxBufferSize: HLS_MAX_BUFFER_SIZE_BYTES,
        backBufferLength: HLS_BACK_BUFFER_SECONDS,
        liveBackBufferLength: HLS_BACK_BUFFER_SECONDS,
        xhrSetup: (xhr) => {
            const token = getMediaAuthToken();
            if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`);
            xhr.withCredentials = true;
        }
    });
    video._hlsPlayer = player;
    player.loadSource(hlsUrl);
    player.attachMedia(video);
    player.on(window.Hls.Events.ERROR, (_event, data) => {
        if (!data.fatal) return;
        player.destroy();
        video._hlsPlayer = null;
        stopPlayback();
    });
    player.on(window.Hls.Events.MANIFEST_PARSED, () => {
        syncVideoToLiveLatency(video, player, true);
        video.play().catch(() => {});
    });
    player.on(window.Hls.Events.FRAG_LOADED, () => {
        syncVideoToLiveLatency(video, player);
    });
    startHlsPlaybackWatchdog(video, stopPlayback);
}

function getMediaAuthToken() {
    const queryToken = new URLSearchParams(window.location.search).get('token');
    if (queryToken) return queryToken;

    const tokenKeys = ['access_token', 'token', 'jwt', 'authToken'];
    try {
        for (const storage of [window.localStorage, window.sessionStorage]) {
            if (!storage) continue;
            for (const key of tokenKeys) {
                const token = storage.getItem(key);
                if (token) return token;
            }
        }
    } catch (_error) {}
    return '';
}

function syncMediaAuthCookie() {
    const token = getMediaAuthToken();
    if (!token) return;
    const cookieSecurity = window.location.protocol === 'https:'
        ? '; Secure; SameSite=None'
        : '; SameSite=Lax';
    document.cookie = (
        `${MEDIA_AUTH_COOKIE_NAME}=${encodeURIComponent(token)}; `
        + `path=/${cookieSecurity}`
    );
}

function startHlsPlaybackWatchdog(video, onStalled) {
    let lastCurrentTime = 0;
    let stalledChecks = 0;
    video._hlsPlaybackTimer = window.setInterval(() => {
        const currentTime = video.currentTime || 0;
        if (currentTime > lastCurrentTime + 0.05) {
            lastCurrentTime = currentTime;
            stalledChecks = 0;
            syncVideoToLiveLatency(video, video._hlsPlayer);
            return;
        }
        stalledChecks += 1;
        if (stalledChecks >= HLS_STALL_SEEK_CHECKS) {
            syncVideoToLiveLatency(video, video._hlsPlayer, true);
        }
        if (stalledChecks >= HLS_STALL_CHECKS) onStalled();
    }, 2000);
}

function syncVideoToLiveLatency(video, player, force = false) {
    const liveEdge = getLiveEdge(video, player);
    if (!Number.isFinite(liveEdge) || liveEdge <= 0) return;
    const currentTime = video.currentTime || 0;
    const latency = liveEdge - currentTime;
    const targetTime = Math.max(0, liveEdge - HLS_TARGET_LATENCY_SECONDS);
    const isTooFarBehind = latency > HLS_MAX_LATENCY_SECONDS;
    const isTooCloseToEdge = latency < HLS_MIN_LATENCY_SECONDS;
    if (force || isTooFarBehind || isTooCloseToEdge || currentTime > liveEdge) {
        if (Math.abs(currentTime - targetTime) > 0.25) {
            video.currentTime = targetTime;
        }
        video.play().catch(() => {});
    }
}

function getLiveEdge(video, player) {
    if (player && Number.isFinite(player.liveSyncPosition)) {
        return player.liveSyncPosition + HLS_TARGET_LATENCY_SECONDS;
    }
    const ranges = video.seekable;
    if (!ranges || ranges.length === 0) return Number.NaN;
    return ranges.end(ranges.length - 1);
}

/**
 * Close live playback resources gracefully.
 */
function closePlayback() {
    document.querySelectorAll('video').forEach((video) => {
        if (video._hlsPlayer) video._hlsPlayer.destroy();
        if (video._hlsPlaybackTimer) {
            window.clearInterval(video._hlsPlaybackTimer);
            video._hlsPlaybackTimer = null;
        }
    });
}

/**
 * Create a camera div for the given key and image.
 *
 * @param {string} key - The unique key for the camera.
 * @returns {HTMLElement} The camera div element.
 */
function createCameraDiv(key, streamId) {
    const cameraDiv = createElementWithClass('div', 'camera');
    cameraDiv.dataset.key = key;
    if (streamId) cameraDiv.dataset.streamId = streamId;

    const title = createElementWithText('h2', key);
    const img = setupCameraImage(key);

    cameraDiv.appendChild(title);
    cameraDiv.appendChild(img);

    cameraDiv.addEventListener('click', () => {
        redirectToCameraPage(key, cameraDiv.dataset.streamId || '');
    });
    return cameraDiv;
}

/**
 * Create an element with a specific class name.
 *
 * @param {string} tagName - The HTML tag name.
 * @param {string} className - The class name to assign.
 * @returns {HTMLElement} The created element.
 */
function createElementWithClass(tagName, className) {
    const element = document.createElement(tagName);
    element.className = className;
    return element;
}

/**
 * Create an element with text content.
 *
 * @param {string} tagName - The HTML tag name.
 * @param {string} text - The text content to assign.
 * @returns {HTMLElement} The created element.
 */
function createElementWithText(tagName, text) {
    const element = document.createElement(tagName);
    element.textContent = text;
    return element;
}

/**
 * Set up the camera image element.
 *
 * @param {string} key - The unique key for the camera.
 * @returns {HTMLImageElement} The image element.
 */
function setupCameraImage(key) {
    const img = document.createElement('img');
    img.alt = `${key} image`;
    return img;
}

/**
 * Redirect the user to the camera page for the specified key.
 *
 * @param {string} key - The unique key for the camera.
 */
function redirectToCameraPage(key, streamId) {
    const urlParams = new URLSearchParams(window.location.search);
    const label = urlParams.get('label');
    const nextParams = new URLSearchParams({
        label,
        key
    });
    ['overlay', 'lang', 'language', 'min_confidence', 'token'].forEach((name) => {
        const value = urlParams.get(name);
        if (value) nextParams.set(name, value);
    });
    if (streamId) nextParams.set('stream_id', streamId);
    const cameraDiv = document
        .getElementById('camera-grid')
        .querySelector(`[data-key="${CSS.escape(key)}"]`);
    if (cameraDiv?.dataset.hlsUrl) {
        nextParams.set('transport', 'hls');
        nextParams.set('playback_url', cameraDiv.dataset.hlsUrl);
        nextParams.set('hls_url', cameraDiv.dataset.hlsUrl);
    }
    if (cameraDiv?.dataset.annotatedHlsUrl) {
        nextParams.set('annotated_playback_url', cameraDiv.dataset.annotatedHlsUrl);
        nextParams.set('annotated_hls_url', cameraDiv.dataset.annotatedHlsUrl);
    }
    window.location.href = `/camera.html?${nextParams.toString()}`;
}

/**
 * Custom logging functions to replace console statements.
 */

function logInfo(message) {
    // Uncomment this for development or logging services
    // console.log(`[INFO] ${message}`);
}

function logError(message) {
    // Uncomment this for development or logging services
    // console.error(`[ERROR] ${message}`);
}
