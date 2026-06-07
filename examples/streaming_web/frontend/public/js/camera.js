/* ----------------------------------
   Constants
------------------------------------- */
// A list of supported "no warnings" messages in various languages
const NO_WARNINGS_MESSAGES = [
  'No warning',             // English
  '無警告',                  // Traditional Chinese
  '无警告',                  // Simplified Chinese
  "Pas d'avertissement",    // French
  'Không có cảnh báo',       // Vietnamese
  'Tidak ada peringatan',    // Indonesian
  'ไม่มีคำเตือน'             // Thai
];
const COMPACT_WARNING_MESSAGES = {
  en: 'Warning shown on video',
  'zh-TW': '直播畫面已標示違規',
  'zh-CN': '直播画面已标示违规',
  ja: '映像に警告を表示中',
  vi: 'Canh bao hien tren video',
  id: 'Peringatan ditampilkan di video',
  fr: 'Avertissement affiche sur la video',
  th: 'แสดงคำเตือนบนวิดีโอ'
};

/* ----------------------------------
 Global Variables & Element Selectors
------------------------------------- */
let metadataSource;
let hlsPlayer;
let hlsWatchdogTimer;
let hlsPlaybackTimer;
let lastWarningsData = '';
let lastMetaUpdate = 0;
let streamPreferences = {
  overlay: 'none',
  language: 'en',
  minConfidence: 0.4
};
const CLASS_NAMES = {
  0: 'hardhat',
  1: 'mask',
  2: 'no-hardhat',
  3: 'no-mask',
  4: 'no-safety-vest',
  5: 'person',
  6: 'safety-cone',
  7: 'safety-vest',
  8: 'machinery',
  9: 'utility-pole',
  10: 'vehicle'
};
const CLASS_LABELS = {
  en: {
      hardhat: 'hardhat',
      mask: 'mask',
      'no-hardhat': 'no hardhat',
      'no-mask': 'no mask',
      'no-safety-vest': 'no safety vest',
      person: 'person',
      'safety-cone': 'safety cone',
      'safety-vest': 'safety vest',
      machinery: 'machinery',
      'utility-pole': 'utility pole',
      vehicle: 'vehicle',
      unknown: 'unknown'
  },
  'zh-TW': {
      hardhat: '安全帽',
      mask: '口罩',
      'no-hardhat': '未戴安全帽',
      'no-mask': '未戴口罩',
      'no-safety-vest': '未穿安全背心',
      person: '人員',
      'safety-cone': '交通錐',
      'safety-vest': '安全背心',
      machinery: '機具',
      'utility-pole': '電桿',
      vehicle: '車輛',
      unknown: '未知'
  },
  'zh-CN': {
      hardhat: '安全帽',
      mask: '口罩',
      'no-hardhat': '未戴安全帽',
      'no-mask': '未戴口罩',
      'no-safety-vest': '未穿安全背心',
      person: '人员',
      'safety-cone': '交通锥',
      'safety-vest': '安全背心',
      machinery: '机具',
      'utility-pole': '电杆',
      vehicle: '车辆',
      unknown: '未知'
  },
  ja: {
      hardhat: 'ヘルメット',
      mask: 'マスク',
      'no-hardhat': 'ヘルメットなし',
      'no-mask': 'マスクなし',
      'no-safety-vest': '安全ベストなし',
      person: '作業員',
      'safety-cone': 'カラーコーン',
      'safety-vest': '安全ベスト',
      machinery: '重機',
      'utility-pole': '電柱',
      vehicle: '車両',
      unknown: '不明'
  },
  vi: {
      hardhat: 'mu bao ho',
      mask: 'khau trang',
      'no-hardhat': 'khong mu bao ho',
      'no-mask': 'khong khau trang',
      'no-safety-vest': 'khong ao bao ho',
      person: 'nguoi',
      'safety-cone': 'coc an toan',
      'safety-vest': 'ao bao ho',
      machinery: 'may moc',
      'utility-pole': 'cot dien',
      vehicle: 'xe',
      unknown: 'khong ro'
  },
  id: {
      hardhat: 'helm',
      mask: 'masker',
      'no-hardhat': 'tanpa helm',
      'no-mask': 'tanpa masker',
      'no-safety-vest': 'tanpa rompi',
      person: 'orang',
      'safety-cone': 'kerucut',
      'safety-vest': 'rompi',
      machinery: 'mesin',
      'utility-pole': 'tiang listrik',
      vehicle: 'kendaraan',
      unknown: 'tidak dikenal'
  },
  fr: {
      hardhat: 'casque',
      mask: 'masque',
      'no-hardhat': 'sans casque',
      'no-mask': 'sans masque',
      'no-safety-vest': 'sans gilet',
      person: 'personne',
      'safety-cone': 'cone',
      'safety-vest': 'gilet',
      machinery: 'machine',
      'utility-pole': 'poteau',
      vehicle: 'vehicule',
      unknown: 'inconnu'
  },
  th: {
      hardhat: 'หมวกนิรภัย',
      mask: 'หน้ากาก',
      'no-hardhat': 'ไม่สวมหมวก',
      'no-mask': 'ไม่สวมหน้ากาก',
      'no-safety-vest': 'ไม่สวมเสื้อสะท้อนแสง',
      person: 'คนงาน',
      'safety-cone': 'กรวยจราจร',
      'safety-vest': 'เสื้อสะท้อนแสง',
      machinery: 'เครื่องจักร',
      'utility-pole': 'เสาไฟ',
      vehicle: 'ยานพาหนะ',
      unknown: 'ไม่ทราบ'
  }
};
const CLASS_COLORS = {
  hardhat: '#4caf50',
  helmet: '#4caf50',
  'safety-vest': '#4caf50',
  vest: '#4caf50',
  'no-hardhat': '#f44336',
  'no-safety-vest': '#f44336',
  person: '#ff9800',
  machinery: '#ffab40',
  vehicle: '#ffff00',
  car: '#ffff00',
  'utility-pole': '#2196f3',
  'safety-cone': '#2196f3'
};
let lastOverlayData;
let currentHlsUrl = '';
const HLS_TARGET_LATENCY_SECONDS = 2;
const HLS_MIN_LATENCY_SECONDS = 0.75;
const HLS_MAX_LATENCY_SECONDS = 4;
const HLS_STALL_SEEK_CHECKS = 3;
const HLS_STALL_CHECKS = 12;
const HLS_FORWARD_BUFFER_SECONDS = 30;
const HLS_BACK_BUFFER_SECONDS = 30;
const HLS_MAX_BUFFER_SIZE_BYTES = 60 * 1000 * 1000;
const MEDIA_AUTH_COOKIE_NAME = 'hazard_access_token';

/* ----------------------------------
 Utility Functions
------------------------------------- */
/**
* Custom logging function to avoid direct console usage
*
* @param {string} message - The message to log.
*/
function logInformation(message) {
  // For production, consider removing or sending to a logging service
  // console.log(`[INFO] ${message}`);
}

/**
* Custom error logging function to avoid direct console usage
*
* @param {string} message - The error message to log.
*/
function logError(message) {
  // For production, consider removing or sending to an error tracking service
  // console.error(`[ERROR] ${message}`);
}

/* ----------------------------------
 Initialization
------------------------------------- */
document.addEventListener('DOMContentLoaded', () => {
  syncMediaAuthCookie();
  initialisePage();
  setupUnloadHandler();
});

/**
* Initialise the page by validating parameters and setting up live playback.
*/
function initialisePage() {
  const {
      label,
      key,
      streamId,
      overlay,
      language,
      minConfidence,
      transport,
      annotatedHlsUrl
  } = getURLParameters();

  // Redirect to index page if either label or key is missing
  if (!label || !key) {
      logError('Label or key parameter is missing');
      redirectToIndex();
      return;
  }

  const domReferences = getDOMReferences();
  setCameraTitle(domReferences.cameraTitle, label, key);
  streamPreferences = { overlay, language, minConfidence };
  initialiseStreamControls({
      label,
      key,
      streamId,
      transport,
      annotatedHlsUrl,
      domReferences
  });
  initialiseHlsStream({ label, key, streamId, annotatedHlsUrl, domReferences });
}

/**
* Retrieve and validate URL parameters.
*
* @returns {Object} An object containing label and key.
*/
function getURLParameters() {
  const urlParameters = new URLSearchParams(window.location.search);
  const label = urlParameters.get('label');
  const key = urlParameters.get('key');
  const streamId = urlParameters.get('stream_id') || '';
  const overlay = urlParameters.get('overlay') || 'backend';
  const language = urlParameters.get('lang') || urlParameters.get('language') || 'en';
  const minConfidence = Number.parseFloat(urlParameters.get('min_confidence') || '0.4');
  const transport = urlParameters.get('transport') || 'hls';
  const annotatedHlsUrl = urlParameters.get('annotated_playback_url')
      || urlParameters.get('annotated_hls_url')
      || '';
  return {
      label,
      key,
      streamId,
      overlay,
      language,
      minConfidence: Number.isFinite(minConfidence) ? minConfidence : 0.4,
      transport,
      annotatedHlsUrl
  };
}

/**
* Retrieve necessary DOM elements for dynamic updates.
*
* @returns {Object} An object containing DOM references.
*/
function getDOMReferences() {
  return {
      cameraTitle: document.getElementById('camera-title'),
      mediaStage: document.getElementById('media-stage'),
      streamImage: document.getElementById('stream-image'),
      streamVideo: document.getElementById('stream-video'),
      overlayCanvas: document.getElementById('overlay-canvas'),
      overlayToggle: document.getElementById('overlay-toggle'),
      languageSelect: document.getElementById('language-select'),
      confidenceInput: document.getElementById('confidence-input'),
      confidenceOutput: document.getElementById('confidence-output'),
      loadingIndicator: document.getElementById('loading-indicator'),
      streamMeta: document.getElementById('stream-meta'),
      warningsList: document.getElementById('warnings-ul')
  };
}

function initialiseStreamControls({
  label,
  key,
  streamId,
  transport,
  annotatedHlsUrl,
  domReferences
}) {
  domReferences.overlayToggle.checked = streamPreferences.overlay !== 'none';
  domReferences.languageSelect.value = normaliseLanguage(streamPreferences.language);
  domReferences.confidenceInput.value = String(streamPreferences.minConfidence);
  updateConfidenceOutput(domReferences);

  domReferences.overlayToggle.addEventListener('change', () => {
      streamPreferences.overlay = domReferences.overlayToggle.checked
          ? 'backend'
          : 'none';
      persistViewerPreferences({
          label, key, streamId, transport, annotatedHlsUrl,
      });
      if (transport === 'hls' && annotatedHlsUrl) {
          switchHlsSource(domReferences);
      }
      renderOverlay(domReferences);
  });
  domReferences.languageSelect.addEventListener('change', () => {
      streamPreferences.language = normaliseLanguage(
          domReferences.languageSelect.value
      );
      persistViewerPreferences({
          label, key, streamId, transport, annotatedHlsUrl,
      });
      renderOverlay(domReferences);
  });
  domReferences.confidenceInput.addEventListener('input', () => {
      const parsed = Number.parseFloat(domReferences.confidenceInput.value);
      streamPreferences.minConfidence = Number.isFinite(parsed) ? parsed : 0.4;
      updateConfidenceOutput(domReferences);
      persistViewerPreferences({
          label, key, streamId, transport, annotatedHlsUrl,
      });
      renderOverlay(domReferences);
  });
  window.addEventListener('resize', () => renderOverlay(domReferences));
}

function updateConfidenceOutput(domReferences) {
  domReferences.confidenceOutput.textContent = Number(
      streamPreferences.minConfidence
  ).toFixed(2);
}

function persistViewerPreferences({
  label,
  key,
  streamId,
  transport,
  annotatedHlsUrl
}) {
  const params = new URLSearchParams(window.location.search);
  params.set('label', label);
  params.set('key', key);
  if (streamId) params.set('stream_id', streamId);
  if (transport) params.set('transport', transport);
  params.set('overlay', streamPreferences.overlay);
  params.set('lang', normaliseLanguage(streamPreferences.language));
  params.set('min_confidence', String(streamPreferences.minConfidence));
  if (annotatedHlsUrl) {
      params.set('annotated_playback_url', annotatedHlsUrl);
      params.set('annotated_hls_url', annotatedHlsUrl);
  }
  window.history.replaceState(null, '', `${window.location.pathname}?${params}`);
}

/**
* Set the camera title to include the label and key.
*
* @param {HTMLElement} cameraTitle - The DOM element for camera title.
* @param {string} label - The label parameter.
* @param {string} key - The key parameter.
*/
function setCameraTitle(cameraTitle, label, key) {
  cameraTitle.textContent = `${label} - ${key}`;
}

/**
* Ensure live playback resources are closed when the page is unloaded.
*/
function setupUnloadHandler() {
  window.addEventListener('beforeunload', () => {
      closePlayback();
      closeMetadataSource();
      closeHlsPlayer();
  });
}

/**
* Redirect the user to the index page.
*/
function redirectToIndex() {
  window.location.href = 'index.html';
}

function initialiseHlsStream({
  label,
  key,
  streamId,
  annotatedHlsUrl,
  domReferences
}) {
  const urlParameters = new URLSearchParams(window.location.search);
  const hlsUrl = urlParameters.get('playback_url') || urlParameters.get('hls_url');
  if (!hlsUrl) {
      handleServerError('Missing live playback URL');
      return;
  }
  currentHlsUrl = getActiveHlsUrl(hlsUrl, annotatedHlsUrl);

  domReferences.streamImage.style.display = 'none';
  domReferences.streamVideo.style.display = 'none';
  closeHlsPlayer();
  const stopPlayback = () => {
      closeHlsPlayer();
      closeMetadataSource();
      domReferences.streamVideo.onerror = null;
      domReferences.streamVideo.onloadeddata = null;
      domReferences.streamVideo.onplaying = null;
      domReferences.streamVideo.ontimeupdate = null;
      domReferences.streamVideo.removeAttribute('src');
      domReferences.streamVideo.load();
      domReferences.streamVideo.style.display = 'none';
      handleServerError('HLS playback failed');
  };
  hlsWatchdogTimer = window.setTimeout(stopPlayback, 8000);
  let lastCurrentTime = 0;
  let stalledChecks = 0;
  let hlsReady = false;
  hlsPlaybackTimer = window.setInterval(() => {
      const currentTime = domReferences.streamVideo.currentTime || 0;
      if (currentTime > lastCurrentTime + 0.05) {
          lastCurrentTime = currentTime;
          stalledChecks = 0;
          syncVideoToLiveLatency(domReferences.streamVideo, hlsPlayer);
          return;
      }
      stalledChecks += 1;
      if (stalledChecks >= HLS_STALL_SEEK_CHECKS) {
          syncVideoToLiveLatency(domReferences.streamVideo, hlsPlayer, true);
      }
      if (stalledChecks >= HLS_STALL_CHECKS) stopPlayback();
  }, 2000);
  const markVideoReady = () => {
      if (hlsReady) return;
      if (!domReferences.streamVideo.videoWidth) return;
      hlsReady = true;
      clearHlsWatchdogTimer();
      domReferences.loadingIndicator.style.display = 'none';
      domReferences.streamVideo.style.display = 'block';
      updateStreamMeta(domReferences.streamMeta);
      initialiseMetadataStream({ label, key, streamId, domReferences });
      renderOverlay(domReferences);
  };
  domReferences.streamVideo.onloadeddata = markVideoReady;
  domReferences.streamVideo.onplaying = markVideoReady;
  domReferences.streamVideo.ontimeupdate = markVideoReady;
  domReferences.streamVideo.onerror = () => {
      stopPlayback();
  };
  attachHls(
      domReferences.streamVideo,
      currentHlsUrl,
      stopPlayback
  );
}

function getActiveHlsUrl(cleanHlsUrl, annotatedHlsUrl) {
  if (streamPreferences.overlay !== 'none' && annotatedHlsUrl) {
      return annotatedHlsUrl;
  }
  return cleanHlsUrl;
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

function switchHlsSource(domReferences) {
  const params = new URLSearchParams(window.location.search);
  const cleanHlsUrl = params.get('playback_url') || params.get('hls_url') || '';
  const annotatedHlsUrl = params.get('annotated_playback_url')
      || params.get('annotated_hls_url')
      || '';
  const nextHlsUrl = getActiveHlsUrl(cleanHlsUrl, annotatedHlsUrl);
  if (!nextHlsUrl || nextHlsUrl === currentHlsUrl) return;
  currentHlsUrl = nextHlsUrl;
  closeHlsPlayer();
  domReferences.streamVideo.removeAttribute('src');
  domReferences.streamVideo.load();
  attachHls(domReferences.streamVideo, currentHlsUrl, () => {});
  clearOverlay(domReferences.overlayCanvas, domReferences.overlayCanvas.getContext('2d'));
}

function attachHls(video, hlsUrl, onError) {
  if (video.canPlayType('application/vnd.apple.mpegurl')) {
      video.src = hlsUrl;
      video.play().catch(() => {});
      return;
  }
  if (!window.Hls || !window.Hls.isSupported()) {
      onError();
      return;
  }

  hlsPlayer = new window.Hls({
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
  hlsPlayer.loadSource(hlsUrl);
  hlsPlayer.attachMedia(video);
  hlsPlayer.on(window.Hls.Events.ERROR, (_event, data) => {
      if (!data.fatal) return;
      closeHlsPlayer();
      onError();
  });
  hlsPlayer.on(window.Hls.Events.MANIFEST_PARSED, () => {
      syncVideoToLiveLatency(video, hlsPlayer, true);
      video.play().catch(() => {});
  });
  hlsPlayer.on(window.Hls.Events.FRAG_LOADED, () => {
      syncVideoToLiveLatency(video, hlsPlayer);
  });
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

function closeHlsPlayer() {
  clearHlsWatchdogTimer();
  clearHlsPlaybackTimer();
  if (hlsPlayer) {
      hlsPlayer.destroy();
      hlsPlayer = null;
  }
}

function clearHlsWatchdogTimer() {
  if (hlsWatchdogTimer) {
      window.clearTimeout(hlsWatchdogTimer);
      hlsWatchdogTimer = undefined;
  }
}

function clearHlsPlaybackTimer() {
  if (hlsPlaybackTimer) {
      window.clearInterval(hlsPlaybackTimer);
      hlsPlaybackTimer = undefined;
  }
}

function initialiseMetadataStream({ label, key, streamId, domReferences }) {
  closeMetadataSource();
  const urlParameters = new URLSearchParams(window.location.search);
  const token = urlParameters.get('token');
  const path = streamId
      ? `/api/metadata/stream-id/${encodeURIComponent(label)}/${encodeURIComponent(streamId)}`
      : `/api/metadata/stream/${encodeURIComponent(label)}/${encodeURIComponent(key)}`;
  const query = new URLSearchParams();
  if (token) query.set('token', token);
  metadataSource = new EventSource(`${path}?${query.toString()}`);
  metadataSource.addEventListener('metadata', (event) => {
      const data = JSON.parse(event.data);
      processMetadataData(data, domReferences);
      updateStreamMeta(domReferences.streamMeta);
  });
}

function closeMetadataSource() {
  if (metadataSource) {
      metadataSource.close();
      metadataSource = null;
  }
}

/**
* Process compact metadata from the live metadata stream.
*
* @param {Object} data - The metadata received from the server.
* @param {Object} domReferences - The DOM elements to update.
*/
function processMetadataData(data, domReferences) {
  if (data.error) {
      handleServerError(data.error);
      return;
  }

  lastOverlayData = data;
  renderOverlay(domReferences);

  updateWarningState(domReferences.warningsList, data);
}

function updateWarningState(warningsList, data) {
  updateWarningsList(
      warningsList,
      compactWarningMessage(Boolean(data.has_warning))
  );
}

function compactWarningMessage(hasWarning) {
  if (hasWarning) {
      const language = normaliseLanguage(streamPreferences.language);
      return COMPACT_WARNING_MESSAGES[language]
          || COMPACT_WARNING_MESSAGES.en;
  }
  return NO_WARNINGS_MESSAGES[0];
}

/**
* Handle error messages from the server.
*
* @param {string} errorMessage - The error message from the server.
*/
function handleServerError(errorMessage) {
  logError(errorMessage);
  redirectToIndex();
}

function normaliseLanguage(value) {
  const key = String(value || 'en').replace('_', '-');
  const aliases = {
      zh: 'zh-TW',
      'zh-hant': 'zh-TW',
      'zh-tw': 'zh-TW',
      'zh-hk': 'zh-TW',
      'zh-hans': 'zh-CN',
      'zh-cn': 'zh-CN',
      jp: 'ja',
      'ja-jp': 'ja',
      'vi-vn': 'vi',
      'id-id': 'id',
      'fr-fr': 'fr',
      'th-th': 'th'
  };
  if (CLASS_LABELS[key]) return key;
  return aliases[key.toLowerCase()] || 'en';
}

function renderOverlay(domReferences) {
  const canvas = domReferences.overlayCanvas;
  const context = canvas.getContext('2d');
  clearOverlay(canvas, context);
}

function getVisibleMedia(domReferences) {
  if (domReferences.streamVideo.style.display !== 'none'
      && !domReferences.streamVideo.classList.contains('hidden')) {
      return domReferences.streamVideo;
  }
  if (domReferences.streamImage.style.display !== 'none'
      && !domReferences.streamImage.classList.contains('hidden')) {
      return domReferences.streamImage;
  }
  return null;
}

function clearOverlay(canvas, context) {
  context.clearRect(0, 0, canvas.width, canvas.height);
}

function resizeOverlayCanvas(canvas, mediaRect) {
  const width = Math.max(1, Math.round(mediaRect.width));
  const height = Math.max(1, Math.round(mediaRect.height));
  if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
  }
}

function parseJsonList(value) {
  if (Array.isArray(value)) return value;
  if (!value) return [];
  try {
      const parsed = JSON.parse(value);
      return Array.isArray(parsed) ? parsed : [];
  } catch (_error) {
      return [];
  }
}

function parseWarnings(value) {
  if (!value) return {};
  if (typeof value === 'object') return value;
  try {
      const parsed = JSON.parse(value);
      return parsed && typeof parsed === 'object' ? parsed : {};
  } catch (_error) {
      return {};
  }
}

function parseDetections(value, sourceWidth, sourceHeight, warnings) {
  const warningClasses = getWarningClasses(warnings);
  return parseJsonList(value)
      .map((item) => parseDetection(
          item,
          sourceWidth,
          sourceHeight,
          warningClasses
      ))
      .filter(Boolean);
}

function parseDetection(
  item,
  sourceWidth,
  sourceHeight,
  warningClasses
) {
  if (Array.isArray(item) && item.length >= 6) {
      let [x1, y1, x2, y2] = item.slice(0, 4).map(Number);
      const confidence = Number(item[4]) || 0;
      const className = CLASS_NAMES[Number(item[5])] || `class-${item[5]}`;
      if (looksNormalised([x1, y1, x2, y2])) {
          x1 *= sourceWidth;
          x2 *= sourceWidth;
          y1 *= sourceHeight;
          y2 *= sourceHeight;
      }
      return buildDetection(
          className,
          confidence,
          x1,
          y1,
          x2,
          y2,
          warningClasses
      );
  }
  if (!item || typeof item !== 'object') return null;
  const bbox = item.bbox || {};
  const className = String(
      item.class_name || item.class || item.label || 'unknown'
  ).toLowerCase();
  const confidence = Number(item.confidence || item.conf || 0);
  let x = Number(bbox.x || 0);
  let y = Number(bbox.y || 0);
  let width = Number(bbox.width || bbox.w || 0);
  let height = Number(bbox.height || bbox.h || 0);
  if (looksNormalised([x, y, width, height])) {
      x *= sourceWidth;
      width *= sourceWidth;
      y *= sourceHeight;
      height *= sourceHeight;
  }
  return buildDetection(
      className,
      confidence,
      x,
      y,
      x + width,
      y + height,
      warningClasses
  );
}

function buildDetection(
  className,
  confidence,
  x1,
  y1,
  x2,
  y2,
  warningClasses
) {
  const left = Math.max(0, Math.min(x1, x2));
  const top = Math.max(0, Math.min(y1, y2));
  const right = Math.max(left, Math.max(x1, x2));
  const bottom = Math.max(top, Math.max(y1, y2));
  if (right <= left || bottom <= top) return null;
  const bbox = { left, top, right, bottom };
  return {
      className,
      confidence,
      isWarning: warningClasses.has(className),
      bbox
  };
}

function looksNormalised(values) {
  return values.every((value) => Number.isFinite(value) && value >= 0 && value <= 1);
}

function getWarningClasses(warnings) {
  const data = parseWarnings(warnings);
  const classes = new Set();
  if (data.warning_no_hardhat) classes.add('no-hardhat');
  if (data.warning_no_safety_vest) classes.add('no-safety-vest');
  if (data.warning_people_in_controlled_area) classes.add('person');
  if (data.warning_people_in_utility_pole_controlled_area) classes.add('person');
  if (data.detect_machinery_close_to_pole) {
      classes.add('machinery');
      classes.add('vehicle');
  }
  return classes;
}

function drawPolygons(
  context,
  polygons,
  sourceWidth,
  sourceHeight,
  transformPoint,
  fillStyle,
  strokeStyle
) {
  polygons.forEach((polygon) => {
      const points = normalisePolygon(polygon, sourceWidth, sourceHeight);
      if (points.length < 3) return;
      context.beginPath();
      points.forEach(([x, y], index) => {
          const point = transformPoint(x, y);
          if (index === 0) context.moveTo(point.x, point.y);
          else context.lineTo(point.x, point.y);
      });
      context.closePath();
      context.fillStyle = fillStyle;
      context.strokeStyle = strokeStyle;
      context.lineWidth = 2;
      context.fill();
      context.stroke();
  });
}

function normalisePolygon(polygon, sourceWidth, sourceHeight) {
  if (!Array.isArray(polygon)) return [];
  const points = polygon.map((point) => {
      if (Array.isArray(point)) return [Number(point[0]), Number(point[1])];
      if (point && typeof point === 'object') {
          return [Number(point.x), Number(point.y)];
      }
      return [NaN, NaN];
  }).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  const flat = points.flat();
  if (looksNormalised(flat)) {
      return points.map(([x, y]) => [x * sourceWidth, y * sourceHeight]);
  }
  return points;
}

function drawDetection(context, detection, transformPoint, scale) {
  const { left, top, right, bottom } = detection.bbox;
  const p1 = transformPoint(left, top);
  const p2 = transformPoint(right, bottom);
  const color = detection.isWarning
      ? '#f44336'
      : CLASS_COLORS[detection.className] || '#29b6f6';
  const lineWidth = Math.max(2, Math.round(scale * 3));
  context.strokeStyle = color;
  context.lineWidth = lineWidth;
  context.strokeRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
  drawDetectionLabel(context, detection, p1.x, p1.y, color);
}

function drawDetectionLabel(context, detection, x, y, color) {
  const language = normaliseLanguage(streamPreferences.language);
  const labels = CLASS_LABELS[language] || CLASS_LABELS.en;
  const label = labels[detection.className]
      || CLASS_LABELS.en[detection.className]
      || detection.className;
  context.font = '600 14px Arial, sans-serif';
  const paddingX = 6;
  const paddingY = 4;
  const textWidth = context.measureText(label).width;
  const labelWidth = textWidth + paddingX * 2;
  const labelHeight = 22;
  const labelY = y - labelHeight < 0 ? y : y - labelHeight;
  context.fillStyle = color;
  context.globalAlpha = 0.82;
  context.fillRect(x, labelY, labelWidth, labelHeight);
  context.globalAlpha = 1;
  context.fillStyle = '#fff';
  context.fillText(label, x + paddingX, labelY + labelHeight - paddingY - 2);
}

/* ----------------------------------
 DOM Updates
------------------------------------- */
function updateStreamMeta(streamMeta) {
  const now = Date.now();
  if (now - lastMetaUpdate < 1000) {
      return;
  }
  streamMeta.textContent = `Last updated: ${new Date().toLocaleString()}`;
  lastMetaUpdate = now;
}

/**
* Update the warnings list.
*
* @param {HTMLElement} warningsList - The unordered list element to update.
* @param {string} warningsData - The warnings data as a newline-separated string.
*/
function updateWarningsList(warningsList, warningsData) {
  if (warningsData === lastWarningsData) {
      return;
  }
  lastWarningsData = warningsData;
  const warnings = warningsData.split('\n');
  warningsList.innerHTML = '';

  // Check if there are no warnings
  if (warnings.length === 1 && NO_WARNINGS_MESSAGES.includes(warnings[0])) {
      warningsList.className = 'no-warnings';
      appendWarningItem(warningsList, warnings[0], ['no-warning']);
  } else {
      warningsList.className = 'warnings';
      warnings.forEach((warning) => appendWarningItem(warningsList, warning));
  }
}

/**
* Append a single warning to the list.
*
* @param {HTMLElement} warningsList - The unordered list element.
* @param {string} warningText - The warning text to append.
* @param {Array} additionalClasses - Additional CSS classes to add to the warning item.
*/
function appendWarningItem(warningsList, warningText, additionalClasses = []) {
  const paragraph = document.createElement('p');
  paragraph.textContent = warningText;
  paragraph.classList.add('warning-item', ...additionalClasses);
  warningsList.appendChild(paragraph);
}
