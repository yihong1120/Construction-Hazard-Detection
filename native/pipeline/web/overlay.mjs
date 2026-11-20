// Canvas overlay for a clean <video> and the existing authorised metadata API.
// Call setSnapshot(event.overlay) from SSE/WS. Never adds a media connection.
const CLASS_CODES = ['helmet', 'mask', 'no_helmet', 'no_mask', 'no_safety_vest',
  'person', 'safety_cone', 'safety_vest', 'machinery', 'utility_pole', 'vehicle'];
const CLASS_COLOURS = ['#4CAF50', '#26A69A', '#F44336', '#EF5350', '#E53935',
  '#FF9800', '#FF5722', '#8BC34A', '#FFC107', '#03A9F4', '#FFEB3B'];
const WARNING_COLOUR = '#F44336';
const WARNING_CLASSES = {
  no_helmet: ['warning_no_hardhat'],
  no_safety_vest: ['warning_no_safety_vest'],
  person: ['warning_people_in_controlled_area', 'warning_people_in_utility_pole_controlled_area'],
  machinery: ['detect_machinery_close_to_pole'],
  vehicle: ['detect_machinery_close_to_pole'],
};
const INFERRED_WARNINGS = {
  no_helmet: 'warning_no_hardhat',
  no_mask: 'warning_no_mask',
  no_safety_vest: 'warning_no_safety_vest',
};

export function contain(width, height, sourceWidth, sourceHeight) {
  const scale = Math.min(width / sourceWidth, height / sourceHeight);
  return { scale, x: (width - sourceWidth * scale) / 2,
    y: (height - sourceHeight * scale) / 2 };
}

export class OverlayState {
  constructor(maxAgeMs = 1000) { this.maxAgeMs = maxAgeMs; this.snapshot = null; }
  update(snapshot, now) {
    if (!snapshot || typeof snapshot.owner !== 'string' ||
        !Number.isSafeInteger(snapshot.sequence) || !Number.isSafeInteger(snapshot.generation) ||
        !(Number.isFinite(snapshot.frame_width) && Number.isFinite(snapshot.frame_height) && snapshot.frame_width > 0 && snapshot.frame_width <= 16384 && snapshot.frame_height > 0 && snapshot.frame_height <= 16384) ||
        !Array.isArray(snapshot.track_data) || snapshot.track_data.length > 4096) return false;
    const old = this.snapshot;
    if (old && old.owner === snapshot.owner &&
        (snapshot.generation < old.generation ||
         (snapshot.generation === old.generation && snapshot.sequence <= old.sequence))) return false;
    this.snapshot = snapshot; this.received = now;
    return true;
  }
  current(now, sourcePts = null) {
    const s = this.snapshot;
    if (!s || now - this.received > this.maxAgeMs) return null;
    // Optional: the player supplies its displayed PTS in the same source clock.
    // Without that mapping this is a latest-state live overlay, not frame sync.
    if (sourcePts !== null && (s.source_pts === null ||
        Math.abs(s.source_pts - sourcePts) > 0.15)) return null;
    return s;
  }
}

export async function loadDictionary(url = new URL('../assets/labels.json', import.meta.url)) {
  const response = await fetch(url);
  if (!response.ok) throw new Error('Overlay labels unavailable');
  return response.json();
}

export function playbackMetadataPath(playback, base = '/bff/streaming') {
  if (playback?.profile !== 'clean' || typeof playback.label !== 'string' ||
      !playback.label || !/^[A-Za-z0-9_-]+={0,2}$/.test(playback.stream_id || '')) {
    throw new TypeError('Browser overlay requires a clean playback descriptor');
  }
  return `${base.replace(/\/$/, '')}/metadata/stream-id/${encodeURIComponent(playback.label)}/${encodeURIComponent(playback.stream_id)}`;
}

export function subscribeOverlay(renderer, url, onMetadata) {
  const source = new EventSource(url);
  source.addEventListener('metadata', event => {
    const snapshot = JSON.parse(event.data).overlay;
    const accepted = renderer.setSnapshot(snapshot);
    onMetadata?.(event, snapshot, accepted);
  });
  return () => source.close();
}

export function attachPlaybackOverlay(video, canvas, playback, options = {}) {
  const path = playbackMetadataPath(playback, options.metadataBase);
  const renderer = createOverlay(video, canvas, options);
  const unsubscribe = subscribeOverlay(renderer, path);
  return {
    setLanguage: renderer.setLanguage,
    destroy: () => {
      unsubscribe();
      renderer.destroy();
    },
  };
}

export function createOverlay(video, canvas, { language = 'zh-TW', dictionary = {},
  maxAgeMs = 1000, sourcePts = () => null } = {}) {
  const state = new OverlayState(maxAgeMs), ctx = canvas.getContext('2d');
  let running = true, request, previous = null;
  video.style.objectFit = 'contain';
  const label = (s, id, c, code) => s.labels?.[id]?.[language] ||
    dictionary.CLASS_LABELS?.[language]?.[code] || dictionary.CLASS_LABELS?.en?.[code] ||
    s.labels?.[id]?.en || c?.display_name || code;
  const draw = now => {
    if (!running) return;
    const width = video.clientWidth, height = video.clientHeight, dpr = devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.style.width = `${width}px`; canvas.style.height = `${height}px`;
      canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
    }
    const s = video.readyState >= 2 && !video.paused ? state.current(now, sourcePts()) : null;
    const mediaWidth = video.videoWidth || s?.frame_width, mediaHeight = video.videoHeight || s?.frame_height;
    const key = [s?.owner, s?.generation, s?.sequence, language, width, height, dpr,
      mediaWidth, mediaHeight].join('|');
    if (key === previous) { request = requestAnimationFrame(draw); return; }
    previous = key;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
    if (s && video.readyState >= 2 && !video.paused) {
      const t = contain(width, height, mediaWidth, mediaHeight);
      const visibleWidth = mediaWidth * t.scale, visibleHeight = mediaHeight * t.scale;
      const scaleX = visibleWidth / s.frame_width, scaleY = visibleHeight / s.frame_height;
      const fontSize = Math.max(14, Math.min(32, Math.round(Math.min(visibleWidth, visibleHeight) / 32)));
      const classes = new Map((s.class_metadata || []).map(c => [c.id, c]));
      const warnings = s.warnings || {};
      const inferred = new Map();
      ctx.save(); ctx.translate(t.x, t.y);
      ctx.beginPath(); ctx.rect(0, 0, visibleWidth, visibleHeight); ctx.clip();
      ctx.lineWidth = 2; ctx.font = `${fontSize}px sans-serif`;
      for (const [polygons, fill, stroke] of [
        [s.cone_polys, 'rgba(255,64,129,.18)', '#e91e63'],
        [s.pole_polys, 'rgba(68,138,255,.18)', '#448aff'],
      ]) {
        for (const polygon of polygons || []) {
          if (!Array.isArray(polygon) || polygon.length < 3 ||
              !polygon.every(p => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite))) continue;
          ctx.beginPath();
          polygon.forEach(([x, y], i) => i ? ctx.lineTo(x * scaleX, y * scaleY) : ctx.moveTo(x * scaleX, y * scaleY));
          ctx.closePath(); ctx.fillStyle = fill; ctx.fill();
          ctx.strokeStyle = stroke; ctx.stroke();
        }
      }
      let labels = 0;
      function drawBox(bounds, id, classInfo, code, warning) {
        const [x1, y1, x2, y2] = bounds;
        if (!(x1 < x2 && y1 < y2)) return;
        const x = x1 * scaleX, y = y1 * scaleY;
        const colour = /^#[0-9a-f]{6}$/i.test(classInfo?.color || '') ? classInfo.color :
          warning ? WARNING_COLOUR : CLASS_COLOURS[CLASS_CODES.indexOf(code)] || '#ffffff';
        ctx.strokeStyle = colour;
        ctx.strokeRect(x, y, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
        if (labels++ >= 40) return;
        const caption = label(s, id, classInfo, code);
        const baseline = Math.max(fontSize + 2, y - 4);
        ctx.fillStyle = colour;
        ctx.fillRect(x, baseline - fontSize, (ctx.measureText(caption)?.width || 0) + 8, fontSize + 4);
        ctx.fillStyle = '#ffffff'; ctx.fillText(caption, x + 4, baseline);
      }
      for (const row of s.track_data) {
        if (!Array.isArray(row) || row.length < 7 || !row.every(Number.isFinite) || row[4] < .25) continue;
        const id = row[5], classInfo = classes.get(id);
        const code = classInfo?.code || CLASS_CODES[id] || 'unknown';
        if (code === 'safety_cone') continue;
        drawBox(row, id, classInfo, code, WARNING_CLASSES[code]?.some(key => key in warnings));
        const key = INFERRED_WARNINGS[code];
        if (key) inferred.set(key, (inferred.get(key) || 0) + 1);
      }
      const person = [...classes.values()].find(c => c.code === 'person');
      const targets = new Set();
      for (const key of ['warning_close_to_machinery', 'warning_close_to_vehicle']) {
        for (const bounds of warnings[key]?.person_bboxes || []) {
          if (!Array.isArray(bounds) || bounds.length !== 4 || !bounds.every(Number.isFinite)) continue;
          const pixels = bounds.every(v => v >= 0 && v <= 1)
            ? bounds.map((v, i) => v * (i % 2 ? s.frame_height : s.frame_width)) : bounds;
          const id = pixels.join(',');
          if (targets.has(id)) continue;
          targets.add(id);
          drawBox(pixels, person?.id ?? 5, person, 'person', true);
        }
      }
      const lines = new Map(Object.entries(warnings)
        .map(([code, value]) => [code, value?.count ?? value])
        .filter(([, count]) => Number.isFinite(count) && count > 0));
      for (const [code, count] of inferred) {
        if (!lines.has(code)) lines.set(code, count);
      }
      ctx.fillStyle = WARNING_COLOUR;
      [...lines].slice(0, 5).forEach(([code, count], i) => {
        const caption = dictionary.WARNING_LABELS?.[language]?.[code] || dictionary.WARNING_LABELS?.en?.[code] || code;
        ctx.fillText(count > 1 ? `${caption} x${count}` : caption, 8, 24 + i * 20);
      });
      ctx.restore();
    }
    request = requestAnimationFrame(draw);
  };
  request = requestAnimationFrame(draw);
  return { setSnapshot: snapshot => state.update(snapshot, performance.now()),
    setLanguage: value => { language = value; },
    destroy: () => { running = false; cancelAnimationFrame(request); ctx.clearRect(0, 0, canvas.width, canvas.height); } };
}
