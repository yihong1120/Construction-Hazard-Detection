// Canvas overlay for a clean <video> and the existing authorised metadata API.
// Call setSnapshot(event.overlay) from SSE/WS. Never adds a media connection.
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

export function createOverlay(video, canvas, { language = 'zh-TW', dictionary = {},
  maxAgeMs = 1000, sourcePts = () => null } = {}) {
  const state = new OverlayState(maxAgeMs), ctx = canvas.getContext('2d');
  let running = true, request, previous = null;
  video.style.objectFit = 'contain';
  const label = (s, id) => {
    const c = (s.class_metadata || []).find(c => c.id === id);
    return s.labels?.[id]?.[language] || dictionary.CLASS_LABELS?.[language]?.[c?.code] ||
      c?.display_name || s.labels?.[id]?.en || c?.code || String(id);
  };
  const draw = now => {
    if (!running) return;
    const width = video.clientWidth, height = video.clientHeight, dpr = devicePixelRatio || 1;
    canvas.style.width = `${width}px`; canvas.style.height = `${height}px`;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
    }
    const s = video.readyState >= 2 && !video.paused ? state.current(now, sourcePts()) : null;
    const key = [s?.owner, s?.generation, s?.sequence, language, width, height, dpr].join('|');
    if (key === previous) { request = requestAnimationFrame(draw); return; }
    previous = key;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
    if (s && video.readyState >= 2 && !video.paused) {
      const t = contain(width, height, s.frame_width, s.frame_height);
      ctx.save(); ctx.translate(t.x, t.y); ctx.scale(t.scale, t.scale);
      ctx.beginPath(); ctx.rect(0, 0, s.frame_width, s.frame_height); ctx.clip();
      ctx.lineWidth = 2 / t.scale; ctx.font = `${14 / t.scale}px sans-serif`;
      for (const polygon of [...(s.cone_polys || []), ...(s.pole_polys || [])]) {
        if (!Array.isArray(polygon) || polygon.length < 3 ||
            !polygon.every(p => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite))) continue;
        ctx.beginPath(); polygon.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
        ctx.closePath(); ctx.fillStyle = 'rgba(255,160,0,.18)'; ctx.fill();
        ctx.strokeStyle = '#ffa000'; ctx.stroke();
      }
      for (const row of s.track_data) {
        if (!Array.isArray(row) || row.length < 7 || !row.every(Number.isFinite)) continue;
        const [x1, y1, x2, y2, confidence, id, track] = row;
        ctx.strokeStyle = '#29ef8e'; ctx.fillStyle = '#29ef8e';
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.fillText(`${label(s, id)} #${track} ${(confidence * 100).toFixed(0)}%`, x1, Math.max(14 / t.scale, y1 - 4 / t.scale));
      }
      ctx.restore();
      ctx.fillStyle = '#ff5252'; ctx.font = '16px sans-serif';
      Object.entries(s.warnings || {}).forEach(([code, value], i) => {
        const text = dictionary.WARNING_LABELS?.[language]?.[code] || code;
        ctx.fillText(`${text}: ${value.count ?? ''}`, 8, 24 + i * 20);
      });
    }
    request = requestAnimationFrame(draw);
  };
  request = requestAnimationFrame(draw);
  return { setSnapshot: snapshot => state.update(snapshot, performance.now()),
    setLanguage: value => { language = value; },
    destroy: () => { running = false; cancelAnimationFrame(request); ctx.clearRect(0, 0, canvas.width, canvas.height); } };
}
