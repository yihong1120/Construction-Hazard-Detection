import test from 'node:test';
import assert from 'node:assert/strict';
import { contain, OverlayState, createOverlay, playbackMetadataPath, subscribeOverlay } from './overlay.mjs';
const frame = overrides => ({ owner: 'worker-a', generation: 2, sequence: 10,
  frame_width: 1920, frame_height: 1080, source_pts: 2, track_data: [], ...overrides });
test('contain maps portrait/square players without stretching source boxes', () => {
  assert.deepEqual(contain(640, 480, 1920, 1080), { scale: 1 / 3, x: 0, y: 60 });
});
test('fences old frames and resets on a new worker owner', () => {
  const state = new OverlayState();
  assert.equal(state.update(frame(), 0), true);
  assert.equal(state.update(frame({ generation: 1, sequence: 999 }), 10), false);
  assert.equal(state.update(frame({ sequence: 9 }), 10), false);
  assert.equal(state.update(frame({ owner: 'worker-b', generation: 1, sequence: 1 }), 10), true);
});
test('stale or misaligned metadata is hidden', () => {
  const state = new OverlayState(1000);
  state.update(frame(), 100);
  assert.equal(state.current(200, 2).sequence, 10);
  assert.equal(state.current(200, 3), null);
  assert.equal(state.current(1101), null);
});
test('clean preview playback routes one authorised same-origin metadata subscription', () => {
  const playback = {profile: 'clean', rendition: 'preview', label: '工地 A', stream_id: 'Y2FtMQ=='};
  const path = playbackMetadataPath(playback);
  assert.equal(path, '/bff/streaming/metadata/stream-id/%E5%B7%A5%E5%9C%B0%20A/Y2FtMQ%3D%3D');
  assert.throws(() => playbackMetadataPath({...playback, profile: 'overlay'}), TypeError);
  const original = globalThis.EventSource;
  const instances = [];
  globalThis.EventSource = class {
    constructor(url) { this.url = url; instances.push(this); }
    addEventListener(name, listener) { assert.equal(name, 'metadata'); this.listener = listener; }
    close() { this.closed = true; }
  };
  try {
    const entries = [];
    const detach = subscribeOverlay({setSnapshot: snapshot => entries.push(snapshot)}, path);
    assert.equal(instances.length, 1);
    assert.equal(instances[0].url, path);
    const overlay = frame();
    instances[0].listener({data: JSON.stringify({overlay})});
    assert.deepEqual(entries, [overlay]);
    detach();
    assert.equal(instances[0].closed, true);
  } finally {
    globalThis.EventSource = original;
  }
});
test('unchanged metadata does not redraw full overlays on every animation tick', () => {
  const callbacks = [], saved = [globalThis.requestAnimationFrame,
    globalThis.cancelAnimationFrame, globalThis.devicePixelRatio];
  globalThis.requestAnimationFrame = f => { callbacks.push(f); return callbacks.length; };
  globalThis.cancelAnimationFrame = () => {};
  globalThis.devicePixelRatio = 1;
  const boxes = [], captions = [];
  const ctx = new Proxy({}, { get: (state, name) => name === 'strokeRect' ?
    (...coordinates) => boxes.push({ coordinates, colour: state.strokeStyle }) :
    name === 'fillText' ? text => captions.push(text) : () => {} });
  const canvas = { width: 0, height: 0, style: {}, getContext: () => ctx };
  const video = { clientWidth: 640, clientHeight: 480, videoWidth: 640, videoHeight: 360,
    readyState: 2, paused: false, style: {} };
  const renderer = createOverlay(video, canvas, { dictionary: {
    CLASS_LABELS: { 'zh-TW': { no_helmet: '未戴安全帽' }, en: { no_helmet: 'No helmet' } },
  } });
  try {
    renderer.setSnapshot(frame({ frame_width: 1280, frame_height: 960,
      track_data: [[20, 80, 60, 160, .9, 2, 7, 0], [20, 80, 60, 160, .9, 6, 8, 0]],
      class_metadata: [{ id: 2, code: 'no_helmet', color: '#F44336' }] }));
    callbacks.shift()(performance.now());
    callbacks.shift()(performance.now());
    assert.deepEqual(boxes, [{coordinates: [10, 30, 20, 30], colour: '#F44336'}]);
    assert.match(captions[0], /未戴安全帽/);
    renderer.setLanguage('en'); callbacks.shift()(performance.now());
    assert.equal(boxes.length, 2);
    assert.match(captions[2], /No helmet/);
    renderer.destroy();
  } finally {
    [globalThis.requestAnimationFrame, globalThis.cancelAnimationFrame,
      globalThis.devicePixelRatio] = saved;
  }
});
