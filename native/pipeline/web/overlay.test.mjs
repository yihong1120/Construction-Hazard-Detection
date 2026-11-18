import test from 'node:test';
import assert from 'node:assert/strict';
import { contain, OverlayState, createOverlay } from './overlay.mjs';
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
test('unchanged metadata does not redraw full overlays on every animation tick', () => {
  const callbacks = [], saved = [globalThis.requestAnimationFrame,
    globalThis.cancelAnimationFrame, globalThis.devicePixelRatio];
  globalThis.requestAnimationFrame = f => { callbacks.push(f); return callbacks.length; };
  globalThis.cancelAnimationFrame = () => {};
  globalThis.devicePixelRatio = 1;
  let boxes = 0; const captions = [];
  const ctx = new Proxy({}, { get: (_, name) => name === 'strokeRect' ?
    () => { boxes++; } : name === 'fillText' ? text => captions.push(text) : () => {} });
  const canvas = { width: 0, height: 0, style: {}, getContext: () => ctx };
  const video = { clientWidth: 640, clientHeight: 480, readyState: 2, paused: false, style: {} };
  const renderer = createOverlay(video, canvas, { dictionary: {
    CLASS_LABELS: { 'zh-TW': { no_helmet: '未戴安全帽' }, en: { no_helmet: 'No helmet' } },
  } });
  try {
    renderer.setSnapshot(frame({ track_data: [[10, 20, 30, 40, .9, 2, 7, 0]],
      class_metadata: [{ id: 2, code: 'no_helmet' }] }));
    callbacks.shift()(performance.now());
    callbacks.shift()(performance.now());
    assert.equal(boxes, 1);
    assert.match(captions[0], /未戴安全帽/);
    renderer.setLanguage('en'); callbacks.shift()(performance.now());
    assert.equal(boxes, 2);
    assert.match(captions[1], /No helmet/);
    renderer.destroy();
  } finally {
    [globalThis.requestAnimationFrame, globalThis.cancelAnimationFrame,
      globalThis.devicePixelRatio] = saved;
  }
});
