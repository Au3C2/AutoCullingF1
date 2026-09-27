/**
 * Unit tests for the Intelligent High-Res Zoom on-demand state machine and
 * its wiring into the real UI sources.
 *
 * Part 1 exercises the debounce/generation-token contract that
 * scheduleHighResEvaluation + the highres_ready handler implement in
 * ui/app.js (mirrored here as a mock state manager so it runs under plain
 * node). Part 2 asserts the production wiring actually exists (Canvas 2D
 * overlay, on-demand request path, no stale SVG overlay remnants).
 */
const fs = require('fs');
const path = require('path');
const assert = require('assert');

// 1. High-Res State Machine and Debounce Logic Test
class MockHighResStateManager {
  constructor(dispatchCallback) {
    this.activeGenId = 0;
    this.debounceTimer = null;
    this.dispatchCallback = dispatchCallback;
    this.isHighResVisible = false;
  }

  onViewportChange(level, isIntent = false) {
    this.activeGenId++;
    const genId = this.activeGenId;

    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }

    if (level <= 1.5) {
      this.isHighResVisible = false;
      return;
    }

    const delay = isIntent ? 30 : 150;
    this.debounceTimer = setTimeout(() => {
      this.debounceTimer = null;
      this.dispatchCallback(genId);
    }, delay);
  }

  onHighResReady(genId, path) {
    if (genId !== this.activeGenId) {
      // Obsolete generation discarded
      return false;
    }
    this.isHighResVisible = true;
    return true;
  }
}

function testHighResStateMachine() {
  let dispatchedGenId = null;
  const sm = new MockHighResStateManager((genId) => {
    dispatchedGenId = genId;
  });

  // Normal zoom <= 1.5 does not trigger high-res
  sm.onViewportChange(1.2);
  assert.strictEqual(sm.isHighResVisible, false);
  assert.strictEqual(dispatchedGenId, null);

  // Rapid scrolling > 1.5 should debounce and only dispatch the LAST genId
  sm.onViewportChange(1.8);
  sm.onViewportChange(2.0);
  sm.onViewportChange(2.2);
  assert.strictEqual(dispatchedGenId, null); // still debouncing

  // Wait out the 150 ms debounce
  setTimeout(() => {
    // Generation increments even for the sub-threshold 1.2 zoom, so
    // 1 + 3 viewport changes = 4 (mirrors scheduleHighResEvaluation).
    assert.strictEqual(dispatchedGenId, 4);
    assert.strictEqual(sm.debounceTimer, null);

    // Response for the active generation is accepted
    assert.strictEqual(sm.onHighResReady(4, '/path/to/highres.jpg'), true);
    assert.strictEqual(sm.isHighResVisible, true);

    // Stale responses are discarded
    assert.strictEqual(sm.onHighResReady(2, '/stale.jpg'), false);

    console.log('✓ testHighResStateMachine passed');

    // Intent path (double-click / hotkey) dispatches after 30 ms, not 150 ms.
    // Check inside a 60 ms window: past the 30 ms intent debounce, but well
    // before the 150 ms normal debounce would fire.
    const sm2 = new MockHighResStateManager(() => {});
    let dispatched = false;
    sm2.dispatchCallback = () => { dispatched = true; };
    sm2.onViewportChange(2.0, true);
    setTimeout(() => {
      assert.ok(dispatched, 'intent path dispatched within 60ms');
      sm2.onViewportChange(1.0);
      assert.strictEqual(sm2.isHighResVisible, false);

      console.log('✓ testHighResIntentDebounce passed');
      testHighResWiring();
    }, 60);
  }, 250);
}

// 2. Production wiring smoke test against the real UI sources.
function testHighResWiring() {
  const baseDir = path.resolve(__dirname, '../../ui');
  const html = fs.readFileSync(path.join(baseDir, 'index.html'), 'utf8');
  const appJs = fs.readFileSync(path.join(baseDir, 'app.js'), 'utf8');

  // Canvas 2D overlay is the real implementation (a CSS-transformed SVG
  // overlay blanks sibling layers in WKWebView compositing — see the
  // drawBoxOverlay comment in app.js).
  assert.ok(html.includes('id="previewBoxCanvas"'), 'index.html must contain the canvas overlay element');
  assert.ok(appJs.includes('function drawBoxOverlay'), 'app.js must implement drawBoxOverlay');
  assert.ok(!html.includes('previewSvgOverlay'), 'stale SVG overlay element must be gone');

  // On-demand request plumbing
  assert.ok(appJs.includes("invokeTauri('request_highres'"), 'app.js must request highres via the Tauri command');
  assert.ok(appJs.includes("'highres_ready'") || appJs.includes('"highres_ready"'),
    'app.js must listen for highres_ready events');
  assert.ok(appJs.includes('scheduleHighResEvaluation(level, isIntent)'),
    'applyZoomTransform must forward the intent flag to the high-res scheduler');

  console.log('✓ testHighResWiring passed');
  console.log('All highres state tests passed');
}

testHighResStateMachine();
