/**
 * Unit tests for Intelligent High-Res Zoom on-demand state machine and SVG overlay.
 */
const assert = require('assert');

// 1. High-Res State Machine and Debounce Logic Test
class MockHighResStateManager {
  constructor(dispatchCallback) {
    this.activeGenId = 0;
    this.debounceTimer = null;
    this.dispatchCallback = dispatchCallback;
    this.isHighResVisible = false;
    this.loadedPath = null;
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
    this.loadedPath = path;
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

  // Rapid scrolling > 1.5 should debounce and only dispatch the last genId
  sm.onViewportChange(1.8);
  sm.onViewportChange(2.0);
  sm.onViewportChange(2.2);
  assert.strictEqual(dispatchedGenId, null);

  // Wait for debounce timer (150ms)
  setTimeout(() => {
    assert.strictEqual(dispatchedGenId, 4); // 1 + 3 changes = 4
    // Simulate response arrives
    const ok = sm.onHighResReady(4, '/path/to/highres.jpg');
    assert.strictEqual(ok, true);
    assert.strictEqual(sm.isHighResVisible, true);

    // Obsolete response test
    const staleOk = sm.onHighResReady(2, '/stale.jpg');
    assert.strictEqual(staleOk, false);
    console.log('✓ testHighResStateMachine passed');
  }, 200);
}

// 2. SVG Normalized Vector Overlay Calculation Test
function testSvgOverlayGeneration() {
  function renderSvgBoxes(boxes, imgW, imgH) {
    if (!boxes || boxes.length === 0) return '';
    return boxes.map((box) => {
      const [x1, y1, x2, y2, label, conf] = box;
      // Normalized coordinates [0, 1]
      const nx = Math.min(x1, x2) / imgW;
      const ny = Math.min(y1, y2) / imgH;
      const nw = Math.abs(x2 - x1) / imgW;
      const nh = Math.abs(y2 - y1) / imgH;
      return `<rect x="${nx.toFixed(4)}" y="${ny.toFixed(4)}" width="${nw.toFixed(4)}" height="${nh.toFixed(4)}" />`;
    }).join('');
  }

  const boxes = [[100, 150, 500, 450, 'car', 0.95]];
  const svg = renderSvgBoxes(boxes, 1000, 1000);
  assert(svg.includes('x="0.1000"'));
  assert(svg.includes('y="0.1500"'));
  assert(svg.includes('width="0.4000"'));
  assert(svg.includes('height="0.3000"'));
  console.log('✓ testSvgOverlayGeneration passed');
}

testHighResStateMachine();
testSvgOverlayGeneration();
