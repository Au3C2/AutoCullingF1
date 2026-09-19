/**
 * Unit tests for Image Preview Pan & Zoom Interaction (Feature 3)
 */
const assert = require('assert');

class ZoomPanController {
  constructor(minZoom = 1.0, maxZoom = 5.0, step = 0.25) {
    this.minZoom = minZoom;
    this.maxZoom = maxZoom;
    this.step = step;
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;
  }

  onWheel(deltaY) {
    if (deltaY < 0) {
      this.zoom = Math.min(this.maxZoom, parseFloat((this.zoom + this.step).toFixed(2)));
    } else {
      this.zoom = Math.max(this.minZoom, parseFloat((this.zoom - this.step).toFixed(2)));
    }
    if (this.zoom === this.minZoom) {
      this.panX = 0;
      this.panY = 0;
    }
    return this.zoom;
  }

  onPan(dx, dy) {
    if (this.zoom <= 1.0) return false;
    this.panX += dx;
    this.panY += dy;
    return true;
  }

  onDoubleClick() {
    if (this.zoom > 1.0) {
      this.reset();
    } else {
      this.zoom = 2.5;
    }
    return this.zoom;
  }

  reset() {
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;
  }
}

// --- Test Cases ---
console.log('Testing Feature 3: Pan & Zoom Interactions...');

const ctrl = new ZoomPanController();

// 1. Zoom in & out
assert.strictEqual(ctrl.zoom, 1.0);
ctrl.onWheel(-100); // in
assert.strictEqual(ctrl.zoom, 1.25);

ctrl.onWheel(-100); // in
assert.strictEqual(ctrl.zoom, 1.5);

ctrl.onWheel(100); // out
assert.strictEqual(ctrl.zoom, 1.25);

// 2. Cannot pan when zoom <= 1.0
ctrl.reset();
assert.strictEqual(ctrl.onPan(10, 20), false);
assert.strictEqual(ctrl.panX, 0);

// 3. Pan works when zoomed in
ctrl.onWheel(-100); // 1.25
assert.strictEqual(ctrl.onPan(15, -25), true);
assert.strictEqual(ctrl.panX, 15);
assert.strictEqual(ctrl.panY, -25);

// 4. Zooming back to 1.0 resets pan
ctrl.onWheel(100); // 1.0
assert.strictEqual(ctrl.zoom, 1.0);
assert.strictEqual(ctrl.panX, 0);
assert.strictEqual(ctrl.panY, 0);

// 5. Double click toggles between 1.0 and 2.5
ctrl.onDoubleClick();
assert.strictEqual(ctrl.zoom, 2.5);
ctrl.onDoubleClick();
assert.strictEqual(ctrl.zoom, 1.0);

// 6. Reset on photo switch
ctrl.onWheel(-100);
ctrl.onPan(50, 50);
ctrl.reset();
assert.strictEqual(ctrl.zoom, 1.0);
assert.strictEqual(ctrl.panX, 0);

console.log('✔ Feature 3 test_preview_zoom.js passed successfully.');
