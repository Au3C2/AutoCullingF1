/**
 * Unit tests for Right-Click Context Menu Positioning & Clamping (Feature 7)
 */
const assert = require('assert');

function calculateMenuPosition(clickX, clickY, menuWidth, menuHeight, winWidth, winHeight) {
  let x = clickX;
  let y = clickY;

  if (x + menuWidth > winWidth) {
    x = Math.max(0, winWidth - menuWidth - 8);
  }
  if (y + menuHeight > winHeight) {
    y = Math.max(0, winHeight - menuHeight - 8);
  }

  return { x, y };
}

// --- Test Cases ---
console.log('Testing Feature 7: Context Menu Positioning...');

const menuW = 180;
const menuH = 150;
const winW = 1000;
const winH = 800;

// Case 1: Normal middle click
let pos = calculateMenuPosition(200, 300, menuW, menuH, winW, winH);
assert.strictEqual(pos.x, 200);
assert.strictEqual(pos.y, 300);

// Case 2: Near right edge
pos = calculateMenuPosition(950, 300, menuW, menuH, winW, winH);
assert(pos.x <= winW - menuW);
assert.strictEqual(pos.x, 1000 - 180 - 8);

// Case 3: Near bottom edge
pos = calculateMenuPosition(200, 750, menuW, menuH, winW, winH);
assert(pos.y <= winH - menuH);
assert.strictEqual(pos.y, 800 - 150 - 8);

console.log('✔ Feature 7 test_context_menu.js passed successfully.');
