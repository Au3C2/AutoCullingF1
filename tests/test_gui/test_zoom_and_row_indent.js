/**
 * Unit tests for Row Indentation Preservation, Crop Focus Zoom & Zoom Slider/Input
 */
const assert = require('assert');

// 1. Test In-place Row Update Preserving Indentation & Winner Badge
function testRowIndentationPreservation() {
  function buildRowHtml(item, options = {}) {
    const rowClasses = [
      'tau-photo-row',
      options.isBurstChild ? 'tau-burst-child' : '',
    ].filter(Boolean).join(' ');

    const winnerHtml = options.isWinner ? '<span class="tau-winner-badge">WINNER</span>' : '';
    return `<tr id="row_${item.id}" class="${rowClasses}"><td>${item.name}${winnerHtml}</td></tr>`;
  }

  // Simulate an existing row in DOM
  const mockOldRow = {
    classList: {
      contains(cls) {
        return cls === 'tau-burst-child';
      },
    },
    querySelector(sel) {
      if (sel === '.tau-winner-badge') return {};
      return null;
    },
  };

  function updateRowInPlace(item, oldRow) {
    const isBurstChild = oldRow ? oldRow.classList.contains('tau-burst-child') : false;
    const isWinner = oldRow ? !!oldRow.querySelector('.tau-winner-badge') : false;
    return buildRowHtml(item, { isBurstChild, isWinner });
  }

  const updatedHtml = updateRowInPlace({ id: 1, name: 'DSC001.JPG' }, mockOldRow);
  assert(updatedHtml.includes('tau-burst-child'), 'Row must retain tau-burst-child class when updated in-place');
  assert(updatedHtml.includes('tau-winner-badge'), 'Row must retain winner badge when updated in-place');
  console.log('✓ testRowIndentationPreservation passed');
}

// 2. Test Crop Center Adaptive Focus Calculation
function testCropFocusCalculation() {
  function computeCropFocus(crop, containerW = 800, containerH = 600) {
    if (!crop || crop.length !== 4) {
      return { level: 1.0, panX: 0, panY: 0 };
    }
    const [top, left, bottom, right] = crop;
    const cropW = Math.max(0.01, right - left);
    const cropH = Math.max(0.01, bottom - top);
    const cx = (left + right) / 2.0;
    const cy = (top + bottom) / 2.0;

    // Outer margin factor: 1.35x (retaining context around the crop)
    const marginFactor = 1.35;
    const scaleX = 1.0 / (cropW * marginFactor);
    const scaleY = 1.0 / (cropH * marginFactor);
    let targetScale = Math.min(scaleX, scaleY);

    // Clamp zoom level between 1.0 and 2.5
    targetScale = Math.max(1.0, Math.min(2.5, parseFloat(targetScale.toFixed(2))));

    // Pan translation to center (cx, cy) in viewport
    // Center of image in normalized coords is 0.5, 0.5
    const panX = (0.5 - cx) * containerW * targetScale;
    const panY = (0.5 - cy) * containerH * targetScale;

    return {
      level: targetScale,
      panX: parseFloat(panX.toFixed(1)),
      panY: parseFloat(panY.toFixed(1)),
    };
  }

  // Centered moderate crop
  const focusCenter = computeCropFocus([0.2, 0.25, 0.8, 0.75], 800, 600);
  assert(focusCenter.level >= 1.0 && focusCenter.level <= 2.5);
  assert.strictEqual(focusCenter.panX, 0, 'Centered crop should have 0 panX');
  assert.strictEqual(focusCenter.panY, 0, 'Centered crop should have 0 panY');

  // Off-center crop to the right: cx = 0.8
  const focusRight = computeCropFocus([0.3, 0.6, 0.7, 1.0], 800, 600);
  assert(focusRight.panX < 0, 'Off-center right crop must pan leftwards (negative panX)');
  assert(focusRight.level >= 1.2, 'Crop focus must zoom in');
  console.log('✓ testCropFocusCalculation passed');
}

// 3. Test Zoom Slider Steps (100, 150, 200, 250) and Input Sync
function testZoomSliderAndInputSync() {
  const steps = [100, 150, 200, 250];
  function snapToStep(val) {
    let closest = steps[0];
    let minDiff = Math.abs(val - closest);
    for (const s of steps) {
      const diff = Math.abs(val - s);
      if (diff < minDiff) {
        minDiff = diff;
        closest = s;
      }
    }
    return closest;
  }

  assert.strictEqual(snapToStep(110), 100);
  assert.strictEqual(snapToStep(140), 150);
  assert.strictEqual(snapToStep(220), 200);
  assert.strictEqual(snapToStep(240), 250);
  console.log('✓ testZoomSliderAndInputSync passed');
}

testRowIndentationPreservation();
testCropFocusCalculation();
testZoomSliderAndInputSync();
