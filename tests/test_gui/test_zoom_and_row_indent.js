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

// Helper function for crop focus calculation
function computeCropFocus(crop, containerW = 800, containerH = 600) {
  if (!crop || crop.length !== 4) {
    return { level: 1.0, panX: 0, panY: 0 };
  }
  const [top, left, bottom, right] = crop;
  const cropW = Math.max(0.01, right - left);
  const cropH = Math.max(0.01, bottom - top);
  const cx = (left + right) / 2.0;
  const cy = (top + bottom) / 2.0;

  // Outer margin factor: 1.30x (retaining context around the crop)
  const marginFactor = 1.30;
  const scaleX = 1.0 / (cropW * marginFactor);
  const scaleY = 1.0 / (cropH * marginFactor);
  let targetScale = Math.min(scaleX, scaleY);

  // Clamp zoom level between 1.0 and 2.5
  targetScale = Math.max(1.0, Math.min(2.5, parseFloat(targetScale.toFixed(2))));

  // Pan translation to center (cx, cy) in viewport
  const panX = (0.5 - cx) * containerW * targetScale;
  const panY = (0.5 - cy) * containerH * targetScale;

  return {
    level: targetScale,
    panX: parseFloat(panX.toFixed(1)),
    panY: parseFloat(panY.toFixed(1)),
  };
}

// Helper function for mouse-centered zoom transformation
function computeMouseCenteredZoom(oldLevel, newLevel, oldPanX, oldPanY, mouseDx, mouseDy) {
  if (newLevel <= 1.0) {
    return { level: 1.0, panX: 0, panY: 0 };
  }
  const ratio = newLevel / oldLevel;
  const newPanX = mouseDx - (mouseDx - oldPanX) * ratio;
  const newPanY = mouseDy - (mouseDy - oldPanY) * ratio;
  return {
    level: newLevel,
    panX: parseFloat(newPanX.toFixed(1)),
    panY: parseFloat(newPanY.toFixed(1)),
  };
}

// 2. Test Crop Center Adaptive Focus Calculation
function testCropFocusCalculation() {
  // Centered moderate crop
  const focusCenter = computeCropFocus([0.2, 0.25, 0.8, 0.75], 800, 600);
  assert(focusCenter.level >= 1.0 && focusCenter.level <= 2.5);
  assert.strictEqual(focusCenter.panX, 0, 'Centered crop should have 0 panX');
  assert.strictEqual(focusCenter.panY, 0, 'Centered crop should have 0 panY');

  // Off-center crop to the right: cx = 0.8
  const focusRight = computeCropFocus([0.3, 0.6, 0.7, 1.0], 800, 600);
  assert(focusRight.panX < 0, 'Off-center right crop must pan leftwards (negative panX)');
  assert(focusRight.level >= 1.2, 'Crop focus must zoom in');

  // Test Mouse-centered zoom calculation
  // 1. Center mouse: dx=0, dy=0, level 1.0 -> 2.0 -> panX=0, panY=0
  const z1 = computeMouseCenteredZoom(1.0, 2.0, 0, 0, 0, 0);
  assert.strictEqual(z1.level, 2.0);
  assert.strictEqual(z1.panX, 0);
  assert.strictEqual(z1.panY, 0);

  // 2. Off-center mouse: dx=100, dy=50, level 1.0 -> 2.0 -> panX must shift by -100
  const z2 = computeMouseCenteredZoom(1.0, 2.0, 0, 0, 100, 50);
  assert.strictEqual(z2.level, 2.0);
  assert.strictEqual(z2.panX, -100);
  assert.strictEqual(z2.panY, -50);

  console.log('✓ testCropFocusCalculation & MouseCenteredZoom passed');
}

// 3. Test Zoom Slider Steps (100, 150, 200, 250), Reset to 100%, and Input Clamp (100-250)
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

  // Test clamp logic (100 to 250)
  function clampZoomInput(val) {
    let v = parseInt(val, 10);
    if (isNaN(v) || v < 100) v = 100;
    if (v > 250) v = 250;
    return v;
  }
  assert.strictEqual(clampZoomInput(80), 100);
  assert.strictEqual(clampZoomInput(180), 180);
  assert.strictEqual(clampZoomInput(350), 250);

  // Test resetZoom always resets to 100% (level = 1.0, manual)
  const state = { zoom: { level: 2.5, panX: 100, panY: 50, mode: 'auto' } };
  function resetZoom() {
    state.zoom.mode = 'manual';
    state.zoom.level = 1.0;
    state.zoom.panX = 0;
    state.zoom.panY = 0;
  }
  resetZoom();
  assert.strictEqual(state.zoom.level, 1.0, 'Reset zoom must strictly reset to 100%');
  assert.strictEqual(state.zoom.mode, 'manual');
  assert.strictEqual(state.zoom.panX, 0);
  assert.strictEqual(state.zoom.panY, 0);

  // Test switching to Auto mode (slider value 50)
  function handleSliderChange(val, photo) {
    if (val <= 50) {
      state.zoom.mode = 'auto';
      if (photo && photo.crop) {
        const focus = computeCropFocus(photo.crop);
        state.zoom.level = focus.level;
        state.zoom.panX = focus.panX;
        state.zoom.panY = focus.panY;
      } else {
        state.zoom.level = 1.0;
        state.zoom.panX = 0;
        state.zoom.panY = 0;
      }
      return 50;
    }
    state.zoom.mode = 'manual';
    state.zoom.level = val / 100;
    return val;
  }

  const dummyPhoto = { crop: [0.2, 0.2, 0.8, 0.8] };
  const sliderVal = handleSliderChange(50, dummyPhoto);
  assert.strictEqual(state.zoom.mode, 'auto', 'Mode must switch to auto when slider dragged to 50');
  assert.strictEqual(sliderVal, 50, 'Slider value must remain at 50 in auto mode');
  assert(state.zoom.level > 1.0, 'Auto mode must calculate crop focus zoom');

  console.log('✓ testZoomSliderAndInputSync passed');
}

// 4. Test Manual Save Button Behavior (Spinner icon, save all scored if no dirty, and prompt)
function testSaveButtonInteraction() {
  const state = {
    isSaving: false,
    dirtyPhotos: new Set(),
    photos: [
      { path: '/a.arw', rating: 5, crop: null, status: 'scored' },
      { path: '/b.arw', rating: -1, crop: null, status: 'scored' },
    ],
  };

  const savedBatches = [];
  function mockSaveTauri(items) {
    savedBatches.push(items);
    return Promise.resolve({ count: items.length });
  }

  let toastMessage = null;
  function mockShowToast(msg) {
    toastMessage = msg;
  }

  let iconSpinning = false;
  function setSaveIconSpinning(spinning) {
    iconSpinning = spinning;
  }

  async function handleSaveButtonClick() {
    if (state.isSaving) return;
    state.isSaving = true;
    setSaveIconSpinning(true);

    // If dirtyPhotos has items, save dirty; if empty, save all scored items so manual click always works!
    const targetItems = state.dirtyPhotos.size > 0 
      ? Array.from(state.dirtyPhotos)
      : state.photos.filter((p) => p.status === 'scored');

    const itemsToSave = targetItems.map((p) => ({ path: p.path, rating: p.rating, crop: p.crop }));

    try {
      if (itemsToSave.length > 0) {
        await mockSaveTauri(itemsToSave);
        state.dirtyPhotos.clear();
      }
      mockShowToast('保存完成');
    } finally {
      state.isSaving = false;
      setSaveIconSpinning(false);
    }
  }

  return handleSaveButtonClick().then(() => {
    assert.strictEqual(savedBatches.length, 1);
    assert.strictEqual(savedBatches[0].length, 2, 'Must save all scored items even if dirtyPhotos was clean');
    assert.strictEqual(iconSpinning, false, 'Spinner must be removed after save');
    assert.strictEqual(toastMessage, '保存完成');
    console.log('✓ testSaveButtonInteraction passed');
  });
}

testRowIndentationPreservation();
testCropFocusCalculation();
testZoomSliderAndInputSync();
testSaveButtonInteraction().catch((err) => {
  console.error('Test failed:', err);
  process.exit(1);
});

