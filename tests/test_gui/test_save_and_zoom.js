/**
 * Unit tests for Viewport Lock & Metadata Save State Machine
 */
const assert = require('assert');

// 1. Test Viewport Lock Logic
function testViewportLockBehavior() {
  const state = {
    zoom: {
      level: 2.5,
      panX: 120,
      panY: -50,
      locked: false,
    },
    selectedPhoto: { name: 'DSC001.JPG', path: '/photos/DSC001.JPG' },
  };

  function selectPhoto(newPhoto, shouldResetZoom = true) {
    state.selectedPhoto = newPhoto;
    if (shouldResetZoom && (!state.zoom.locked || state.zoom.level <= 1.0)) {
      state.zoom.level = 1.0;
      state.zoom.panX = 0;
      state.zoom.panY = 0;
    }
  }

  // When unlocked: selecting next photo resets zoom
  selectPhoto({ name: 'DSC002.JPG', path: '/photos/DSC002.JPG' }, true);
  assert.strictEqual(state.zoom.level, 1.0, 'Zoom should reset when unlocked');
  assert.strictEqual(state.zoom.panX, 0);

  // Now lock viewport at 3.0x zoom
  state.zoom.level = 3.0;
  state.zoom.panX = 200;
  state.zoom.panY = -80;
  state.zoom.locked = true;

  // When locked: selecting next photo PRESERVES zoom level and pan coordinates
  selectPhoto({ name: 'DSC003.JPG', path: '/photos/DSC003.JPG' }, true);
  assert.strictEqual(state.zoom.level, 3.0, 'Zoom level must be preserved when locked');
  assert.strictEqual(state.zoom.panX, 200, 'Pan X must be preserved when locked');
  assert.strictEqual(state.zoom.panY, -80, 'Pan Y must be preserved when locked');
  console.log('✓ testViewportLockBehavior passed');
}

// 2. Test Save Triggers & Dirty Set Management
function testSaveTriggers() {
  const state = {
    dirtyPhotos: new Set(),
    isSaving: false,
    photos: [
      { name: 'DSC001.JPG', path: '/photos/DSC001.JPG', rating: 0, crop: null },
      { name: 'DSC002.JPG', path: '/photos/DSC002.JPG', rating: 0, crop: null },
    ],
  };

  const savedItems = [];
  function mockInvokeTauri(cmd, payload) {
    if (cmd === 'save_metadata') {
      savedItems.push(...payload.items);
      return Promise.resolve({ status: 'ok' });
    }
    return Promise.resolve();
  }

  function manuallySetRating(photo, newRating) {
    photo.rating = newRating;
    state.dirtyPhotos.add(photo);
  }

  // Trigger 2: user modifies rating -> photo marked dirty
  manuallySetRating(state.photos[0], 5);
  assert.strictEqual(state.dirtyPhotos.size, 1);
  assert(state.dirtyPhotos.has(state.photos[0]));

  manuallySetRating(state.photos[1], -1);
  assert.strictEqual(state.dirtyPhotos.size, 2);

  // Trigger 3: flushSaveMetadata -> dispatches payload and clears saved
  async function flushSaveMetadata() {
    const itemsToSave = [];
    for (const p of state.dirtyPhotos) {
      itemsToSave.push({ path: p.path, rating: p.rating, crop: p.crop });
    }
    await mockInvokeTauri('save_metadata', { items: itemsToSave });
    for (const it of itemsToSave) {
      const found = state.photos.find((p) => p.path === it.path);
      if (found) state.dirtyPhotos.delete(found);
    }
  }

  return flushSaveMetadata().then(() => {
    assert.strictEqual(savedItems.length, 2);
    assert.strictEqual(savedItems[0].rating, 5);
    assert.strictEqual(savedItems[1].rating, -1);
    assert.strictEqual(state.dirtyPhotos.size, 0, 'Dirty set must be empty after save');
    console.log('✓ testSaveTriggers passed');
  });
}

testViewportLockBehavior();
testSaveTriggers().catch((err) => {
  console.error('Test failed:', err);
  process.exit(1);
});
