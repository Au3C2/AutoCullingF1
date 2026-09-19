/**
 * Unit tests for Keyboard Navigation & Culling Hotkeys (Feature 1)
 */
const assert = require('assert');

function handleHotkey(event, state, actions) {
  // Input protection: ignore if target is editable
  const tag = (event.target && event.target.tagName ? event.target.tagName.toLowerCase() : '');
  if (tag === 'input' || tag === 'textarea' || tag === 'select') {
    return false;
  }
  if (event.ctrlKey || event.metaKey || event.altKey) {
    return false;
  }

  const key = event.key ? event.key.toLowerCase() : '';
  const visible = state.visiblePhotos || [];
  if (visible.length === 0) return false;

  const currentIdx = visible.findIndex((p) => p.path === (state.selectedPhoto ? state.selectedPhoto.path : null));

  switch (key) {
    case 'arrowdown':
    case 'j': {
      const nextIdx = currentIdx < visible.length - 1 ? currentIdx + 1 : currentIdx;
      actions.selectPhoto(visible[nextIdx]);
      return true;
    }
    case 'arrowup':
    case 'k': {
      const prevIdx = currentIdx > 0 ? currentIdx - 1 : 0;
      actions.selectPhoto(visible[prevIdx]);
      return true;
    }
    case 'home': {
      actions.selectPhoto(visible[0]);
      return true;
    }
    case 'end': {
      actions.selectPhoto(visible[visible.length - 1]);
      return true;
    }
    case '1':
    case '2':
    case '3':
    case '4':
    case '5': {
      if (state.selectedPhoto) {
        actions.setRating(state.selectedPhoto, parseInt(key, 10));
        return true;
      }
      break;
    }
    case '0':
    case 'x': {
      if (state.selectedPhoto) {
        actions.setRating(state.selectedPhoto, -1);
        return true;
      }
      break;
    }
    case ' ': {
      actions.togglePreviewMode();
      return true;
    }
  }
  return false;
}

// --- Test Cases ---
console.log('Testing Feature 1: Keyboard Navigation & Shortcuts...');

const photos = [
  { name: 'p1.jpg', path: '/p1.jpg', rating: 0 },
  { name: 'p2.jpg', path: '/p2.jpg', rating: 0 },
  { name: 'p3.jpg', path: '/p3.jpg', rating: 0 },
];

let selected = photos[0];
let previewToggled = false;

const state = {
  visiblePhotos: photos,
  get selectedPhoto() { return selected; },
};

const actions = {
  selectPhoto: (p) => { selected = p; },
  setRating: (p, r) => { p.rating = r; },
  togglePreviewMode: () => { previewToggled = !previewToggled; },
};

// 1. ArrowDown navigation
assert(handleHotkey({ key: 'ArrowDown' }, state, actions));
assert.strictEqual(selected.path, '/p2.jpg');

assert(handleHotkey({ key: 'j' }, state, actions));
assert.strictEqual(selected.path, '/p3.jpg');

// Boundary: cannot go past end
assert(handleHotkey({ key: 'ArrowDown' }, state, actions));
assert.strictEqual(selected.path, '/p3.jpg');

// 2. ArrowUp navigation
assert(handleHotkey({ key: 'k' }, state, actions));
assert.strictEqual(selected.path, '/p2.jpg');

// 3. Home / End
assert(handleHotkey({ key: 'Home' }, state, actions));
assert.strictEqual(selected.path, '/p1.jpg');

assert(handleHotkey({ key: 'End' }, state, actions));
assert.strictEqual(selected.path, '/p3.jpg');

// 4. Rating hotkeys: 1-5 and x/0
assert(handleHotkey({ key: '5' }, state, actions));
assert.strictEqual(selected.rating, 5);

assert(handleHotkey({ key: 'x' }, state, actions));
assert.strictEqual(selected.rating, -1);

// 5. Space toggle
assert(handleHotkey({ key: ' ' }, state, actions));
assert.strictEqual(previewToggled, true);

// 6. Focus protection: typing inside input should be ignored
assert.strictEqual(handleHotkey({ key: '5', target: { tagName: 'INPUT' } }, state, actions), false);
assert.strictEqual(handleHotkey({ key: 'j', target: { tagName: 'TEXTAREA' } }, state, actions), false);

console.log('✔ Feature 1 test_keyboard_shortcuts.js passed successfully.');
