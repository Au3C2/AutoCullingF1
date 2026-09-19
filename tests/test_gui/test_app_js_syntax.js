/**
 * Verification test for ui/app.js syntax and top-level evaluation.
 */
const fs = require('fs');
const path = require('path');
const assert = require('assert');
const vm = require('vm');

const appJsPath = path.resolve(__dirname, '../../ui/app.js');
const code = fs.readFileSync(appJsPath, 'utf8');

// 1. Verify syntax via new vm.Script
let script;
try {
  script = new vm.Script(code, { filename: 'ui/app.js' });
} catch (e) {
  assert.fail(`Syntax error in ui/app.js: ${e.message}\n${e.stack}`);
}

// 2. Mock browser DOM environment and execute script
const domElements = new Map();
function getOrCreateElement(id) {
  if (!domElements.has(id)) {
    domElements.set(id, {
      id,
      style: {},
      classList: {
        add: () => {},
        remove: () => {},
        contains: () => false,
      },
      addEventListener: () => {},
      setAttribute: () => {},
      getAttribute: () => null,
      removeAttribute: () => {},
      appendChild: () => {},
      children: [],
      textContent: '',
      value: '',
    });
  }
  return domElements.get(id);
}

const mockStorage = new Map();

const context = vm.createContext({
  window: {
    addEventListener: () => {},
    innerWidth: 1280,
    innerHeight: 800,
    __TAURI__: {
      core: { invoke: async () => 6 },
      event: { listen: async () => () => {} },
      webviewWindow: {
        getCurrentWebviewWindow: () => ({
          onDragDropEvent: () => {},
        }),
      },
    },
  },
  document: {
    getElementById: (id) => getOrCreateElement(id),
    querySelector: (sel) => getOrCreateElement(sel),
    querySelectorAll: () => [],
    createElement: (tag) => getOrCreateElement(tag),
    documentElement: { lang: 'zh-CN' },
    title: '',
  },
  navigator: {
    language: 'zh-CN',
    clipboard: { writeText: async () => {} },
  },
  localStorage: {
    getItem: (k) => mockStorage.get(k) || null,
    setItem: (k, v) => mockStorage.set(k, String(v)),
    removeItem: (k) => mockStorage.delete(k),
  },
  fetch: async () => ({
    json: async () => ({}),
  }),
  console,
  setTimeout,
  clearTimeout,
  performance: { now: () => Date.now() },
});

try {
  script.runInContext(context);
  console.log('✔ ui/app.js evaluated cleanly with zero syntax or runtime initialization errors.');
} catch (err) {
  assert.fail(`Runtime error during ui/app.js execution: ${err.message}\n${err.stack}`);
}
