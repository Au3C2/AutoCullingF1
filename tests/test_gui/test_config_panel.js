/**
 * Unit tests for Collapsible & Compact Config Panel (Feature 2)
 */
const assert = require('assert');

function generateConfigSummary(config) {
  const parts = [];
  if (config.top_n !== undefined) parts.push(`Top-N: ${config.top_n}`);
  if (config.workers !== undefined) parts.push(`Workers: ${config.workers}`);
  if (config.sharp_thresh !== undefined) parts.push(`Sharp: ${config.sharp_thresh}`);
  if (config.conf !== undefined) parts.push(`YOLO: ${config.conf}`);
  if (config.p4_policy && config.p4_policy !== 'never') parts.push(`P4: ${config.p4_policy}`);
  return parts.join(' · ');
}

function resetConfigToDefaults(paramsMeta, currentConfig, storage) {
  const resetConfig = {};
  for (const p of paramsMeta) {
    resetConfig[p.key] = p.default;
    if (storage) {
      storage.removeItem(`ac-param-${p.key}`);
    }
  }
  return resetConfig;
}

// --- Test Cases ---
console.log('Testing Feature 2: Config Panel Collapse & Defaults...');

const mockParams = [
  { key: 'top_n', default: 11 },
  { key: 'workers', default: 4 },
  { key: 'sharp_thresh', default: 0.05 },
  { key: 'conf', default: 0.25 },
  { key: 'p4_policy', default: 'never' },
];

const activeConfig = {
  top_n: 15,
  workers: 8,
  sharp_thresh: 0.08,
  conf: 0.35,
  p4_policy: 'always',
};

// 1. Summary string generation
const summary = generateConfigSummary(activeConfig);
assert(summary.includes('Top-N: 15'));
assert(summary.includes('Workers: 8'));
assert(summary.includes('Sharp: 0.08'));
assert(summary.includes('YOLO: 0.35'));
assert(summary.includes('P4: always'));

// 2. Reset defaults
const mockStorage = new Map();
mockStorage.set('ac-param-top_n', '15');

const defaults = resetConfigToDefaults(mockParams, activeConfig, {
  removeItem: (k) => mockStorage.delete(k)
});

assert.strictEqual(defaults.top_n, 11);
assert.strictEqual(defaults.workers, 4);
assert.strictEqual(defaults.p4_policy, 'never');
assert(!mockStorage.has('ac-param-top_n'), 'Storage key should be cleared');

// 3. Dynamic optimal workers override on hybrid architecture (e.g. Apple Silicon 6 E-cores)
const optimalWorkers = 6;
const workerParam = mockParams.find((p) => p.key === 'workers');
if (workerParam) workerParam.default = optimalWorkers;
const dynamicDefaults = resetConfigToDefaults(mockParams, activeConfig, null);
assert.strictEqual(dynamicDefaults.workers, 6, 'Workers should reset to optimal detected architecture count');

console.log('✔ Feature 2 test_config_panel.js passed successfully.');
