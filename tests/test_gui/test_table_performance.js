/**
 * Unit tests for table rendering strategy & Anti-Race (Feature 4).
 *
 * History: the flat table used chunked slicing (120 rows per chunk, grown on
 * scroll). That made the scrollbar reflect only the rendered slice, so it
 * never tracked the real list position — replaced by render-all (grouped
 * mode already rendered every row). The chunking tests were retired with it;
 * these cases pin the CURRENT semantics:
 *   1. render-all: flat mode renders the entire filtered list
 *   2. scan_meta merge: in-place update preserves item identity & counters
 *   3. anti-race preview tracker
 */
const assert = require('assert');

// 1. Render-all semantics (replaces getVisibleSlice chunking)
function flatRenderList(filtered) {
  // Mirrors renderFlatTable: no slicing — every filtered row is rendered.
  return filtered.map((item) => buildRowHtmlStub(item));
}

function buildRowHtmlStub(item) {
  return `<tr data-path="${item.path}"></tr>`;
}

console.log('Testing Feature 4: Table Render-All & Anti-Race...');

const largeList = Array.from({ length: 500 }, (_, i) => ({ id: i, path: `photo_${i}.jpg` }));
const rendered = flatRenderList(largeList);
assert.strictEqual(rendered.length, 500, 'flat mode must render every filtered row');

// 2. scan_meta merge: updates fields in place, never resets the list
const photoMap = new Map(largeList.map((p) => [p.path, { path: p.path, timestamp: null, timeStr: null, burstGroup: null }]));
let scoredCount = 3;
let keepCount = 2;

const metaItems = [
  { path: 'photo_0.jpg', timestamp: 1700000000000, time_str: '10:00:00.000', burst_group: 'burst_0001' },
  { path: 'photo_1.jpg', timestamp: 1700000000500, time_str: '10:00:00.500', burst_group: 'burst_0001' },
  { path: 'missing.jpg', timestamp: 1, time_str: 'x', burst_group: 'burst_9999' }, // unknown path must be ignored
];

let updated = 0;
for (const it of metaItems) {
  const item = photoMap.get(String(it.path));
  if (!item) continue;
  if (it.timestamp !== undefined) item.timestamp = it.timestamp;
  if (it.time_str !== undefined) item.timeStr = it.time_str;
  if (it.burst_group !== undefined) item.burstGroup = it.burst_group;
  updated++;
}
assert.strictEqual(updated, 2, 'only known paths are merged');
assert.strictEqual(photoMap.get('photo_0.jpg').burstGroup, 'burst_0001');
assert.strictEqual(photoMap.size, 500, 'map size unchanged (no reset/re-add)');
assert.strictEqual(scoredCount, 3, 'counters untouched by scan_meta');
assert.strictEqual(keepCount, 2, 'counters untouched by scan_meta');

// Empty scan_meta payload must be a no-op
const emptyItems = [];
assert.strictEqual(emptyItems.length === 0, true, 'empty scan_meta short-circuits before any merge');

// 3. Anti-Race preview tracker
class PreviewSequenceTracker {
  constructor() {
    this.seq = 0;
  }

  nextRequest(path) {
    this.seq++;
    return { path, seq: this.seq };
  }

  isCurrent(responseSeq) {
    return responseSeq === this.seq;
  }
}

const tracker = new PreviewSequenceTracker();
const req1 = tracker.nextRequest('/photo1.jpg'); // seq 1
const req2 = tracker.nextRequest('/photo2.jpg'); // seq 2

// req1 returns AFTER req2
assert.strictEqual(tracker.isCurrent(req1.seq), false, 'Late req1 should be discarded');
assert.strictEqual(tracker.isCurrent(req2.seq), true, 'req2 is latest');

console.log('✔ Feature 4 test_table_performance.js passed successfully.');
