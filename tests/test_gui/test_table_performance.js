/**
 * Unit tests for Virtual / Chunked Table Slicing & Anti-Race (Feature 4)
 */
const assert = require('assert');

// Chunking helper
function getVisibleSlice(totalItems, currentChunkCount, chunkSize = 100) {
  const count = Math.min(totalItems.length, currentChunkCount * chunkSize);
  return totalItems.slice(0, count);
}

// Anti-race preview tracker
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

// --- Test Cases ---
console.log('Testing Feature 4: Table Chunking & Anti-Race...');

// 1. Chunking
const largeList = Array.from({ length: 500 }, (_, i) => ({ id: i, name: `photo_${i}.jpg` }));
let chunkCount = 1;
let slice = getVisibleSlice(largeList, chunkCount, 100);
assert.strictEqual(slice.length, 100);

chunkCount++;
slice = getVisibleSlice(largeList, chunkCount, 100);
assert.strictEqual(slice.length, 200);

// Cap at max length
chunkCount = 10;
slice = getVisibleSlice(largeList, chunkCount, 100);
assert.strictEqual(slice.length, 500);

// 2. Anti-Race
const tracker = new PreviewSequenceTracker();
const req1 = tracker.nextRequest('/photo1.jpg'); // seq 1
const req2 = tracker.nextRequest('/photo2.jpg'); // seq 2

// req1 returns AFTER req2
assert.strictEqual(tracker.isCurrent(req1.seq), false, 'Late req1 should be discarded');
assert.strictEqual(tracker.isCurrent(req2.seq), true, 'req2 is latest');

console.log('✔ Feature 4 test_table_performance.js passed successfully.');
