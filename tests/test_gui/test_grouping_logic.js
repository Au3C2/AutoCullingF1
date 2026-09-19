/**
 * Unit tests for Burst Grouping & Single Shot Interleaving Logic (Feature 6).
 *
 * The clustering functions are extracted verbatim from ui/app.js so the
 * mirror cannot drift from production code. Includes a regression suite for
 * rename invariance: checking "rename" and starting a run renames files to
 * IMG_YYYYMMDD_HHMMSS_mmm form and re-renders the table — the grouping MUST
 * be identical before and after.
 */
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const appSrc = fs.readFileSync(
  path.join(__dirname, '..', '..', 'ui', 'app.js'),
  'utf8',
);

// Extract a top-level `function name(...) { ... }` by brace matching. All
// braces inside regex literals and template strings in these functions are
// balanced, so naive depth counting is safe here.
function extractFunction(name) {
  const start = appSrc.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `function ${name} not found in app.js`);
  const open = appSrc.indexOf('{', start);
  let depth = 0;
  for (let i = open; i < appSrc.length; i++) {
    if (appSrc[i] === '{') depth++;
    else if (appSrc[i] === '}') {
      depth--;
      if (depth === 0) return appSrc.slice(start, i + 1);
    }
  }
  throw new Error(`unbalanced braces while extracting ${name}`);
}

const factory = new Function(
  `${extractFunction('parsePhotoTimeOrSeq')}\n`
  + `${extractFunction('formatClusterTimeRange')}\n`
  + `${extractFunction('buildBurstEntry')}\n`
  + `${extractFunction('computeBurstClusters')}\n`
  + 'return { parsePhotoTimeOrSeq, formatClusterTimeRange, buildBurstEntry, computeBurstClusters };',
);
const { parsePhotoTimeOrSeq, computeBurstClusters } = factory();

console.log('Testing Feature 6: Burst Grouping & Single Interleaving...');

// --- Helper: turn clusters into comparable shape (stable original index) ---
// Photos get an `idx` before renaming; clusters are compared by membership
// of these stable ids, not by filenames (which rename legitimately changes).
function withIdx(photos) {
  return photos.map((p, i) => ({ ...p, idx: i }));
}

function clusterIds(photos) {
  return computeBurstClusters(photos).map((entry) => (entry.type === 'single'
    ? [entry.photo.idx]
    : entry.items.map((p) => p.idx)));
}

// --- Case 1: Empty input ---
assert.deepStrictEqual(computeBurstClusters([]), [], 'Empty input should return empty array');

// --- Case 2: Interleaved bursts and singles (time-based, engine-aligned 2.0s) ---
const mockPhotos = [
  { name: 'IMG_0001.JPG', path: '/a/IMG_0001.JPG', timestamp: 1000, raw: 3.5, rating: 3 },
  { name: 'IMG_0002.JPG', path: '/a/IMG_0002.JPG', timestamp: 1500, raw: 4.2, rating: 5 }, // burst 1 (2 items)
  { name: 'IMG_0010.JPG', path: '/a/IMG_0010.JPG', timestamp: 5000, raw: 2.1, rating: -1 }, // single (gap 3.5s > 2.0s)
  { name: 'IMG_0020.JPG', path: '/a/IMG_0020.JPG', timestamp: 8000, raw: 4.0, rating: 4 },
  { name: 'IMG_0021.JPG', path: '/a/IMG_0021.JPG', timestamp: 8300, raw: 3.8, rating: 3 },
  { name: 'IMG_0022.JPG', path: '/a/IMG_0022.JPG', timestamp: 8600, raw: 1.5, rating: -1, veto: 'burst_group_topn' }, // burst 2 (3 items)
];

const result = computeBurstClusters(mockPhotos);
assert.strictEqual(result.length, 3, 'Should produce 3 top-level items: burst, single, burst');

assert.strictEqual(result[0].type, 'burst_header');
assert.strictEqual(result[0].count, 2);
assert.strictEqual(result[0].keepCount, 2);
assert.strictEqual(result[0].winnerPath, '/a/IMG_0002.JPG', 'Winner should be highest raw score');

assert.strictEqual(result[1].type, 'single');
assert.strictEqual(result[1].photo.name, 'IMG_0010.JPG');

assert.strictEqual(result[2].type, 'burst_header');
assert.strictEqual(result[2].count, 3);
assert.strictEqual(result[2].keepCount, 2);
assert.strictEqual(result[2].rejectCount, 1);
assert.strictEqual(result[2].winnerPath, '/a/IMG_0020.JPG');

// --- Case 3: Realistic NEF / Camera timestamp formatted filenames (IMG_YYYYMMDD_HHMMSS_mmm) ---
const t0 = Date.UTC(2026, 2, 15, 16, 41, 2, 480);
const t1 = Date.UTC(2026, 2, 15, 16, 41, 2, 540);
const t2 = Date.UTC(2026, 2, 15, 16, 41, 2, 600);
const t3 = Date.UTC(2026, 2, 15, 16, 41, 33, 610);
const t4 = Date.UTC(2026, 2, 15, 16, 41, 33, 680);
const nefPhotos = [
  { name: 'IMG_20260315_164102_480.nef', path: '/nef/IMG_20260315_164102_480.nef', timestamp: t0, raw: 3.5, rating: 3 },
  { name: 'IMG_20260315_164102_540.nef', path: '/nef/IMG_20260315_164102_540.nef', timestamp: t1, raw: 4.1, rating: 5 },
  { name: 'IMG_20260315_164102_600.nef', path: '/nef/IMG_20260315_164102_600.nef', timestamp: t2, raw: 3.8, rating: 3 },
  // Gap of 31 seconds -> next burst
  { name: 'IMG_20260315_164133_610.nef', path: '/nef/IMG_20260315_164133_610.nef', timestamp: t3, raw: 2.2, rating: -1 },
  { name: 'IMG_20260315_164133_680.nef', path: '/nef/IMG_20260315_164133_680.nef', timestamp: t4, raw: 2.5, rating: -1 },
];

const nefResult = computeBurstClusters(nefPhotos);
assert.strictEqual(nefResult.length, 2, 'Should cluster into 2 burst groups');
assert.strictEqual(nefResult[0].type, 'burst_header');
assert.strictEqual(nefResult[0].count, 3, 'First burst should have 3 photos');
assert.strictEqual(nefResult[1].type, 'burst_header');
assert.strictEqual(nefResult[1].count, 2, 'Second burst should have 2 photos');

// --- Case 4: Sony ARW with EXIF timestamps — engine-aligned gap semantics ---
// Burst gap of 3.5s between DSC00880 and DSC00886: the old "sequential number
// within 30s" heuristic merged them; the engine (group_bursts, gap 2.0s)
// splits them and the GUI now mirrors that.
const arwPhotos = [
  { name: 'DSC00879.ARW', path: '/test_arw/DSC00879.ARW', timestamp: 1742520800100, timeStr: '09:33:20.100', raw: 3.5, rating: 3 },
  { name: 'DSC00880.ARW', path: '/test_arw/DSC00880.ARW', timestamp: 1742520800350, timeStr: '09:33:20.350', raw: 4.2, rating: 5 },
  { name: 'DSC00886.ARW', path: '/test_arw/DSC00886.ARW', timestamp: 1742520900000, timeStr: '09:35:00.000', raw: 2.1, rating: -1 },
  { name: 'DSC00887.ARW', path: '/test_arw/DSC00887.ARW', timestamp: 1742520900300, timeStr: '09:35:00.300', raw: 3.9, rating: 4 },
  { name: 'DSC00888.ARW', path: '/test_arw/DSC00888.ARW', timestamp: 1742520900600, timeStr: '09:35:00.600', raw: 3.8, rating: 3 },
  { name: 'DSC00958.ARW', path: '/test_arw/DSC00958.ARW', timestamp: 1742521500000, timeStr: '09:45:00.000', raw: 1.5, rating: -1 }, // single
];

const arwResult = computeBurstClusters(arwPhotos);
assert.strictEqual(arwResult.length, 3, 'ARW should split into 2 bursts + 1 single');
assert.strictEqual(arwResult[0].type, 'burst_header');
assert.strictEqual(arwResult[0].count, 2);
assert.strictEqual(arwResult[1].type, 'burst_header');
assert.strictEqual(arwResult[1].count, 3);
assert.strictEqual(arwResult[2].type, 'single');

// --- Case 5 (REGRESSION): rename must NOT change grouping ---
// Replicates the GUI flow: scan delivers items with EXIF timestamps under
// original names; the engine then renames files to IMG_YYYYMMDD_HHMMSS_mmm
// (collision counter suffixes appended when sub-second data is missing) and
// the 'renamed' event re-renders the table. The handler anchors each item's
// pre-rename filename in `originalName`, so sequence-based grouping keeps
// resolving against the original names.
function sonyBurstSet() {
  const base = Date.UTC(2024, 2, 16, 13, 30, 0, 0);
  // Burst @100ms intervals, 6s gap, burst again, 6s gap, single, 48s gap, single.
  // Consecutive capture numbers + gaps <= 30s merge via Rule B, exactly like
  // the pre-fix preview grouping the user expects.
  const offsets = [0, 100, 200, 300, 400, 500, 6000, 6100, 6200, 6300, 6400, 6500, 12000, 60000];
  return offsets.map((off, i) => ({
    name: `DSC045${String(i + 1).padStart(2, '0')}.ARW`,
    path: `/card/DCIM/DSC045${String(i + 1).padStart(2, '0')}.ARW`,
    timestamp: base + off,
    timeStr: null,
  }));
}

// Renamer replication: IMG_YYYYMMDD_HHMMSS_mmm.ext, `_N` on collision.
// Mirrors the app.js 'renamed' handler: item.originalName = old name.
function renameAll(photos, { withSubSec }) {
  const used = new Set();
  const pad = (n, l = 2) => String(n).padStart(l, '0');
  return photos.map((p) => {
    const ts = withSubSec ? p.timestamp : Math.floor(p.timestamp / 1000) * 1000;
    const d = new Date(ts);
    const base = `IMG_${d.getUTCFullYear()}${pad(d.getUTCMonth() + 1)}${pad(d.getUTCDate())}_${pad(d.getUTCHours())}${pad(d.getUTCMinutes())}${pad(d.getUTCSeconds())}_${pad(d.getUTCMilliseconds(), 3)}.arw`;
    const stem = base.slice(0, -4);
    let name = base;
    let c = 1;
    while (used.has(name.toLowerCase())) {
      name = `${stem}_${c}.arw`;
      c += 1;
    }
    used.add(name.toLowerCase());
    return { ...p, name, path: `/card/DCIM/${name}`, originalName: p.name };
  });
}

// 5a. Full sub-second data: unique ms names
const before5a = clusterIds(withIdx(sonyBurstSet()));
const after5a = clusterIds(withIdx(renameAll(sonyBurstSet(), { withSubSec: true })));
assert.deepStrictEqual(after5a, before5a, 'Grouping must be identical after rename (subsec names)');

// 5b. No sub-second data: collision counter suffixes _1.._N
const before5b = clusterIds(withIdx(sonyBurstSet()));
const after5b = clusterIds(withIdx(renameAll(sonyBurstSet(), { withSubSec: false })));
assert.deepStrictEqual(after5b, before5b, 'Grouping must be identical after rename (collision-suffixed names)');

// 5c. Mixed: some files skipped by the renamer (no EXIF time) keep old names.
//     They still carry their scan timestamps, so grouping is unchanged.
const mixed = sonyBurstSet().map((p, i) => (i % 3 === 0 ? p : renameAll([p], { withSubSec: true })[0]));
assert.deepStrictEqual(clusterIds(withIdx(mixed)), before5a, 'Grouping must survive partial renames');

// 5d. The expected pre-fix grouping itself: consecutive capture numbers with
//     gaps <= 30s merge (Rule B), the 48s gap splits.
assert.deepStrictEqual(before5a, [
  Array.from({ length: 13 }, (_, i) => i),
  [13],
], 'Rule B must merge consecutive capture numbers within 30s (pre-fix behavior)');

// --- Case 6: seq fallback ONLY when no timing info exists ---
const noTimePhotos = [
  { name: 'DSC00100.JPG', path: '/x/DSC00100.JPG', timestamp: null, timeStr: null },
  { name: 'DSC00101.JPG', path: '/x/DSC00101.JPG', timestamp: null, timeStr: null },
  { name: 'DSC00105.JPG', path: '/x/DSC00105.JPG', timestamp: null, timeStr: null },
];
const noTimeResult = computeBurstClusters(noTimePhotos);
assert.strictEqual(noTimeResult.length, 2, 'Seq-adjacent pair merges, seq gap splits');
assert.strictEqual(noTimeResult[0].type, 'burst_header');
assert.strictEqual(noTimeResult[0].count, 2);
assert.strictEqual(noTimeResult[1].type, 'single');

// --- Case 7: parsePhotoTimeOrSeq must not treat timestamp tails as seq ---
const renamedMeta = parsePhotoTimeOrSeq({ name: 'IMG_20240316_213037_123.jpg', timestamp: null });
assert.ok(renamedMeta.time !== null, 'Timestamp should parse from renamed filename');
assert.strictEqual(renamedMeta.seq, null, 'Trailing ms field must not be parsed as capture sequence');
const collidedMeta = parsePhotoTimeOrSeq({ name: 'IMG_20240316_213037_123_1.jpg', timestamp: null });
assert.ok(collidedMeta.time !== null, 'Collision-suffixed name should still parse its timestamp');
assert.strictEqual(collidedMeta.seq, null, 'Collision counter must not be parsed as capture sequence');
const seqMeta = parsePhotoTimeOrSeq({ name: 'DSC00886.ARW', timestamp: null });
assert.strictEqual(seqMeta.seq, 886, 'Plain camera names keep sequence parsing');
// After a rename, sequence numbers must resolve against the anchored original name
const anchoredMeta = parsePhotoTimeOrSeq({
  name: 'IMG_20240316_213037_123.jpg',
  originalName: 'DSC04506.ARW',
  timestamp: 1710595837123,
});
assert.strictEqual(anchoredMeta.seq, 4506, 'Sequence must come from the anchored original filename');
assert.ok(anchoredMeta.time !== null, 'EXIF timestamp stays authoritative after rename');

// --- Case 8: backend-authoritative grouping (scan carries engine burst ids) ---
// When every photo carries the engine's burst_group, the GUI must group by
// those ids directly (single pass, no client-side re-clustering).
const backendPhotos = [
  { name: 'DSC001.ARW', path: '/b/DSC001.ARW', timestamp: 1000, burstGroup: 'burst_0001', raw: 3.5, rating: 3 },
  { name: 'DSC002.ARW', path: '/b/DSC002.ARW', timestamp: 1100, burstGroup: 'burst_0001', raw: 4.2, rating: 5 },
  { name: 'DSC003.ARW', path: '/b/DSC003.ARW', timestamp: 120000, burstGroup: 'burst_0002', raw: 2.0, rating: -1 },
  { name: 'DSC004.ARW', path: '/b/DSC004.ARW', timestamp: 240000, burstGroup: 'burst_0003', raw: 3.0, rating: 4 },
];
const backendResult = computeBurstClusters(backendPhotos);
assert.strictEqual(backendResult.length, 3, 'Backend groups: burst(2), single, single');
assert.strictEqual(backendResult[0].type, 'burst_header');
assert.strictEqual(backendResult[0].groupId, 'burst_0001', 'Group id must come from the engine');
assert.strictEqual(backendResult[0].count, 2);
assert.strictEqual(backendResult[0].winnerPath, '/b/DSC002.ARW');
assert.strictEqual(backendResult[1].type, 'single');
assert.strictEqual(backendResult[2].type, 'single');

// Backend groups survive a rename untouched: the engine ids derive from EXIF,
// so renamed paths/names must not change the cluster layout.
const renamedBackend = backendPhotos.map((p) => ({
  ...p,
  name: `IMG_20240316_2130${String(p.timestamp).padStart(5, '0')}_123.jpg`,
  originalName: p.name,
}));
assert.deepStrictEqual(
  clusterIds(withIdx(renamedBackend)),
  clusterIds(withIdx(backendPhotos)),
  'Backend grouping must be rename-invariant',
);

// --- Case 9: BEST winner only among SCORED frames ---
// Regression: after a reopen+rescan every frame is pending with raw=0; the
// old loop (maxScore=-1, score=0>-1) pinned the BEST badge on the FIRST
// frame of every group. Pending/decode-failed frames must never win, and an
// all-pending group must have no winner at all.
const pendingPhotos = [
  { name: 'DSC001.ARW', path: '/p/DSC001.ARW', timestamp: 1000, burstGroup: 'burst_0001', raw: 0, rating: 0, status: 'pending' },
  { name: 'DSC002.ARW', path: '/p/DSC002.ARW', timestamp: 1100, burstGroup: 'burst_0001', raw: 0, rating: 0, status: 'pending' },
];
const pendingResult = computeBurstClusters(pendingPhotos);
assert.strictEqual(pendingResult[0].type, 'burst_header');
assert.strictEqual(pendingResult[0].winnerPath, null, 'All-pending group must have no BEST winner');

const mixedScorePhotos = [
  { name: 'DSC001.ARW', path: '/m/DSC001.ARW', timestamp: 1000, burstGroup: 'burst_0001', raw: 0, rating: 0, status: 'pending' },
  { name: 'DSC002.ARW', path: '/m/DSC002.ARW', timestamp: 1100, burstGroup: 'burst_0001', raw: 4.2, rating: 5 },
  { name: 'DSC003.ARW', path: '/m/DSC003.ARW', timestamp: 1200, burstGroup: 'burst_0001', raw: 3.9, rating: 3 },
  { name: 'DSC004.ARW', path: '/m/DSC004.ARW', timestamp: 1300, burstGroup: 'burst_0001', raw: 9.9, rating: 0, status: 'decode_failed' },
];
const mixedResult = computeBurstClusters(mixedScorePhotos);
assert.strictEqual(mixedResult[0].winnerPath, '/m/DSC002.ARW',
  'Winner must be the best SCORED frame; pending and decode_failed frames are excluded');

console.log('OK Feature 6 test_grouping_logic.js passed successfully.');
