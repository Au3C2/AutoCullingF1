/**
 * Unit tests for the filename format-tag display (extension replaced by
 * HEIF/HEIC/JPG/RAW/XML tag chips in the photo list).
 */
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const appSrc = fs.readFileSync(
  path.join(__dirname, '..', '..', 'ui', 'app.js'),
  'utf8',
);

// Extract a `function name(...) { ... }` (or `const X = {...};`) block by
// brace matching — same technique as test_grouping_logic.js.
function extractBlock(declaration) {
  const start = appSrc.indexOf(declaration);
  assert.ok(start >= 0, `block not found in app.js: ${declaration}`);
  const open = appSrc.indexOf('{', start);
  let depth = 0;
  for (let i = open; i < appSrc.length; i++) {
    if (appSrc[i] === '{') depth++;
    else if (appSrc[i] === '}') {
      depth--;
      if (depth === 0) {
        const snippet = appSrc.slice(start, i + 1);
        return snippet.endsWith(';') ? snippet : `${snippet};`;
      }
    }
  }
  throw new Error(`unbalanced braces while extracting ${declaration}`);
}

const factory = new Function(
  `${extractBlock('const FORMAT_TAG_MAP')}\n`
  + `${extractBlock('const FORMAT_TAG_ORDER')}\n`
  + `${extractBlock('const RAW_FORMAT_TAG')}\n`
  + `${extractBlock('function formatTagFor')}\n`
  + `${extractBlock('function formatTagsForExts')}\n`
  + `${extractBlock('function stemOf')}\n`
  + 'return { FORMAT_TAG_MAP, FORMAT_TAG_ORDER, RAW_FORMAT_TAG, formatTagFor, formatTagsForExts, stemOf };',
);
const { formatTagFor, formatTagsForExts, stemOf, RAW_FORMAT_TAG } = factory();

console.log('Testing filename format-tag display...');

// Tag mapping mirrors the engine format sets (cull/loader.py RAW_EXTS/COOKED_EXTS)
assert.strictEqual(formatTagFor('DSC00827.heif'), 'HEIF');
assert.strictEqual(formatTagFor('IMG_0001.HIF'), 'HEIF', 'Sony HIF is displayed as HEIF');
assert.strictEqual(formatTagFor('IMG_0002.heic'), 'HEIC');
assert.strictEqual(formatTagFor('DSC00827.ARW'), 'RAW', 'Camera RAW families collapse to RAW');
assert.strictEqual(formatTagFor('D_20260315.nef'), 'RAW');
assert.strictEqual(formatTagFor('a.cr3'), 'RAW');
assert.strictEqual(formatTagFor('a.dng'), 'RAW');
assert.strictEqual(formatTagFor('photo.jpg'), 'JPG');
assert.strictEqual(formatTagFor('photo.jpeg'), 'JPG');
assert.strictEqual(formatTagFor('sidecar.xmp'), 'XML');
assert.strictEqual(formatTagFor('noext'), null, 'Extension-less names show no tag');
assert.strictEqual(RAW_FORMAT_TAG, 'RAW');

// Stem: extension stripped, dots inside the stem preserved
assert.strictEqual(stemOf('DSC00827.heif'), 'DSC00827');
assert.strictEqual(stemOf('IMG_2026.03.15.jpg'), 'IMG_2026.03.15');
assert.strictEqual(stemOf('noext'), 'noext');

// Row HTML contract: the rendered name cell carries stem + tag chip, and the
// full filename stays in the title attribute (hover) for disambiguation.
const stem = stemOf('DSC00827.ARW');
const tag = formatTagFor('DSC00827.ARW');
assert.strictEqual(stem, 'DSC00827');
assert.strictEqual(tag, RAW_FORMAT_TAG, 'RAW rows must use the highlighted tag variant');

// Multi-format tags: every on-disk format of a shot is shown, in the fixed
// order (cooked image formats → RAW → sidecars), deduplicated.
assert.deepStrictEqual(
  formatTagsForExts(['.heif', '.arw', '.xmp']),
  ['HEIF', 'RAW', 'XML'],
  'RAW+HEIF+sidecar shot shows all three tags in fixed order',
);
assert.deepStrictEqual(formatTagsForExts(['.heif']), ['HEIF'], 'cooked-only shot shows only its own tag');
assert.deepStrictEqual(formatTagsForExts(['.nef']), ['RAW'], 'pure RAW shot shows RAW');
assert.deepStrictEqual(
  formatTagsForExts(['.hif', '.heif']),
  ['HEIF'],
  'HIF and HEIF dedupe to one tag',
);
assert.deepStrictEqual(
  formatTagsForExts(['.txt', '.ini']),
  [],
  'unknown extensions are skipped, not rendered as junk tags',
);
assert.deepStrictEqual(formatTagsForExts(null), [], 'missing exts renders no chips');
assert.deepStrictEqual(formatTagsForExts(['arw']), ['RAW'], 'extensions with or without leading dot');

console.log('OK test_format_tag.js passed successfully.');
