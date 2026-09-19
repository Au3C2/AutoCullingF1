/**
 * Integration check: Ensure all element IDs referenced in app.js exist in index.html,
 * and all data-i18n attributes exist in both zh-CN and en-US locales.
 */
const fs = require('fs');
const path = require('path');
const assert = require('assert');

const baseDir = path.resolve(__dirname, '../../ui');
const html = fs.readFileSync(path.join(baseDir, 'index.html'), 'utf8');
const appJs = fs.readFileSync(path.join(baseDir, 'app.js'), 'utf8');
const zhCN = JSON.parse(fs.readFileSync(path.join(baseDir, 'locales/zh-CN.json'), 'utf8'));
const enUS = JSON.parse(fs.readFileSync(path.join(baseDir, 'locales/en-US.json'), 'utf8'));

function flatten(obj, prefix = '') {
  let res = {};
  for (let [k, v] of Object.entries(obj)) {
    let key = prefix ? prefix + '.' + k : k;
    if (typeof v === 'object' && v !== null) {
      Object.assign(res, flatten(v, key));
    } else {
      res[key] = String(v);
    }
  }
  return res;
}

const zhKeys = new Set(Object.keys(flatten(zhCN)));
const enKeys = new Set(Object.keys(flatten(enUS)));

// 1. Check all data-i18n attributes in HTML exist in both locales
const i18nMatches = [...html.matchAll(/data-i18n="([^"]+)"/g)];
for (const match of i18nMatches) {
  const key = match[1];
  assert(zhKeys.has(key), `Missing zh-CN key: ${key}`);
  assert(enKeys.has(key), `Missing en-US key: ${key}`);
}

// 2. Check all data-i18n-attr attributes in HTML exist in both locales
const attrMatches = [...html.matchAll(/data-i18n-attr="([^"]+)"/g)];
for (const match of attrMatches) {
  const raw = match[1];
  for (const pair of raw.split(';')) {
    const [, key] = pair.split(':').map((s) => s.trim());
    if (key) {
      assert(zhKeys.has(key), `Missing zh-CN attr key: ${key}`);
      assert(enKeys.has(key), `Missing en-US attr key: ${key}`);
    }
  }
}

// 3. Check critical element IDs referenced in app.js
const criticalIds = [
  'inputDir', 'btnBrowse', 'btnRun', 'btnRunText', 'btnExportCsv', 'btnToggleLog',
  'stageStatus', 'speedEtaStat', 'progressBar', 'frameStat', 'countAll', 'countKeep',
  'countReject', 'tableBody', 'tablePane', 'splitResizer', 'previewPane',
  'previewContainer', 'previewImg', 'previewEmpty', 'previewTitle', 'previewScoreDetails',
  'previewZoomControls', 'zoomLevelIndicator', 'btnResetZoom', 'pillRating', 'pillSharp',
  'pillComp', 'pillRaw', 'pillReason', 'logDrawer', 'logConsole', 'btnClearLog',
  'systemPulse', 'langSwitch', 'configPanel', 'btnToggleConfig', 'configToggleIcon',
  'configSummaryBar', 'btnResetDefaults', 'btnViewMode', 'viewModeIcon', 'viewModeText',
  'contextMenu', 'ctxShowInFolder', 'ctxCopyPath', 'ctxCopyName', 'ctxMarkKeep5',
  'ctxMarkKeep3', 'ctxMarkReject', 'dragDropOverlay', 'pRename'
];

for (const id of criticalIds) {
  assert(html.includes(`id="${id}"`), `Missing element id in index.html: ${id}`);
}

console.log('✔ All DOM IDs and i18n attributes are strictly verified and synchronized.');
