/**
 * Automated guard: Statically extract all I18N.t(...) calls in ui/app.js
 * and assert that every single translation key exists in both zh-CN.json and en-US.json.
 */
const fs = require('fs');
const path = require('path');
const assert = require('assert');

const baseDir = path.resolve(__dirname, '../../ui');
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

// Regex to find I18N.t('xxx') or I18N.t("xxx")
const tMatches = [...appJs.matchAll(/I18N\.t\(\s*['"]([a-zA-Z0-9_.]+)['"]/g)];
assert(tMatches.length > 0, 'Should find I18N.t calls in app.js');

const missingInZh = [];
const missingInEn = [];

for (const match of tMatches) {
  const key = match[1];
  if (!zhKeys.has(key)) missingInZh.push(key);
  if (!enKeys.has(key)) missingInEn.push(key);
}

// Regex to find conditional ternary keys: e.g. I18N.t(isDry ? 'topbar.btn_run_dry' : 'topbar.btn_run')
const ternaryMatches = [...appJs.matchAll(/I18N\.t\([^)]*?\?\s*['"]([a-zA-Z0-9_.]+)['"]\s*:\s*['"]([a-zA-Z0-9_.]+)['"]/g)];
for (const match of ternaryMatches) {
  const k1 = match[1];
  const k2 = match[2];
  if (!zhKeys.has(k1)) missingInZh.push(k1);
  if (!enKeys.has(k1)) missingInEn.push(k1);
  if (!zhKeys.has(k2)) missingInZh.push(k2);
  if (!enKeys.has(k2)) missingInEn.push(k2);
}

assert.deepStrictEqual(missingInZh, [], `The following I18N.t keys in app.js are MISSING in zh-CN.json: ${missingInZh.join(', ')}`);
assert.deepStrictEqual(missingInEn, [], `The following I18N.t keys in app.js are MISSING in en-US.json: ${missingInEn.join(', ')}`);

// Specific assertion for the dry-run toggle button text
const dictZh = flatten(zhCN);
assert.strictEqual(dictZh['topbar.btn_run_dry'], '⚡️试运行');
const dictEn = flatten(enUS);
assert.strictEqual(dictEn['topbar.btn_run_dry'], '⚡️Dry-Run');

console.log(`✔ Verified all ${tMatches.length + ternaryMatches.length * 2} dynamic I18N.t() keys in app.js exist in both locales.`);
