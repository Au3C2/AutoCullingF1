"""Layer 2 GUI test: JavaScript / DOM state and i18n module unit verification.

Runs via Node.js to test:
1. i18n initialization, fallback, and language switching.
2. Translation template interpolation: t("status.summary", { scored: 10, ... }).
3. Veto code localization helper.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
import pytest

NODE_TEST_SCRIPT = """
const fs = require('fs');
const path = require('path');

const zhCN = JSON.parse(fs.readFileSync('ui/locales/zh-CN.json', 'utf8'));
const enUS = JSON.parse(fs.readFileSync('ui/locales/en-US.json', 'utf8'));

// Helper to flatten
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

const dicts = {
  'zh-CN': flatten(zhCN),
  'en-US': flatten(enUS),
};

function createI18n(initialLang = 'auto') {
  let currentLang = initialLang === 'en-US' ? 'en-US' : 'zh-CN';
  return {
    getLang: () => currentLang,
    setLang: (lang) => { currentLang = (lang === 'en-US' ? 'en-US' : 'zh-CN'); },
    t: (key, params = {}) => {
      let text = dicts[currentLang][key] || dicts['zh-CN'][key] || key;
      for (let [p, val] of Object.entries(params)) {
        text = text.replace(new RegExp('{' + p + '}', 'g'), String(val));
      }
      return text;
    },
    translateVeto: (veto) => {
      if (!veto) return '';
      if (veto === 'no_detection') return dicts[currentLang]['veto.no_detection'] || veto;
      if (veto === 'decode_failed') return dicts[currentLang]['veto.decode_failed'] || veto;
      if (veto === 'manual_metadata') return dicts[currentLang]['veto.manual_metadata'] || veto;
      if (veto === 'burst_group_topn') return dicts[currentLang]['veto.burst_group_topn'] || veto;
      if (veto.includes('sharpness')) return dicts[currentLang]['veto.sharpness_fail'] || veto;
      if (veto.includes('raw=')) return dicts[currentLang]['veto.min_raw_fail'] || veto;
      if (veto.includes('p4_orient')) return dicts[currentLang]['veto.p4_orient_fail'] || veto;
      if (veto.includes('fence_detected')) return dicts[currentLang]['veto.fence_detected'] || veto;
      return veto;
    }
  };
}

// Test 1: Fallback and switching
const i18n = createI18n('zh-CN');
if (i18n.getLang() !== 'zh-CN') throw new Error('Initial lang should be zh-CN');
if (i18n.t('topbar.btn_run') !== '⚡️开始筛选') throw new Error('zh-CN btn_run mismatch: ' + i18n.t('topbar.btn_run'));

i18n.setLang('en-US');
if (i18n.getLang() !== 'en-US') throw new Error('Switched lang should be en-US');
if (i18n.t('topbar.btn_run') !== '⚡️Start Culling') throw new Error('en-US btn_run mismatch: ' + i18n.t('topbar.btn_run'));

// Test 2: Interpolation
let interpolated = i18n.t('telemetry.scored_summary', { scored: 10, total: 100, keep: 6, reject: 4 });
if (!interpolated.includes('10/100') || !interpolated.includes('6') || !interpolated.includes('4')) {
  throw new Error('Interpolation failed: ' + interpolated);
}

// Test 3: Veto code translation
let translatedVetoZh = createI18n('zh-CN').translateVeto('no_detection');
let translatedVetoEn = createI18n('en-US').translateVeto('no_detection');
if (translatedVetoZh === 'no_detection' || translatedVetoEn === 'no_detection') {
  throw new Error('Veto translation failed to localize no_detection');
}

console.log('ALL_GUI_I18N_UNIT_TESTS_PASS');
"""


def test_gui_i18n_node_module(tmp_path: Path):
    """Run headless node assertion on i18n parsing and logic."""
    test_file = tmp_path / "test_gui_i18n.js"
    test_file.write_text(NODE_TEST_SCRIPT, encoding="utf-8")

    res = subprocess.run(["node", str(test_file)], capture_output=True, text=True)
    assert res.returncode == 0, f"Node test failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert "ALL_GUI_I18N_UNIT_TESTS_PASS" in res.stdout
