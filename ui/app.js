/**
 * Auto-Culling Tauri GUI Application Logic (Taubyte / Tau Cyber Edition)
 * 
 * Handles:
 * - IPC Communication with Tauri Rust / Resident Python Engine
 * - Dynamic Configuration Persistence with localStorage
 * - Multi-Language Localization (i18n: zh-CN / en-US)
 * - Real-time Progress, Speed (张/秒) & ETA Calculations
 * - Virtual/Incremental Table Rendering, Sorting & Filtering
 * - Asynchronous Image Preview with Detection/Crop Overlays
 * - Draggable Splitter Divider with Ratio Persistence
 */

(function () {
  'use strict';

  // --- Multi-Language (i18n) Engine ---
  const I18N = {
    currentLang: 'zh-CN', // 'zh-CN' | 'en-US'
    preference: 'auto',   // 'auto' | 'zh-CN' | 'en-US'
    dicts: {
      'zh-CN': {},
      'en-US': {},
    },

    flatten(obj, prefix = '') {
      const res = {};
      for (const [k, v] of Object.entries(obj)) {
        const key = prefix ? `${prefix}.${k}` : k;
        if (typeof v === 'object' && v !== null && !Array.isArray(v)) {
          Object.assign(res, this.flatten(v, key));
        } else {
          res[key] = String(v);
        }
      }
      return res;
    },

    detectSystemLang() {
      const sysLang = (navigator.language || navigator.userLanguage || '').toLowerCase();
      return sysLang.startsWith('zh') ? 'zh-CN' : 'en-US';
    },

    async loadLocales() {
      try {
        const [zhRes, enRes] = await Promise.all([
          fetch('locales/zh-CN.json').then((r) => r.json()),
          fetch('locales/en-US.json').then((r) => r.json()),
        ]);
        this.dicts['zh-CN'] = this.flatten(zhRes);
        this.dicts['en-US'] = this.flatten(enRes);
      } catch (err) {
        console.error('[i18n] Failed to load locale files, using in-memory fallbacks', err);
      }
    },

    t(key, params = {}) {
      let text = this.dicts[this.currentLang]?.[key] || this.dicts['zh-CN']?.[key] || key;
      for (const [p, val] of Object.entries(params)) {
        text = text.replace(new RegExp(`\\{${p}\\}`, 'g'), String(val));
      }
      return text;
    },

    translateVeto(veto) {
      if (!veto) return '';
      const map = {
        no_detection: 'veto.no_detection',
        decode_failed: 'veto.decode_failed',
        manual_metadata: 'veto.manual_metadata',
        burst_group_topn: 'veto.burst_group_topn',
      };
      if (map[veto]) return this.t(map[veto]);
      if (veto.includes('sharpness')) return this.t('veto.sharpness_fail');
      if (veto.includes('raw=')) return this.t('veto.min_raw_fail');
      if (veto.includes('p4_orient')) return this.t('veto.p4_orient_fail');
      if (veto.includes('fence_detected')) return this.t('veto.fence_detected');
      return veto;
    },

    applyDOM() {
      // 1. Text elements
      document.querySelectorAll('[data-i18n]').forEach((el) => {
        const key = el.getAttribute('data-i18n');
        if (key) {
          el.textContent = this.t(key);
        }
      });

      // 2. Attributes
      document.querySelectorAll('[data-i18n-attr]').forEach((el) => {
        const raw = el.getAttribute('data-i18n-attr');
        if (!raw) return;
        raw.split(';').forEach((pair) => {
          const [attr, key] = pair.split(':').map((s) => s.trim());
          if (attr && key) {
            el.setAttribute(attr, this.t(key));
          }
        });
      });

      // 3. Document title
      document.title = this.t('app.title');
    },

    setLanguage(pref) {
      this.preference = pref;
      localStorage.setItem('ac-ui-lang', pref);
      if (pref === 'auto') {
        this.currentLang = this.detectSystemLang();
      } else {
        this.currentLang = pref === 'en-US' ? 'en-US' : 'zh-CN';
      }
      document.documentElement.lang = this.currentLang;
      this.updateSwitchUI();
      this.applyDOM();
    },

    updateSwitchUI() {
      const switcher = document.getElementById('langSwitch');
      if (switcher) {
        switcher.setAttribute('data-active', this.currentLang);
      }
    },

    async init() {
      await this.loadLocales();
      const saved = localStorage.getItem('ac-ui-lang') || 'auto';
      this.setLanguage(saved);
    },
  };

  // --- State Management ---
  const state = {
    inputDir: '',
    isRunning: false,
    photos: [],         // Array of photo records: { name, path, rating, sharp, comp, raw, veto, status }
    photoMap: new Map(),// full path -> photo record (basenames are not unique in recursive scans)
    filter: 'all',      // 'all' | 'keep' | 'reject'
    sortField: 'name',
    sortAsc: true,
    selectedPhoto: null,
    totalFiles: 0,
    scoredCount: 0,
    keepCount: 0,
    rejectCount: 0,
    failedCount: 0,
    startTime: 0,
    tableRatio: 0.5,
  };

  const $ = (id) => document.getElementById(id);

  // --- DOM Elements ---
  const els = {
    inputDir: $('inputDir'),
    btnBrowse: $('btnBrowse'),
    btnRun: $('btnRun'),
    btnRunText: $('btnRunText'),
    btnExportCsv: $('btnExportCsv'),
    btnToggleLog: $('btnToggleLog'),
    stageStatus: $('stageStatus'),
    speedEtaStat: $('speedEtaStat'),
    progressBar: $('progressBar'),
    frameStat: $('frameStat'),
    countAll: $('countAll'),
    countKeep: $('countKeep'),
    countReject: $('countReject'),
    tableBody: $('tableBody'),
    tablePane: $('tablePane'),
    splitResizer: $('splitResizer'),
    previewPane: $('previewPane'),
    previewImg: $('previewImg'),
    previewEmpty: $('previewEmpty'),
    previewTitle: $('previewTitle'),
    previewScoreDetails: $('previewScoreDetails'),
    pillRating: $('pillRating'),
    pillSharp: $('pillSharp'),
    pillComp: $('pillComp'),
    pillRaw: $('pillRaw'),
    pillReason: $('pillReason'),
    logDrawer: $('logDrawer'),
    logConsole: $('logConsole'),
    btnClearLog: $('btnClearLog'),
    systemPulse: $('systemPulse'),
    langSwitch: $('langSwitch'),
  };

  // --- Parameter Bindings & Persistence ---
  const PARAMS = [
    { id: 'pTopN', key: 'top_n', type: 'int', default: 11 },
    { id: 'pWorkers', key: 'workers', type: 'int', default: 4 },
    { id: 'pP4', key: 'p4_policy', type: 'string', default: 'never' },
    { id: 'pRecursive', key: 'recursive', type: 'bool', default: false },
    { id: 'pForce', key: 'force', type: 'bool', default: false },
    { id: 'pSharp', key: 'sharp_thresh', type: 'float', default: 0.05 },
    { id: 'pWSharp', key: 'w_sharp', type: 'float', default: 1.5 },
    { id: 'pWComp', key: 'w_comp', type: 'float', default: 2.5 },
    { id: 'pMinRaw', key: 'min_raw', type: 'float', default: 3.1 },
    { id: 'pConf', key: 'conf', type: 'float', default: 0.25 },
    { id: 'pScale', key: 'scale_width', type: 'int', default: 1280 },
    { id: 'pRfKey', key: 'rf_api_key', type: 'string', default: '' },
    { id: 'pAutocrop', key: 'autocrop', type: 'bool', default: true },
    { id: 'pDryRun', key: 'dry_run', type: 'bool', default: false },
  ];

  const SECRET_KEYS = new Set(['rf_api_key']);

  function loadSavedParams() {
    PARAMS.forEach((p) => {
      const el = $(p.id);
      if (!el) return;
      if (SECRET_KEYS.has(p.key)) {
        const legacy = localStorage.getItem(`ac-param-${p.key}`);
        localStorage.removeItem(`ac-param-${p.key}`);
        invokeTauri('secret_get', { key: p.key }).then((v) => {
          if (v) {
            el.value = v;
          } else if (legacy) {
            el.value = legacy;
            return invokeTauri('secret_set', { key: p.key, value: legacy });
          }
          return null;
        }).catch(() => {});
      } else {
        const val = localStorage.getItem(`ac-param-${p.key}`);
        if (val !== null) {
          if (p.type === 'bool') el.checked = val === 'true';
          else el.value = val;
        }
      }
      const persist = () => {
        const currentVal = p.type === 'bool' ? el.checked : el.value;
        if (SECRET_KEYS.has(p.key)) {
          invokeTauri('secret_set', { key: p.key, value: String(currentVal) })
            .then(() => appendLog(`[Secret] ${p.key} stored in the OS credential store`))
            .catch((e) => appendLog(`[Secret Error] ${p.key}: ${e}`));
          return;
        }
        localStorage.setItem(`ac-param-${p.key}`, currentVal);
      };
      if (p.type === 'bool' || el.tagName === 'SELECT') {
        el.addEventListener('change', persist);
      } else {
        el.addEventListener('input', persist);
        el.addEventListener('change', persist);
      }
    });

    const savedRatio = localStorage.getItem('ac-table-ratio');
    if (savedRatio) {
      state.tableRatio = parseFloat(savedRatio);
      els.tablePane.style.setProperty('--table-width', `${(state.tableRatio * 100).toFixed(1)}%`);
    }

    const savedDir = localStorage.getItem('ac-last-dir');
    if (savedDir) {
      els.inputDir.value = savedDir;
      state.inputDir = savedDir;
      els.btnRun.disabled = false;
      triggerScan(savedDir);
    }
  }

  function getEngineConfig() {
    const config = {};
    PARAMS.forEach((p) => {
      const el = $(p.id);
      if (!el) return;
      if (p.type === 'int') config[p.key] = parseInt(el.value, 10);
      else if (p.type === 'float') config[p.key] = parseFloat(el.value);
      else if (p.type === 'bool') config[p.key] = el.checked;
      else config[p.key] = el.value || null;
    });
    return config;
  }

  // --- Tauri IPC Wrapper ---
  async function invokeTauri(cmd, args = {}) {
    if (window.__TAURI__ && window.__TAURI__.core) {
      return await window.__TAURI__.core.invoke(cmd, args);
    }
    console.warn(`[Tauri] invoke "${cmd}" fallback mock`, args);
    return null;
  }

  async function listenTauri(eventName, handler) {
    if (window.__TAURI__ && window.__TAURI__.event) {
      return await window.__TAURI__.event.listen(eventName, handler);
    }
    return () => {};
  }

  // --- Directory Selection & Scan ---
  async function chooseFolder() {
    try {
      const selected = await invokeTauri('select_folder');
      if (selected) {
        els.inputDir.value = selected;
        state.inputDir = selected;
        localStorage.setItem('ac-last-dir', selected);
        els.btnRun.disabled = false;
        await triggerScan(selected);
      }
    } catch (err) {
      appendLog(`[Error] ${err}`);
    }
  }

  async function triggerScan(dirPath) {
    if (!dirPath) return;
    els.stageStatus.textContent = I18N.t('telemetry.scanning_dir');
    const recursive = $('pRecursive')?.checked || false;
    await invokeTauri('scan', { dir: dirPath, recursive });
  }

  // --- Start / Cancel Culling Run ---
  async function handleRunToggle() {
    if (state.isRunning) {
      els.stageStatus.textContent = I18N.t('telemetry.cancelling');
      await invokeTauri('cancel');
      return;
    }

    if (!state.inputDir) {
      alert(I18N.t('dialog.alert_select_dir'));
      return;
    }

    state.isRunning = true;
    state.startTime = performance.now();
    state.scoredCount = 0;
    state.keepCount = 0;
    state.rejectCount = 0;

    if (els.btnRunText) els.btnRunText.textContent = I18N.t('topbar.btn_cancel');
    els.btnRun.classList.remove('tau-btn-primary');
    els.btnRun.classList.add('tau-btn-cancel');
    els.progressBar.style.width = '0%';
    els.stageStatus.textContent = I18N.t('telemetry.starting_engine');
    els.speedEtaStat.innerHTML = `<span class="tau-stat-label">${I18N.t('telemetry.speed')}</span> <span class="tau-stat-val">CALCULATING...</span>`;

    const config = getEngineConfig();
    try {
      await invokeTauri('run', { dir: state.inputDir, config });
    } catch (err) {
      appendLog(`[Error] ${err}`);
      finishRun(I18N.t('telemetry.run_error'));
    }
  }

  function finishRun(statusText = null) {
    state.isRunning = false;
    if (els.btnRunText) els.btnRunText.textContent = I18N.t('topbar.btn_run');
    els.btnRun.classList.add('tau-btn-primary');
    els.btnRun.classList.remove('tau-btn-cancel');
    els.btnExportCsv.disabled = state.photos.length === 0;
    els.stageStatus.textContent = statusText || I18N.t('telemetry.completed');
  }

  // --- Real-time Metrics: Speed (img/s) & ETA ---
  function updateSpeedAndEta() {
    if (!state.isRunning || state.scoredCount <= 0) return;
    const elapsedSec = (performance.now() - state.startTime) / 1000;
    if (elapsedSec <= 0.1) return;

    const speed = state.scoredCount / elapsedSec;
    const speedText = speed.toFixed(1);

    const remainingPhotos = Math.max(0, state.totalFiles - state.scoredCount);
    let etaText = '--';
    if (speed > 0 && remainingPhotos > 0) {
      const remainingSec = Math.round(remainingPhotos / speed);
      const min = Math.floor(remainingSec / 60);
      const sec = remainingSec % 60;
      etaText = min > 0 ? `${min}m ${sec}s` : `${sec}s`;
    } else if (remainingPhotos === 0) {
      etaText = 'DONE';
    }

    els.speedEtaStat.innerHTML = `
      <span class="tau-stat-label">${I18N.t('telemetry.speed')}</span> <span class="tau-stat-val">${speedText} img/s</span>
      <span class="tau-stat-sep">·</span>
      <span class="tau-stat-label">${I18N.t('telemetry.eta')}</span> <span class="tau-stat-val">${etaText}</span>
    `;

    const failText = state.failedCount > 0 ? I18N.t('telemetry.failed_part', { failed: state.failedCount }) : '';
    els.frameStat.textContent = I18N.t('telemetry.scored_summary', {
      scored: state.scoredCount,
      total: state.totalFiles,
      keep: state.keepCount,
      reject: state.rejectCount,
      failed: failText,
    });
  }

  // --- Event Handlers (Engine Stream) ---
  function setupEventListeners() {
    // 1. Directory Scanned
    listenTauri('scanned', ({ payload }) => {
      const paths = Array.isArray(payload.paths)
        ? payload.paths
        : Object.values(payload.paths || {});
      const count = payload.count || paths.length;
      state.totalFiles = count;
      state.photos = [];
      state.photoMap.clear();

      for (const p of paths) {
        const name = String(p).split(/[\\/]/).pop();
        const item = {
          name,
          path: String(p),
          rating: 0,
          sharp: 0,
          comp: 0,
          raw: 0,
          veto: '',
          status: 'pending',
        };
        state.photos.push(item);
        state.photoMap.set(item.path, item);
      }

      state.scoredCount = 0;
      state.keepCount = 0;
      state.rejectCount = 0;
      state.failedCount = 0;

      els.stageStatus.textContent = I18N.t('telemetry.photos_discovered', { count });
      els.frameStat.textContent = I18N.t('telemetry.photos_pending', { count });
      els.countAll.textContent = count;
      els.countKeep.textContent = '0';
      els.countReject.textContent = '0';
      els.progressBar.style.width = '0%';
      state.selectedPhoto = null;
      els.previewImg.style.display = 'none';
      els.previewImg.removeAttribute('src');
      els.previewEmpty.style.display = 'flex';
      els.previewTitle.textContent = I18N.t('preview.title');
      els.previewScoreDetails.style.display = 'none';
      renderTable();
    });

    // 2. Stage updates
    listenTauri('stage', ({ payload }) => {
      const msg = payload.message || payload.msg || '...';
      const pct = (payload.progress ?? payload.pct ?? 0) * 100;
      els.stageStatus.textContent = msg;
      if (!state.isRunning) return;
      if (pct > 0 && pct < 90) {
        els.progressBar.style.width = `${Math.max(pct, parseFloat(els.progressBar.style.width || 0))}%`;
      }
    });

    // 3. Scored Frame Event
    listenTauri('frame', ({ payload }) => {
      const item = state.photoMap.get(payload.path || payload.name);
      if (!item) return;

      item.rating = payload.rating;
      item.sharp = payload.sharp;
      item.comp = payload.comp;
      item.raw = payload.raw;
      item.veto = payload.veto;
      item.status = payload.status;

      if (payload.status === 'topn_final') {
        updateTableRow(item);
        return;
      }
      if (payload.status === 'decode_failed') {
        state.failedCount++;
        updateTableRow(item);
        return;
      }

      state.scoredCount++;
      if (payload.rating > 0) state.keepCount++;
      else state.rejectCount++;

      els.countKeep.textContent = state.keepCount;
      els.countReject.textContent = state.rejectCount;

      if (state.totalFiles > 0) {
        const progressPct = 10 + (state.scoredCount / state.totalFiles) * 85;
        els.progressBar.style.width = `${Math.min(95, progressPct).toFixed(1)}%`;
      }

      updateSpeedAndEta();
      updateTableRow(item);
    });

    // 4. Run Done Event
    listenTauri('done', ({ payload }) => {
      els.progressBar.style.width = '100%';
      const ips = payload.total > 0 && payload.elapsed > 0 ? (payload.total / payload.elapsed).toFixed(1) : '--';
      els.speedEtaStat.innerHTML = `
        <span class="tau-stat-label">AVG:</span> <span class="tau-stat-val">${ips} img/s</span>
        <span class="tau-stat-sep">·</span>
        <span class="tau-stat-label">TIME:</span> <span class="tau-stat-val">${(payload.elapsed || 0).toFixed(1)}s</span>
      `;
      const failedText = payload.failed ? I18N.t('telemetry.failed_part', { failed: payload.failed }) : '';
      finishRun(I18N.t('telemetry.completed_summary', {
        keep: payload.keep,
        reject: payload.reject,
        failed: failedText,
      }));
    });

    // 5. Cancelled Event
    listenTauri('cancelled', () => {
      finishRun(I18N.t('telemetry.cancelled'));
    });

    // 6. Log Events
    listenTauri('log', ({ payload }) => {
      appendLog(payload.line || JSON.stringify(payload));
    });

    // 7. Engine lifecycle errors
    listenTauri('engine-error', ({ payload }) => {
      appendLog(`[Engine Error] ${payload && payload.message ? payload.message : JSON.stringify(payload)}`);
      if (!state.isRunning) {
        els.stageStatus.textContent = I18N.t('telemetry.run_error');
      }
    });

    // 8. Engine run/scan errors
    listenTauri('error', ({ payload }) => {
      const msg = payload && payload.message ? payload.message : JSON.stringify(payload);
      appendLog(`[Error] ${msg}`);
      if (state.isRunning) {
        finishRun(`${I18N.t('telemetry.run_error')}: ${msg}`);
      } else {
        els.stageStatus.textContent = `${I18N.t('telemetry.run_error')}: ${msg}`;
      }
    });

    listenTauri('scan_error', ({ payload }) => {
      const msg = payload && payload.message ? payload.message : JSON.stringify(payload);
      appendLog(`[Scan Error] ${msg}`);
      els.stageStatus.textContent = I18N.t('telemetry.scan_error', { err: msg });
    });

    // 9. CSV export confirmation
    listenTauri('export_done', ({ payload }) => {
      appendLog(`[Export] scores.csv written to ${payload.path}`);
      els.stageStatus.textContent = I18N.t('telemetry.exported_to', { path: payload.path });
    });
  }

  function appendLog(line) {
    if (!els.logConsole) return;
    els.logConsole.textContent += `${line}\n`;
    els.logConsole.scrollTop = els.logConsole.scrollHeight;
  }

  // --- Table Rendering & In-Place Updates ---
  function renderTable() {
    const filtered = getFilteredAndSortedPhotos();
    if (filtered.length === 0) {
      els.tableBody.innerHTML = `
        <tr class="tau-empty-row">
          <td colspan="7">
            <div class="tau-empty-state">
              <span class="tau-empty-icon">📂</span>
              <p>${state.photos.length === 0 ? I18N.t('table.empty_select_prompt') : I18N.t('table.empty_filter')}</p>
            </div>
          </td>
        </tr>
      `;
      return;
    }

    const html = filtered.map((item) => buildRowHtml(item)).join('');
    els.tableBody.innerHTML = html;
  }

  function rowIdFor(item) {
    return `row-${item.path.replace(/[^a-zA-Z0-9_-]/g, '_')}`;
  }

  function esc(value) {
    return String(value).replace(/[&<>"']/g, (c) => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));
  }

  function buildRowHtml(item) {
    const isSelected = state.selectedPhoto && state.selectedPhoto.path === item.path;
    const scored = item.status !== 'pending' && item.status !== 'decode_failed';
    const num = (v, digits) => (scored && Number.isFinite(v) ? v.toFixed(digits) : '—');
    const ratingDisplay = item.status === 'pending' || item.status === 'decode_failed'
      ? '<span style="color:#475569;">—</span>'
      : item.rating > 0
        ? `<span class="tau-stars">${'★'.repeat(item.rating)}</span>`
        : `<span class="tau-reject-tag">${I18N.t('table.tag_reject')}</span>`;

    const translatedVeto = I18N.translateVeto(item.veto);
    const reasonDisplay = item.veto
      ? `<span class="tau-veto-desc" title="${esc(item.veto)}">${esc(translatedVeto)}</span>`
      : item.rating > 0
        ? `<span class="tau-pass-tag">${I18N.t('table.tag_passed')}</span>`
        : '—';

    const statusDisplay = item.status === 'pending'
      ? `<span style="color:#64748b;">${I18N.t('status.queued')}</span>`
      : (item.status === 'decode_failed'
        ? `<span style="color:#f87171;">${I18N.t('status.failed')}</span>`
        : (item.status === 'scored' || item.status === 'topn_final'
          ? `<span style="color:#00e5ff;">${I18N.t('status.scored')}</span>`
          : item.status));

    return `
      <tr id="${rowIdFor(item)}" data-path="${esc(item.path)}" class="${isSelected ? 'selected' : ''}">
        <td title="${esc(item.name)}" style="font-family: var(--tau-font-mono); font-weight: 500;">${esc(item.name)}</td>
        <td class="tau-th-num">${ratingDisplay}</td>
        <td class="tau-th-num" style="font-family: var(--tau-font-mono);">${num(item.sharp, 3)}</td>
        <td class="tau-th-num" style="font-family: var(--tau-font-mono);">${num(item.comp, 3)}</td>
        <td class="tau-th-num" style="font-family: var(--tau-font-mono); font-weight: 600;">${num(item.raw, 2)}</td>
        <td>${reasonDisplay}</td>
        <td class="tau-th-center" style="font-family: var(--tau-font-mono); font-size: 10px;">${statusDisplay}</td>
      </tr>
    `;
  }

  function updateTableRow(item) {
    const row = document.getElementById(rowIdFor(item));
    if (!row) {
      renderTable();
      return;
    }

    if (state.filter === 'keep' && item.rating <= 0) {
      row.style.display = 'none';
      return;
    }
    if (state.filter === 'reject' && item.rating > 0) {
      row.style.display = 'none';
      return;
    }
    row.style.display = '';

    const newHtml = buildRowHtml(item);
    const temp = document.createElement('tbody');
    temp.innerHTML = newHtml;
    const newRow = temp.firstElementChild;
    row.replaceWith(newRow);
    newRow.classList.add('flash');

    if (state.selectedPhoto && state.selectedPhoto.path === item.path) {
      selectPhoto(item);
    }
  }

  function getFilteredAndSortedPhotos() {
    let list = state.photos.slice();
    if (state.filter === 'keep') {
      list = list.filter((p) => p.rating > 0);
    } else if (state.filter === 'reject') {
      list = list.filter((p) => p.rating === -1 || (p.status !== 'pending' && p.rating <= 0));
    }

    list.sort((a, b) => {
      let valA = a[state.sortField];
      let valB = b[state.sortField];
      if (typeof valA === 'string') return state.sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
      valA = valA || 0;
      valB = valB || 0;
      return state.sortAsc ? valA - valB : valB - valA;
    });

    return list;
  }

  // --- Photo Selection & Thumbnail Preview ---
  async function selectPhoto(item) {
    if (!item) return;
    state.selectedPhoto = item;
    document.querySelectorAll('#photoTable tbody tr').forEach((r) => r.classList.remove('selected'));
    const row = document.getElementById(rowIdFor(item));
    if (row) row.classList.add('selected');

    els.previewTitle.textContent = item.name;
    els.previewScoreDetails.style.display = 'flex';
    const pillScored = item.status !== 'pending' && item.status !== 'decode_failed';
    const pillNum = (v, digits) => (pillScored && Number.isFinite(v) ? v.toFixed(digits) : '-');
    els.pillRating.textContent = `RATING: ${item.rating > 0 ? `${item.rating}★` : (item.rating === -1 ? I18N.t('table.tag_reject') : '-')}`;
    els.pillSharp.textContent = `SHARP: ${pillNum(item.sharp, 3)}`;
    els.pillComp.textContent = `COMP: ${pillNum(item.comp, 3)}`;
    els.pillRaw.textContent = `RAW: ${pillNum(item.raw, 2)}`;
    const reasonText = item.veto ? I18N.translateVeto(item.veto) : (item.rating > 0 ? I18N.t('table.tag_passed') : I18N.t('status.queued'));
    els.pillReason.textContent = `REASON: ${reasonText}`;

    const requestedPath = item.path;
    try {
      const res = await invokeTauri('preview', { path: requestedPath, size: 640 });
      if (state.selectedPhoto && state.selectedPhoto.path === requestedPath) {
        if (res && res.data) {
          const src = res.data.startsWith('data:') ? res.data : `data:image/png;base64,${res.data}`;
          els.previewImg.src = src;
          els.previewImg.style.display = 'block';
          els.previewEmpty.style.display = 'none';
        } else {
          els.previewImg.style.display = 'none';
          els.previewEmpty.style.display = 'flex';
          els.previewEmpty.querySelector('.tau-empty-title').textContent = I18N.t('preview.error_load_title');
          els.previewEmpty.querySelector('.tau-empty-desc').textContent = item.name;
        }
      }
    } catch (err) {
      if (state.selectedPhoto && state.selectedPhoto.path === requestedPath) {
        appendLog(`[Preview Error] ${err}`);
        els.previewImg.style.display = 'none';
        els.previewEmpty.style.display = 'flex';
        els.previewEmpty.querySelector('.tau-empty-title').textContent = I18N.t('preview.error_fail_title');
        els.previewEmpty.querySelector('.tau-empty-desc').textContent = `${err}`;
      }
    }
  }

  // --- Draggable Splitter Divider ---
  function initSplitter() {
    let isDragging = false;

    els.splitResizer.addEventListener('mousedown', (e) => {
      isDragging = true;
      els.splitResizer.classList.add('dragging');
      document.body.style.cursor = 'col-resize';
      e.preventDefault();
    });

    window.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      const containerWidth = $('workspace').offsetWidth;
      const minW = 280;
      const maxW = containerWidth - minW;
      const newW = Math.max(minW, Math.min(maxW, e.clientX));
      const ratio = newW / containerWidth;

      state.tableRatio = ratio;
      els.tablePane.style.setProperty('--table-width', `${(ratio * 100).toFixed(1)}%`);
    });

    window.addEventListener('mouseup', () => {
      if (isDragging) {
        isDragging = false;
        els.splitResizer.classList.remove('dragging');
        document.body.style.cursor = '';
        localStorage.setItem('ac-table-ratio', state.tableRatio.toFixed(4));
      }
    });
  }

  // --- UI Event Handlers ---
  async function initUI() {
    await I18N.init();

    if (els.langSwitch) {
      const toggleLang = (target) => {
        const next = target || (I18N.currentLang === 'zh-CN' ? 'en-US' : 'zh-CN');
        I18N.setLanguage(next);
        renderTable();
        if (state.selectedPhoto) selectPhoto(state.selectedPhoto);
        updateSpeedAndEta();
      };

      els.langSwitch.addEventListener('click', (e) => {
        const opt = e.target.closest('.tau-lang-opt');
        if (opt && opt.dataset.lang) {
          toggleLang(opt.dataset.lang);
        } else {
          toggleLang();
        }
      });

      els.langSwitch.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          toggleLang();
        }
      });
    }

    els.btnBrowse.addEventListener('click', chooseFolder);
    els.btnRun.addEventListener('click', handleRunToggle);

    els.inputDir.addEventListener('change', () => {
      const p = els.inputDir.value.trim();
      if (p) {
        state.inputDir = p;
        localStorage.setItem('ac-last-dir', p);
        els.btnRun.disabled = false;
        triggerScan(p);
      }
    });

    // Filter Buttons
    document.querySelectorAll('.tau-tab').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tau-tab').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        state.filter = btn.getAttribute('data-filter');
        renderTable();
      });
    });

    // Table Header Sorting
    document.querySelectorAll('#photoTable th.sortable').forEach((th) => {
      th.addEventListener('click', () => {
        const field = th.getAttribute('data-sort');
        if (state.sortField === field) {
          state.sortAsc = !state.sortAsc;
        } else {
          state.sortField = field;
          state.sortAsc = true;
        }
        renderTable();
      });
    });

    // Row Click Delegation for Thumbnail Selection
    els.tableBody.addEventListener('click', (e) => {
      const row = e.target.closest('tr');
      if (!row || row.classList.contains('tau-empty-row')) return;
      const item = state.photoMap.get(row.getAttribute('data-path'));
      if (item) selectPhoto(item);
    });

    // Export CSV
    els.btnExportCsv.addEventListener('click', async () => {
      if (state.photos.length === 0) return;
      try {
        await invokeTauri('export_csv', { dir: state.inputDir });
        els.stageStatus.textContent = I18N.t('telemetry.exporting_csv');
      } catch (err) {
        alert(I18N.t('dialog.export_failed', { err }));
      }
    });

    // Log Drawer Toggle
    els.btnToggleLog.addEventListener('click', () => {
      const isHidden = els.logDrawer.style.display === 'none';
      els.logDrawer.style.display = isHidden ? 'flex' : 'none';
    });

    els.btnClearLog.addEventListener('click', () => {
      els.logConsole.textContent = '';
    });

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'o') {
        e.preventDefault();
        chooseFolder();
      }
    });

    initSplitter();
    setupEventListeners();
    loadSavedParams();
  }

  window.addEventListener('DOMContentLoaded', initUI);
})();
