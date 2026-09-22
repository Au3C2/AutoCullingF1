/**
 * Auto-Culling Tauri GUI Application Logic (Cyber Edition)
 * 
 * Features Supported:
 * 1. Keyboard Navigation Flow & Culling Hotkeys (Arrow, j/k, Space, 1-5, x/0)
 * 2. Collapsible & Compact Config Panel with Defaults Reset
 * 3. Interactive Pan & Zoom Preview Inspection
 * 4. High-Performance Virtual/Chunked Table Slicing & Anti-Race IPC
 * 5. Drag & Drop Folder Import (HTML5 + Tauri)
 * 6. Burst Group Aggregated View & Single Shot Interleaving
 * 7. Right-Click Context Menu & Local File System Integration
 */

(function () {
  'use strict';

  // --- Multi-Language (i18n) Engine ---
  const I18N = {
    currentLang: 'zh-CN',
    preference: 'auto',
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
        manual_reject: 'veto.manual_reject',
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
      document.querySelectorAll('[data-i18n]').forEach((el) => {
        const key = el.getAttribute('data-i18n');
        if (key) el.textContent = this.t(key);
      });

      document.querySelectorAll('[data-i18n-attr]').forEach((el) => {
        const raw = el.getAttribute('data-i18n-attr');
        if (!raw) return;
        raw.split(';').forEach((pair) => {
          const [attr, key] = pair.split(':').map((s) => s.trim());
          if (attr && key) el.setAttribute(attr, this.t(key));
        });
      });

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
      if (switcher) switcher.setAttribute('data-active', this.currentLang);
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
    photos: [],
    photoMap: new Map(),
    filter: 'all',          // 'all' | 'keep' | 'reject'
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

    // Feature 6: View mode & burst grouping
    viewMode: localStorage.getItem('ac-view-mode') || 'grouped', // 'grouped' | 'flat'
    // True between the fast 'scanned' event and the async 'scan_meta' event
    // (engine EXIF + authoritative burst groups). While pending, grouped view
    // renders FLAT: the client-side fallback clustering would otherwise show
    // wrong groups (filename-less cameras fall back to identical file mtimes)
    // that visibly re-shuffle seconds later when scan_meta lands.
    scanPending: false,
    collapsedGroups: new Set(),

    // Feature 2: Collapsible config
    configCollapsed: localStorage.getItem('ac-config-collapsed') === 'true',

    // Feature 3: Pan & Zoom
    zoom: {
      level: 1.0,
      panX: 0,
      panY: 0,
      locked: localStorage.getItem('ac-zoom-locked') === 'true',
      isPanning: false,
      startX: 0,
      startY: 0,
    },

    // Save triggers & persistence state
    dirtyPhotos: new Set(),
    isSaving: false,
    exitPending: false,

    // Feature 4: Anti-race preview tracker
    previewSeq: 0,
    previewLoadedPath: null,

    // Feature 7: Context menu
    contextPhoto: null,
  };

  const $ = (id) => document.getElementById(id);

  // --- DOM Elements ---
  const els = {
    inputDir: $('inputDir'),
    btnBrowse: $('btnBrowse'),
    btnRun: $('btnRun'),
    btnRunText: $('btnRunText'),
    btnSaveMetadata: $('btnSaveMetadata'),
    btnSaveMetadataText: $('btnSaveMetadataText'),
    saveBadge: $('saveBadge'),
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
    tableContainer: document.querySelector('.tau-table-container'),
    splitResizer: $('splitResizer'),
    previewPane: $('previewPane'),
    previewContainer: $('previewContainer'),
    previewImg: $('previewImg'),
    previewEmpty: $('previewEmpty'),
    previewTitle: $('previewTitle'),
    previewScoreDetails: $('previewScoreDetails'),
    previewZoomControls: $('previewZoomControls'),
    zoomLevelIndicator: $('zoomLevelIndicator'),
    btnResetZoom: $('btnResetZoom'),
    btnLockZoom: $('btnLockZoom'),
    lockZoomIcon: $('lockZoomIcon'),
    savingModal: $('savingModal'),
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
    
    // New Feature elements
    configPanel: $('configPanel'),
    btnToggleConfig: $('btnToggleConfig'),
    configToggleIcon: $('configToggleIcon'),
    configSummaryBar: $('configSummaryBar'),
    btnResetDefaults: $('btnResetDefaults'),
    btnViewMode: $('btnViewMode'),
    viewModeIcon: $('viewModeIcon'),
    viewModeText: $('viewModeText'),
    contextMenu: $('contextMenu'),
    ctxShowInFolder: $('ctxShowInFolder'),
    ctxCopyPath: $('ctxCopyPath'),
    ctxCopyName: $('ctxCopyName'),
    ctxMarkKeep5: $('ctxMarkKeep5'),
    ctxMarkKeep3: $('ctxMarkKeep3'),
    ctxMarkReject: $('ctxMarkReject'),
    dragDropOverlay: $('dragDropOverlay'),
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
    { id: 'pRename', key: 'rename', type: 'bool', default: false },
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
            .then(() => appendLog(`[Secret] ${p.key} stored in OS credential store`))
            .catch((e) => appendLog(`[Secret Error] ${p.key}: ${e}`));
          return;
        }
        localStorage.setItem(`ac-param-${p.key}`, currentVal);
        updateConfigSummary();
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

    applyConfigCollapsedState();
    updateConfigSummary();
    updateViewModeUI();
    updateRunButtonState();

    const dryCheckbox = $('pDryRun');
    if (dryCheckbox) {
      dryCheckbox.addEventListener('change', () => {
        updateRunButtonState();
        if (dryCheckbox.checked) {
          alert(I18N.t('dialog.dry_run_warn'));
        }
      });
    }

    const renameCheckbox = $('pRename');
    if (renameCheckbox) {
      renameCheckbox.addEventListener('change', () => {
        if (renameCheckbox.checked) {
          alert(I18N.t('dialog.rename_warn'));
        }
      });
    }

    // CPU Architecture-aware optimal workers probe
    invokeTauri('get_optimal_workers').then((cores) => {
      if (cores && cores > 0) {
        state.optimalWorkers = cores;
        const workerParam = PARAMS.find((p) => p.key === 'workers');
        if (workerParam) workerParam.default = cores;

        const elWorkers = $('pWorkers');
        const customSaved = localStorage.getItem('ac-param-workers');
        if (elWorkers && (!customSaved || customSaved === '4')) {
          elWorkers.value = cores;
          localStorage.setItem('ac-param-workers', cores);
          updateConfigSummary();
          appendLog(`[CPU Architecture] Decoder workers optimized for platform: ${cores} threads`);
        }
      }
    }).catch(() => {});
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

  // --- Feature 2: Collapsible Config & Defaults Reset ---
  function updateConfigSummary() {
    if (!els.configSummaryBar) return;
    const cfg = getEngineConfig();
    const parts = [
      `Top-N: ${cfg.top_n}`,
      `Workers: ${cfg.workers}`,
      `Sharp: ${cfg.sharp_thresh}`,
      `YOLO: ${cfg.conf}`,
    ];
    if (cfg.p4_policy && cfg.p4_policy !== 'never') {
      parts.push(`P4: ${cfg.p4_policy}`);
    }
    els.configSummaryBar.textContent = parts.join(' · ');
  }

  function applyConfigCollapsedState() {
    if (!els.configPanel) return;
    if (state.configCollapsed) {
      els.configPanel.classList.add('collapsed');
      if (els.configToggleIcon) els.configToggleIcon.textContent = '▼';
    } else {
      els.configPanel.classList.remove('collapsed');
      if (els.configToggleIcon) els.configToggleIcon.textContent = '▲';
    }
  }

  function toggleConfigPanel() {
    state.configCollapsed = !state.configCollapsed;
    localStorage.setItem('ac-config-collapsed', String(state.configCollapsed));
    applyConfigCollapsedState();
  }

  function resetToDefaults() {
    if (state.isRunning) return;
    PARAMS.forEach((p) => {
      const el = $(p.id);
      if (!el) return;
      const defVal = (p.key === 'workers' && state.optimalWorkers) ? state.optimalWorkers : p.default;
      if (p.type === 'bool') el.checked = defVal;
      else el.value = defVal;

      localStorage.removeItem(`ac-param-${p.key}`);
      if (SECRET_KEYS.has(p.key)) {
        invokeTauri('secret_set', { key: p.key, value: '' }).catch(() => {});
      }
    });
    updateConfigSummary();
    updateRunButtonState();
    appendLog('[Config] Parameters restored to default values');
  }

  // --- Feature 6: View Mode Switch ---
  function updateViewModeUI() {
    if (!els.btnViewMode) return;
    const isGrouped = state.viewMode === 'grouped';
    if (els.viewModeIcon) els.viewModeIcon.textContent = isGrouped ? '⊞' : '≡';
    if (els.viewModeText) {
      els.viewModeText.setAttribute('data-i18n', isGrouped ? 'view.mode_grouped' : 'view.mode_flat');
      els.viewModeText.textContent = I18N.t(isGrouped ? 'view.mode_grouped' : 'view.mode_flat');
    }
  }

  function toggleViewMode() {
    state.viewMode = state.viewMode === 'grouped' ? 'flat' : 'grouped';
    localStorage.setItem('ac-view-mode', state.viewMode);
    updateViewModeUI();
    renderTable();
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
        handleDirectoryLoaded(selected);
      }
    } catch (err) {
      appendLog(`[Error] ${err}`);
    }
  }

  async function handleDirectoryLoaded(dirPath) {
    if (!dirPath) return;
    els.inputDir.value = dirPath;
    state.inputDir = dirPath;
    localStorage.setItem('ac-last-dir', dirPath);
    els.btnRun.disabled = false;
    await triggerScan(dirPath);
  }

  async function triggerScan(dirPath) {
    if (!dirPath) return;
    els.stageStatus.textContent = I18N.t('telemetry.scanning_dir');
    const recursive = $('pRecursive')?.checked || false;
    await invokeTauri('scan', { dir: dirPath, recursive });
  }

  function updateRunButtonState() {
    if (state.isRunning) {
      if (els.btnRunText) {
        els.btnRunText.setAttribute('data-i18n', 'topbar.btn_cancel');
        els.btnRunText.textContent = I18N.t('topbar.btn_cancel');
      }
      return;
    }
    const isDry = $('pDryRun')?.checked || false;
    if (els.btnRunText) {
      const key = isDry ? 'topbar.btn_run_dry' : 'topbar.btn_run';
      els.btnRunText.setAttribute('data-i18n', key);
      els.btnRunText.textContent = I18N.t(key);
    }
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

    // Auto-compact config panel during run to maximize workspace
    if (!state.configCollapsed) {
      state.configCollapsed = true;
      applyConfigCollapsedState();
    }

    updateRunButtonState();
    els.btnRun.classList.remove('tau-btn-primary');
    els.btnRun.classList.add('tau-btn-cancel');
    els.progressBar.style.width = '0%';

    const config = getEngineConfig();
    const isDry = Boolean(config.dry_run);
    els.stageStatus.textContent = isDry
      ? I18N.t('telemetry.dry_run_running')
      : I18N.t('telemetry.starting_engine');

    els.speedEtaStat.innerHTML = `<span class="tau-stat-label">${I18N.t('telemetry.speed')}</span> <span class="tau-stat-val">CALCULATING...</span>`;

    try {
      await invokeTauri('run', { dir: state.inputDir, config });
    } catch (err) {
      appendLog(`[Error] ${err}`);
      finishRun(I18N.t('telemetry.run_error'));
    }
  }

  function finishRun(statusText = null) {
    state.isRunning = false;
    updateRunButtonState();
    els.btnRun.classList.add('tau-btn-primary');
    els.btnRun.classList.remove('tau-btn-cancel');
    els.btnExportCsv.disabled = state.photos.length === 0;
    // Final consistency pass after the mid-run in-place refreshes: winner
    // badges settle on final scores, auto-collapse rules apply, headers
    // show final stats.
    renderTable();

    const isDry = $('pDryRun')?.checked || false;
    if (!statusText && isDry) {
      els.stageStatus.textContent = I18N.t('telemetry.dry_run_completed');
    } else {
      els.stageStatus.textContent = statusText || I18N.t('telemetry.completed');
    }
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
    listenTauri('scanned', ({ payload }) => {
      const paths = Array.isArray(payload.paths)
        ? payload.paths
        : Object.values(payload.paths || {});
      const count = payload.count || paths.length;
      state.totalFiles = count;
      state.photos = [];
      state.photoMap.clear();
      state.collapsedGroups.clear();

      const itemsList = Array.isArray(payload.items) && payload.items.length > 0 ? payload.items : null;
      if (itemsList) {
        for (const it of itemsList) {
          const path = String(it.path);
          const name = it.name || path.split(/[\\/]/).pop();
          const item = {
            name,
            path,
            timestamp: it.timestamp || null,
            timeStr: it.time_str || null,
            burstGroup: it.burst_group || null,
            exts: Array.isArray(it.exts) ? it.exts : null,
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
      } else {
        for (const p of paths) {
          const name = String(p).split(/[\\/]/).pop();
          const item = {
            name,
            path: String(p),
            timestamp: null,
            timeStr: null,
            burstGroup: null,
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
      }

      state.scoredCount = 0;
      state.keepCount = 0;
      state.rejectCount = 0;
      state.failedCount = 0;
      state.scanPending = true;

      els.stageStatus.textContent = I18N.t('telemetry.photos_discovered', { count });
      els.frameStat.textContent = I18N.t('telemetry.photos_pending', { count });
      els.countAll.textContent = count;
      els.countKeep.textContent = '0';
      els.countReject.textContent = '0';
      els.progressBar.style.width = '0%';
      state.selectedPhoto = null;
      resetZoom();
      els.previewImg.style.display = 'none';
      els.previewImg.removeAttribute('src');
      state.previewLoadedPath = null;
      els.previewEmpty.style.display = 'flex';
      els.previewTitle.textContent = I18N.t('preview.title');
      els.previewScoreDetails.style.display = 'none';
      if (els.previewZoomControls) els.previewZoomControls.style.display = 'none';
      renderTable();
    });

    listenTauri('stage', ({ payload }) => {
      const msg = payload.message || payload.msg || '...';
      const pct = (payload.progress ?? payload.pct ?? 0) * 100;
      els.stageStatus.textContent = msg;
      if (!state.isRunning) return;
      if (pct > 0 && pct < 90) {
        els.progressBar.style.width = `${Math.max(pct, parseFloat(els.progressBar.style.width || 0))}%`;
      }
    });

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
        // The protocol strips the veto text on Top-N re-emits, but a Top-N
        // re-emit always ends rejected (select_best_n downgrades to -1) —
        // reconstruct the veto so the REASON column / Top-N chip render.
        if (payload.rating < 0) item.veto = 'burst_group_topn';
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

      // Trigger 1: culling run done -> all results automatically saved
      state.dirtyPhotos.clear();
      updateSaveButtonState();
      appendLog('[Cull] All frame ratings and crop coordinates synchronized.');
    });

    listenTauri('save_done', ({ payload }) => {
      appendLog(`[Engine] save_done confirmed: ${payload && payload.count !== undefined ? payload.count : 0} items`);
    });

    listenTauri('save_error', ({ payload }) => {
      appendLog(`[Engine Error] save_error: ${payload && payload.message ? payload.message : JSON.stringify(payload)}`);
    });

    // Trigger 4: intercept window close to safely flush pending metadata
    listenTauri('app-close-requested', async () => {
      if (state.dirtyPhotos.size === 0 && !state.isSaving) {
        await invokeTauri('exit_app');
        return;
      }
      state.exitPending = true;
      if (els.savingModal) els.savingModal.style.display = 'flex';
      appendLog('[Exit] Window close requested with pending metadata changes, flushing...');

      // 4-second safety timeout guard: prevent UI hanging forever if disk/engine stalls
      setTimeout(() => {
        invokeTauri('exit_app');
      }, 4000);

      flushSaveMetadata();
    });

    listenTauri('renamed', ({ payload }) => {
      const map = payload?.map || {};
      let changed = false;
      for (const [oldPath, newPath] of Object.entries(map)) {
        const item = state.photoMap.get(oldPath);
        if (item) {
          state.photoMap.delete(oldPath);
          // Anchor the pre-rename filename: grouping sequence numbers must
          // keep resolving against it so the cluster layout is stable.
          if (!item.originalName) item.originalName = item.name;
          item.path = newPath;
          item.name = newPath.split(/[\\/]/).pop();
          state.photoMap.set(newPath, item);
          changed = true;
          if (state.selectedPhoto && state.selectedPhoto.path === oldPath) {
            state.selectedPhoto = item;
          }
        }
      }
      if (changed) {
        appendLog(`[Rename] Renamed ${Object.keys(map).length} photo files to EXIF timestamp format`);
        renderTable();
        if (state.selectedPhoto) selectPhoto(state.selectedPhoto, false);
      }
    });

    listenTauri('cancelled', () => {
      finishRun(I18N.t('telemetry.cancelled'));
    });

    listenTauri('log', ({ payload }) => {
      appendLog(payload.line || JSON.stringify(payload));
    });

    listenTauri('engine-error', ({ payload }) => {
      appendLog(`[Engine Error] ${payload && payload.message ? payload.message : JSON.stringify(payload)}`);
      if (!state.isRunning) {
        els.stageStatus.textContent = I18N.t('telemetry.run_error');
      }
    });

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
      state.scanPending = false;
      appendLog(`[Scan Error] ${msg}`);
      els.stageStatus.textContent = I18N.t('telemetry.scan_error', { err: msg });
    });

    // Phase-2 scan result: engine EXIF timestamps + authoritative burst
    // groups, computed asynchronously AFTER 'scanned' so the photo list
    // renders immediately on drag-and-drop. Merged in place — selection,
    // counters and scroll position are untouched. A run in progress skips
    // the rebuild: rows stream in via 'frame' events and a full renderTable
    // would fight the per-frame in-place updates.
    listenTauri('scan_meta', ({ payload }) => {
      // Stale worker guard: a slow EXIF pass for a PREVIOUS directory must
      // not clear the current scan's pending flag or merge into its list.
      // Trailing separators are normalized: the engine echoes str(Path(dir))
      // which strips them, while a hand-typed input may keep them.
      const reqDir = payload && payload.dir ? String(payload.dir).replace(/[\\/]+$/, '') : '';
      const curDir = state.inputDir ? String(state.inputDir).replace(/[\\/]+$/, '') : '';
      if (reqDir && curDir && reqDir !== curDir) return;
      // The authoritative EXIF pass is over (success or not) — grouped view
      // may leave the flat hold and use engine groups / fallback clustering.
      state.scanPending = false;
      const items = Array.isArray(payload && payload.items) ? payload.items : [];
      if (items.length === 0) {
        renderTable();
        return;
      }
      let updated = 0;
      for (const it of items) {
        const item = state.photoMap.get(String(it.path));
        if (!item) continue;
        if (it.timestamp !== undefined) item.timestamp = it.timestamp;
        if (it.time_str !== undefined) item.timeStr = it.time_str;
        if (it.burst_group !== undefined) item.burstGroup = it.burst_group;
        if (Array.isArray(it.exts) && it.exts.length > 0) item.exts = it.exts;
        updated++;
      }
      if (updated === 0) {
        renderTable();
        return;
      }
      appendLog(`[Scan] EXIF metadata + burst groups updated (${updated} photos)`);
      if (state.isRunning) return;
      const container = els.tableContainer;
      const prevScrollTop = container ? container.scrollTop : 0;
      renderTable();
      if (container) container.scrollTop = prevScrollTop;
    });

    listenTauri('export_done', ({ payload }) => {
      appendLog(`[Export] scores.csv written to ${payload.path}`);
      els.stageStatus.textContent = I18N.t('telemetry.exported_to', { path: payload.path });
    });

    // Tauri 2 Drag & Drop Events (Native Window OS Hook)
    listenTauri('tauri://drag-enter', () => {
      if (els.dragDropOverlay) els.dragDropOverlay.style.display = 'flex';
    });

    listenTauri('tauri://drag-over', () => {
      if (els.dragDropOverlay) els.dragDropOverlay.style.display = 'flex';
    });

    listenTauri('tauri://drag-leave', () => {
      if (els.dragDropOverlay) els.dragDropOverlay.style.display = 'none';
    });

    listenTauri('tauri://file-drop-hover', () => {
      if (els.dragDropOverlay) els.dragDropOverlay.style.display = 'flex';
    });

    listenTauri('tauri://file-drop-cancelled', () => {
      if (els.dragDropOverlay) els.dragDropOverlay.style.display = 'none';
    });

    listenTauri('tauri://drag-drop', (event) => {
      if (els.dragDropOverlay) els.dragDropOverlay.style.display = 'none';
      const paths = event?.payload?.paths;
      if (Array.isArray(paths) && paths.length > 0) {
        handleDirectoryLoaded(paths[0]);
      }
    });

    listenTauri('tauri://file-drop', (event) => {
      if (els.dragDropOverlay) els.dragDropOverlay.style.display = 'none';
      const paths = event?.payload?.paths || event?.payload;
      if (Array.isArray(paths) && paths.length > 0) {
        handleDirectoryLoaded(paths[0]);
      }
    });
  }

  function appendLog(line) {
    if (!els.logConsole) return;
    els.logConsole.textContent += `${line}\n`;
    els.logConsole.scrollTop = els.logConsole.scrollHeight;
  }

  // --- Feature 6: Burst Group Analysis Algorithm ---
  function parsePhotoTimeOrSeq(photo) {
    let time = photo.timestamp || null;
    let timeStr = photo.timeStr || null;
    let seq = null;

    const name = photo.name || '';
    // Original filename captured before the EXIF-rename pass. The capture
    // sequence must keep coming from it after a rename, otherwise the
    // grouping of already-rendered results would change mid-run.
    const seqName = photo.originalName || name;

    // 1. If timestamp not present on photo object, try to parse from filename
    if (!time) {
      // YYYYMMDD_HHMMSS_mmm or YYYYMMDD-HHMMSS-mmm
      const mTimeMs = name.match(/(\d{4})[_-]?(\d{2})[_-]?(\d{2})[_-](\d{2})(\d{2})(\d{2})[_-](\d{1,3})/);
      if (mTimeMs) {
        const [, Y, M, D, h, min, s, ms] = mTimeMs;
        const parsedMs = parseInt(ms.padEnd(3, '0'), 10);
        time = Date.UTC(parseInt(Y, 10), parseInt(M, 10) - 1, parseInt(D, 10), parseInt(h, 10), parseInt(min, 10), parseInt(s, 10), parsedMs);
        timeStr = `${h}:${min}:${s}.${ms.padEnd(3, '0')}`;
      } else {
        // YYYYMMDD_HHMMSS
        const mTimeSec = name.match(/(\d{4})[_-]?(\d{2})[_-]?(\d{2})[_-](\d{2})(\d{2})(\d{2})/);
        if (mTimeSec) {
          const [, Y, M, D, h, min, s] = mTimeSec;
          time = Date.UTC(parseInt(Y, 10), parseInt(M, 10) - 1, parseInt(D, 10), parseInt(h, 10), parseInt(min, 10), parseInt(s, 10), 0);
          timeStr = `${h}:${min}:${s}`;
        }
      }
    }

    // 2. Sequential integer at tail (e.g. _DSC1001.JPG, DSC00886.ARW).
    //    Skipped when the name itself is a timestamp: in IMG_..._123.jpg the
    //    trailing number is the millisecond field (or a _1 collision counter),
    //    not a capture sequence, and treating it as seq corrupts grouping.
    const isTimestampName = /(\d{4})[_-]?(\d{2})[_-]?(\d{2})[_-](\d{2})(\d{2})(\d{2})/.test(seqName);
    if (!isTimestampName) {
      const mSeq = seqName.match(/(\d+)\.[^.]+$/);
      if (mSeq) {
        seq = parseInt(mSeq[1], 10);
      }
    }

    return {
      type: time ? 'time' : (seq !== null ? 'seq' : 'unknown'),
      val: time || seq,
      time,
      timeStr,
      seq,
    };
  }

  function buildBurstEntry(cluster, groupId) {
    let keep = 0;
    let reject = 0;
    let best = null;
    let maxScore = -1;

    for (const p of cluster) {
      if (p.rating > 0) keep++;
      else if (p.rating <= 0 && p.status !== 'pending') reject++;

      // Winner (BEST badge) is only meaningful among SCORED frames. Before
      // this guard, unscored frames (raw=0 > maxScore=-1) made the FIRST
      // frame of every group wear a BEST badge right after a rescan.
      const scored = p.status !== 'pending' && p.status !== 'decode_failed';
      const score = scored ? (p.raw || 0) : -1;
      if (score > maxScore) {
        maxScore = score;
        best = p;
      }
    }

    return {
      type: 'burst_header',
      groupId,
      count: cluster.length,
      keepCount: keep,
      rejectCount: reject,
      hasKeeps: keep > 0,
      winnerPath: best ? best.path : null,
      timeRange: formatClusterTimeRange(cluster),
      items: cluster,
    };
  }

  function computeBurstClusters(photos) {
    if (!photos || photos.length === 0) return [];

    // Backend-authoritative mode: the scan event carries each photo's burst
    // group computed by the engine's group_bursts(). Grouping is a single
    // O(n) pass over stable engine ids — no client-side re-implementation.
    // Partial coverage (e.g. 0-byte files the engine's EXIF pass skipped)
    // stays in backend mode: uncovered shots render as singles instead of
    // degrading the WHOLE list to client-side clustering.
    const backendGrouped = photos.filter((p) => p.burstGroup !== null && p.burstGroup !== undefined && p.burstGroup !== '');
    if (backendGrouped.length > 0) {
      const ordered = photos.slice().sort((a, b) => {
        const ta = a.timestamp || 0;
        const tb = b.timestamp || 0;
        if (ta !== tb) return ta - tb;
        return a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' });
      });

      const items = [];
      let current = [];
      let currentId = null;

      const flush = () => {
        if (current.length === 0) return;
        if (current.length === 1) {
          items.push({ type: 'single', photo: current[0] });
        } else {
          items.push(buildBurstEntry(current, currentId));
        }
        current = [];
      };

      for (const p of ordered) {
        if (p.burstGroup === null || p.burstGroup === undefined || p.burstGroup === '') {
          flush();
          items.push({ type: 'single', photo: p });
          continue;
        }
        if (current.length > 0 && p.burstGroup !== currentId) {
          flush();
        }
        if (current.length === 0) currentId = p.burstGroup;
        current.push(p);
      }
      flush();
      return items;
    }

    // Fallback: client-side clustering when backend groups are unavailable
    // (legacy engine payload / scan EXIF failure).

    // Chronological and sequential ordering
    const ordered = photos.slice().sort((a, b) => {
      const metaA = parsePhotoTimeOrSeq(a);
      const metaB = parsePhotoTimeOrSeq(b);
      if (metaA.val !== null && metaB.val !== null && metaA.type === metaB.type) {
        return metaA.val - metaB.val;
      }
      return a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' });
    });

    const clusters = [];
    let current = [];

    for (let i = 0; i < ordered.length; i++) {
      const p = ordered[i];
      const meta = parsePhotoTimeOrSeq(p);

      if (current.length === 0) {
        current.push({ p, meta });
      } else {
        const prev = current[current.length - 1];
        let same = false;

        // Rule A: Fast burst interval (time gap <= 2.5s, typical for high-speed sport burst)
        if (prev.meta.time && meta.time) {
          if (Math.abs(meta.time - prev.meta.time) <= 2500) {
            same = true;
          }
        }

        // Rule B: Sequential filename number difference is exactly 1 (e.g. DSC00886 -> DSC00887).
        // Guarded by a 30s time window when timestamps exist so it cannot merge
        // across distinct scenes; seq values are anchored to the original
        // (pre-rename) filename and therefore rename-invariant.
        if (!same && prev.meta.seq !== null && meta.seq !== null) {
          if (Math.abs(meta.seq - prev.meta.seq) === 1) {
            // Guard with reasonable time window if timestamps exist
            if (prev.meta.time && meta.time) {
              if (Math.abs(meta.time - prev.meta.time) <= 30000) {
                same = true;
              }
            } else {
              same = true;
            }
          }
        }

        if (same) {
          current.push({ p, meta });
        } else {
          clusters.push(current.map((x) => x.p));
          current = [{ p, meta }];
        }
      }
    }
    if (current.length > 0) clusters.push(current.map((x) => x.p));

    const items = [];
    let groupNum = 1;

    for (const cluster of clusters) {
      if (cluster.length === 1) {
        items.push({
          type: 'single',
          photo: cluster[0],
        });
      } else {
        items.push(buildBurstEntry(cluster, `burst_${groupNum++}`));
      }
    }

    return items;
  }

  function formatClusterTimeRange(cluster) {
    if (!cluster || cluster.length === 0) return '';
    const first = cluster[0];
    const last = cluster[cluster.length - 1];

    const firstMeta = parsePhotoTimeOrSeq(first);
    const lastMeta = parsePhotoTimeOrSeq(last);

    // If both have valid time information (via EXIF, filename, or file stat)
    if (firstMeta.time && lastMeta.time) {
      const diffSec = Math.max(0, (lastMeta.time - firstMeta.time) / 1000).toFixed(2);
      if (firstMeta.timeStr && lastMeta.timeStr) {
        return `${firstMeta.timeStr} ~ ${lastMeta.timeStr} (${diffSec}s)`;
      }
      const d1 = new Date(firstMeta.time);
      const d2 = new Date(lastMeta.time);
      const pad = (n, l = 2) => String(n).padStart(l, '0');
      const t1 = `${pad(d1.getHours())}:${pad(d1.getMinutes())}:${pad(d1.getSeconds())}.${pad(d1.getMilliseconds(), 3)}`;
      const t2 = `${pad(d2.getHours())}:${pad(d2.getMinutes())}:${pad(d2.getSeconds())}.${pad(d2.getMilliseconds(), 3)}`;
      return `${t1} ~ ${t2} (${diffSec}s)`;
    }

    // Fallback: If only sequential numbering is available
    if (firstMeta.seq !== null && lastMeta.seq !== null) {
      return `#${firstMeta.seq} ~ #${lastMeta.seq}`;
    }
    return '';
  }

  // --- Table Rendering (Virtual/Chunked) ---
  function getFilteredPhotos() {
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

  function getVisiblePhotosList() {
    const filtered = getFilteredPhotos();
    if (state.viewMode === 'flat' || state.scanPending) {
      return filtered;
    }

    // In grouped mode, flatten visible items according to collapse state
    const clusters = computeBurstClusters(filtered);
    const visible = [];

    for (const entry of clusters) {
      if (entry.type === 'single') {
        visible.push(entry.photo);
      } else if (entry.type === 'burst_header') {
        const isCollapsed = state.collapsedGroups.has(entry.groupId);
        if (!isCollapsed) {
          visible.push(...entry.items);
        }
      }
    }
    return visible;
  }

  function renderTable() {
    const filtered = getFilteredPhotos();
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

    if (state.viewMode === 'flat') {
      renderFlatTable(filtered);
    } else {
      renderGroupedTable(filtered);
    }
  }

  function renderFlatTable(filtered) {
    // Render ALL rows. The previous chunked slicing (120 rows per chunk,
    // grown on scroll) made the scrollbar reflect only the rendered slice,
    // so it did not track the real list position until the user hit the
    // bottom. Grouped mode already renders every row; flat mode now does
    // too — per-frame updates go through updateTableRowFlatInPlace, so the
    // full rebuild only happens on scan/filter/sort/language changes.
    els.tableBody.innerHTML = filtered.map((item) => buildRowHtml(item)).join('');
  }

  function renderGroupedTable(filtered) {
    // While the engine's EXIF pass is still running (scanPending), client-side
    // clustering would produce wrong, mtime-guessed groups that visibly
    // re-shuffle when scan_meta lands — hold the flat layout instead.
    if (state.scanPending) {
      renderFlatTable(filtered);
      return;
    }
    const clusters = computeBurstClusters(filtered);
    let html = '';

    for (const entry of clusters) {
      if (entry.type === 'single') {
        html += buildRowHtml(entry.photo, { isSingle: true });
      } else if (entry.type === 'burst_header') {
        // Auto-collapse rule: if user hasn't explicitly toggled it, auto-collapse groups with 0 keeps
        const userToggled = state.collapsedGroups.has(`_toggled_${entry.groupId}`);
        let isCollapsed = state.collapsedGroups.has(entry.groupId);
        if (!userToggled && !entry.hasKeeps && entry.rejectCount > 0) {
          isCollapsed = true;
          state.collapsedGroups.add(entry.groupId);
        }

        html += buildGroupHeaderHtml(entry, isCollapsed);

        if (!isCollapsed) {
          for (const item of entry.items) {
            const isWinner = entry.winnerPath === item.path;
            const isTopN = item.veto === 'burst_group_topn';
            html += buildRowHtml(item, { isBurstChild: true, isWinner, isTopN });
          }
        }
      }
    }

    els.tableBody.innerHTML = html;
  }

  function buildGroupHeaderHtml(group, isCollapsed) {
    return `
      <tr class="tau-group-header-row ${isCollapsed ? '' : 'expanded'}" data-group-id="${esc(group.groupId)}">
        <td colspan="7" class="tau-gh-cell">${buildGroupHeaderCellHtml(group, isCollapsed)}</td>
      </tr>
    `;
  }

  function buildGroupHeaderCellHtml(group, isCollapsed) {
    const keepPercent = group.count > 0 ? ((group.keepCount / group.count) * 100).toFixed(0) : 0;
    const rejectPercent = group.count > 0 ? ((group.rejectCount / group.count) * 100).toFixed(0) : 0;
    const tagText = group.hasKeeps ? I18N.t('burst.has_keeps') : I18N.t('burst.all_rejected');
    const tagClass = group.hasKeeps ? 'tau-chip-cyan' : 'tau-chip-muted';
    const timeDisplay = group.timeRange ? `<span class="tau-gh-time">${esc(group.timeRange)}</span>` : '';

    return `
      <div class="tau-gh-wrap">
        <span class="tau-gh-toggle">${isCollapsed ? '▶' : '▼'}</span>
        <span class="tau-gh-title">${I18N.t('burst.group_title', { id: group.groupId.replace('burst_', '') })}</span>
        ${timeDisplay}
        <span class="tau-gh-stats">${I18N.t('burst.group_stats', { count: group.count, keep: group.keepCount, reject: group.rejectCount })}</span>
        <div class="tau-micro-bar" title="Keep ${keepPercent}% / Discard ${rejectPercent}%">
          <div class="tau-bar-keep" style="width: ${keepPercent}%;"></div>
          <div class="tau-bar-reject" style="width: ${rejectPercent}%;"></div>
        </div>
        <span class="tau-chip ${tagClass}" style="font-size: 9px; padding: 0 4px;">${tagText}</span>
      </div>
    `;
  }

  function rowIdFor(item) {
    return `row-${item.path.replace(/[^a-zA-Z0-9_-]/g, '_')}`;
  }

  // Display format tag for the filename cell. Mirrors the engine's format
  // sets (cull/loader.py RAW_EXTS / COOKED_EXTS); camera RAW families
  // collapse into a single "RAW" tag so the list reads by format class, not
  // by brand-specific extension.
  const FORMAT_TAG_MAP = {
    arw: 'RAW', nef: 'RAW', cr2: 'RAW', cr3: 'RAW',
    orf: 'RAW', rw2: 'RAW', raf: 'RAW', dng: 'RAW',
    hif: 'HEIF', heif: 'HEIF', heic: 'HEIC',
    jpg: 'JPG', jpeg: 'JPG', png: 'PNG', tif: 'TIFF', tiff: 'TIFF',
    xmp: 'XML', xml: 'XML',
  };
  const RAW_FORMAT_TAG = 'RAW';

  function formatTagFor(name) {
    const m = /\.([a-zA-Z0-9]+)$/.exec(String(name));
    if (!m) return null;
    return FORMAT_TAG_MAP[m[1].toLowerCase()] || m[1].toUpperCase();
  }

  // Tags for every format present on disk for a shot (engine sends sibling
  // extensions, e.g. DSC00827 → [".arw", ".heif", ".xmp"]). Unknown
  // extensions are skipped; output follows the fixed display order:
  // cooked image formats, then RAW, then sidecars.
  const FORMAT_TAG_ORDER = ['HEIF', 'HEIC', 'JPG', 'PNG', 'TIFF', 'RAW', 'XML'];

  function formatTagsForExts(exts) {
    const tags = new Set();
    for (const e of exts || []) {
      const m = /\.?([a-zA-Z0-9]+)$/.exec(String(e));
      if (!m) continue;
      const tag = FORMAT_TAG_MAP[m[1].toLowerCase()];
      if (tag) tags.add(tag);
    }
    return [...tags].sort((a, b) => FORMAT_TAG_ORDER.indexOf(a) - FORMAT_TAG_ORDER.indexOf(b));
  }

  function stemOf(name) {
    return String(name).replace(/\.[^.]+$/, '');
  }

  function esc(value) {
    return String(value).replace(/[&<>"']/g, (c) => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));
  }

  function buildRowHtml(item, options = {}) {
    const isSelected = state.selectedPhoto && state.selectedPhoto.path === item.path;
    const scored = item.status !== 'pending' && item.status !== 'decode_failed';
    const num = (v, digits) => (scored && Number.isFinite(v) ? v.toFixed(digits) : '—');
    const ratingDisplay = item.status === 'pending' || item.status === 'decode_failed'
      ? '<span style="color:#475569;">—</span>'
      : item.rating > 0
        ? `<span class="tau-stars">${'★'.repeat(item.rating)}</span>`
        : `<span class="tau-reject-tag">${I18N.t('table.tag_reject')}</span>`;

    const translatedVeto = I18N.translateVeto(item.veto);
    let reasonDisplay = item.veto
      ? `<span class="tau-veto-desc" title="${esc(item.veto)}">${esc(translatedVeto)}</span>`
      : item.rating > 0
        ? `<span class="tau-pass-tag">${I18N.t('table.tag_passed')}</span>`
        : '—';

    if (options.isTopN) {
      reasonDisplay = `<span class="tau-topn-tag" title="${I18N.t('burst.topn_desc')}">${I18N.t('burst.topn_badge')}</span>`;
    }

    const statusDisplay = item.status === 'pending'
      ? `<span style="color:#64748b;">${I18N.t('status.queued')}</span>`
      : (item.status === 'decode_failed'
        ? `<span style="color:#f87171;">${I18N.t('status.failed')}</span>`
        : (item.status === 'scored' || item.status === 'topn_final'
          ? `<span style="color:#00e5ff;">${I18N.t('status.scored')}</span>`
          : item.status));

    // Badge placed as suffix to preserve strict left vertical alignment of file names
    let suffixBadge = '';
    if (options.isWinner) {
      suffixBadge = `<span class="tau-winner-badge">${I18N.t('burst.winner_badge')}</span>`;
    } else if (options.isSingle) {
      suffixBadge = `<span class="tau-single-badge">${I18N.t('burst.single_badge')}</span>`;
    }

    const rowClasses = [
      isSelected ? 'selected' : '',
      options.isBurstChild ? 'tau-burst-child' : '',
      options.isSingle ? 'tau-single-row' : '',
    ].filter(Boolean).join(' ');

    // Extension(s) replaced by format tag chips; the full name stays in the
    // cell title (hover) and in item.name for sorting/copying. When the
    // engine sent sibling extensions (exts), one tag per on-disk format is
    // shown ([HEIF][RAW]...); otherwise fall back to the row's own extension.
    const fmtTags = (Array.isArray(item.exts) && item.exts.length > 0)
      ? formatTagsForExts(item.exts)
      : [formatTagFor(item.name)].filter(Boolean);
    const fmtTagHtml = fmtTags
      .map((t) => `<span class="tau-fmt-tag${t === RAW_FORMAT_TAG ? ' tau-fmt-raw' : ''}">${esc(t)}</span>`)
      .join('');

    return `
      <tr id="${rowIdFor(item)}" data-path="${esc(item.path)}" class="${rowClasses}">
        <td title="${esc(item.name)}" style="font-family: var(--tau-font-mono); font-weight: 500;">
          <span class="tau-fname-text">${esc(stemOf(item.name))}</span>${fmtTagHtml}${suffixBadge}
        </td>
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
    if (state.viewMode === 'grouped') {
      // Scores also affect the group header stats, but a full table rebuild
      // per frame event (17-37 fps) is far too expensive. Update the row in
      // place like flat mode, then coalesce header refresh into one delayed
      // renderTable. A missing row is normal here (collapsed group) — the
      // coalesced refresh restores consistency.
      updateTableRowFlatInPlace(item, false);
      scheduleGroupedRefresh();
      if (state.selectedPhoto && state.selectedPhoto.path === item.path) {
        selectPhoto(item, false);
      }
      return;
    }

    updateTableRowFlatInPlace(item, true);
  }

  function updateTableRowFlatInPlace(item, renderIfMissing) {
    const row = document.getElementById(rowIdFor(item));
    if (!row) {
      if (renderIfMissing) {
        renderTable();
      }
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
  }

  // Coalesced refresh of grouped-view headers/stats: at most one pending
  // refresh, triggered after the burst of frame events subsides. During a
  // run the refresh must NOT rebuild the tbody: render-all makes a full
  // renderTable cost hundreds of ms at 10k rows while frames stream at
  // 17-37 fps, so the old 400ms coalesced rebuild ran continuously. Rows
  // are already updated in place per frame; mid-run only headers and
  // winner badges are refreshed. finishRun does the final full render.
  function scheduleGroupedRefresh() {
    if (state.groupedRefreshTimer) return;
    state.groupedRefreshTimer = setTimeout(() => {
      state.groupedRefreshTimer = null;
      if (state.viewMode !== 'grouped') return;
      if (state.isRunning) {
        refreshGroupedHeadersInPlace();
      } else {
        renderTable();
      }
    }, 400);
  }

  function refreshGroupedHeadersInPlace() {
    const clusters = computeBurstClusters(getFilteredPhotos());
    for (const entry of clusters) {
      if (entry.type !== 'burst_header') continue;
      const tr = els.tableBody.querySelector(
        `tr.tau-group-header-row[data-group-id="${CSS.escape(entry.groupId)}"]`);
      if (!tr) continue;
      const isCollapsed = !tr.classList.contains('expanded');
      tr.querySelector('td.tau-gh-cell').innerHTML = buildGroupHeaderCellHtml(entry, isCollapsed);
    }

    // Winner badges: clear and re-place on the current best scored frame,
    // anchored AFTER the format tag chips to match buildRowHtml's order.
    els.tableBody.querySelectorAll('.tau-winner-badge').forEach((el) => el.remove());
    for (const entry of clusters) {
      if (entry.type !== 'burst_header' || !entry.winnerPath) continue;
      const row = document.getElementById(rowIdFor({ path: entry.winnerPath }));
      if (!row) continue;
      const chips = row.querySelectorAll('.tau-fmt-tag');
      const anchor = chips.length ? chips[chips.length - 1] : row.querySelector('.tau-fname-text');
      if (anchor) {
        anchor.insertAdjacentHTML('afterend',
          `<span class="tau-winner-badge">${I18N.t('burst.winner_badge')}</span>`);
      }
    }
  }

  // --- Feature 3: Pan & Zoom Interactions ---
  function resetZoom() {
    state.zoom.level = 1.0;
    state.zoom.panX = 0;
    state.zoom.panY = 0;
    state.zoom.isPanning = false;
    applyZoomTransform();
  }

  function applyZoomTransform() {
    if (!els.previewImg) return;
    const { level, panX, panY } = state.zoom;
    els.previewImg.style.transform = `scale(${level}) translate(${panX / level}px, ${panY / level}px)`;
    if (els.zoomLevelIndicator) {
      els.zoomLevelIndicator.textContent = `${Math.round(level * 100)}%`;
    }
    if (els.previewContainer) {
      if (level > 1.0) {
        els.previewContainer.classList.add('zoomed');
      } else {
        els.previewContainer.classList.remove('zoomed');
      }
    }
  }

  function initPanZoom() {
    if (!els.previewContainer) return;

    // Wheel Zoom
    els.previewContainer.addEventListener('wheel', (e) => {
      if (!els.previewImg || els.previewImg.style.display === 'none') return;
      e.preventDefault();
      const step = 0.25;
      if (e.deltaY < 0) {
        state.zoom.level = Math.min(5.0, parseFloat((state.zoom.level + step).toFixed(2)));
      } else {
        state.zoom.level = Math.max(1.0, parseFloat((state.zoom.level - step).toFixed(2)));
      }
      if (state.zoom.level === 1.0) {
        state.zoom.panX = 0;
        state.zoom.panY = 0;
      }
      applyZoomTransform();
    }, { passive: false });

    // Drag Pan
    els.previewContainer.addEventListener('mousedown', (e) => {
      if (state.zoom.level <= 1.0) return;
      state.zoom.isPanning = true;
      state.zoom.startX = e.clientX - state.zoom.panX;
      state.zoom.startY = e.clientY - state.zoom.panY;
      e.preventDefault();
    });

    window.addEventListener('mousemove', (e) => {
      if (!state.zoom.isPanning) return;
      let rawX = e.clientX - state.zoom.startX;
      let rawY = e.clientY - state.zoom.startY;

      // Viewport boundary clamp: prevent dragging completely out of sight
      const boundsW = (els.previewContainer.offsetWidth || 400) * (state.zoom.level - 0.5);
      const boundsH = (els.previewContainer.offsetHeight || 400) * (state.zoom.level - 0.5);
      state.zoom.panX = Math.max(-boundsW, Math.min(boundsW, rawX));
      state.zoom.panY = Math.max(-boundsH, Math.min(boundsH, rawY));
      applyZoomTransform();
    });

    window.addEventListener('mouseup', () => {
      state.zoom.isPanning = false;
    });

    // Double Click toggle 1x and 2.5x
    els.previewContainer.addEventListener('dblclick', () => {
      if (!els.previewImg || els.previewImg.style.display === 'none') return;
      if (state.zoom.level > 1.0) {
        resetZoom();
      } else {
        state.zoom.level = 2.5;
        applyZoomTransform();
      }
    });

    if (els.btnResetZoom) {
      els.btnResetZoom.addEventListener('click', resetZoom);
    }

    if (els.btnLockZoom) {
      const updateLockUI = () => {
        if (els.lockZoomIcon) {
          els.lockZoomIcon.textContent = state.zoom.locked ? '🔒' : '🔓';
        }
        if (els.btnLockZoom) {
          if (state.zoom.locked) els.btnLockZoom.classList.add('active');
          else els.btnLockZoom.classList.remove('active');
        }
      };
      updateLockUI();
      els.btnLockZoom.addEventListener('click', () => {
        state.zoom.locked = !state.zoom.locked;
        localStorage.setItem('ac-zoom-locked', state.zoom.locked ? 'true' : 'false');
        updateLockUI();
        appendLog(`[View] Viewport lock ${state.zoom.locked ? 'enabled (zoom preserved across photos)' : 'disabled'}`);
      });
    }
  }

  // --- Photo Selection & Thumbnail Preview (Anti-Race) ---
  async function selectPhoto(item, shouldResetZoom = true) {
    if (!item) return;
    state.selectedPhoto = item;

    document.querySelectorAll('#photoTable tbody tr').forEach((r) => r.classList.remove('selected'));
    const row = document.getElementById(rowIdFor(item));
    if (row) {
      row.classList.add('selected');
      row.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }

    els.previewTitle.textContent = stemOf(item.name);
    els.previewScoreDetails.style.display = 'flex';
    if (els.previewZoomControls) els.previewZoomControls.style.display = 'flex';

    if (shouldResetZoom && (!state.zoom.locked || state.zoom.level <= 1.0)) {
      resetZoom();
    }

    const pillScored = item.status !== 'pending' && item.status !== 'decode_failed';
    const pillNum = (v, digits) => (pillScored && Number.isFinite(v) ? v.toFixed(digits) : '-');
    els.pillRating.textContent = `RATING: ${item.rating > 0 ? `${item.rating}★` : (item.rating === -1 ? I18N.t('table.tag_reject') : '-')}`;
    els.pillSharp.textContent = `SHARP: ${pillNum(item.sharp, 3)}`;
    els.pillComp.textContent = `COMP: ${pillNum(item.comp, 3)}`;
    els.pillRaw.textContent = `RAW: ${pillNum(item.raw, 2)}`;
    const reasonText = item.veto ? I18N.translateVeto(item.veto) : (item.rating > 0 ? I18N.t('table.tag_passed') : I18N.t('status.queued'));
    els.pillReason.textContent = `REASON: ${reasonText}`;

    // Feature 4: Anti-Race Sequence tracking
    const currentSeq = ++state.previewSeq;
    const requestedPath = item.path;

    // Same photo already rendered: skip the decode IPC (re-scoring a selected
    // photo used to re-fetch the 640px preview on every frame event).
    if (state.previewLoadedPath === requestedPath && els.previewImg.style.display === 'block') {
      return;
    }

    try {
      const res = await invokeTauri('preview', { path: requestedPath, size: 640 });
      // Discard stale in-flight responses
      if (currentSeq !== state.previewSeq || !state.selectedPhoto || state.selectedPhoto.path !== requestedPath) {
        return;
      }

      if (res && res.data) {
        const src = res.data.startsWith('data:') ? res.data : `data:image/png;base64,${res.data}`;
        els.previewImg.src = src;
        els.previewImg.style.display = 'block';
        els.previewEmpty.style.display = 'none';
        state.previewLoadedPath = requestedPath;
      } else {
        els.previewImg.style.display = 'none';
        els.previewEmpty.style.display = 'flex';
        els.previewEmpty.querySelector('.tau-empty-title').textContent = I18N.t('preview.error_load_title');
        els.previewEmpty.querySelector('.tau-empty-desc').textContent = item.name;
      }
    } catch (err) {
      if (currentSeq === state.previewSeq && state.selectedPhoto && state.selectedPhoto.path === requestedPath) {
        appendLog(`[Preview Error] ${err}`);
        els.previewImg.style.display = 'none';
        els.previewEmpty.style.display = 'flex';
        els.previewEmpty.querySelector('.tau-empty-title').textContent = I18N.t('preview.error_fail_title');
        els.previewEmpty.querySelector('.tau-empty-desc').textContent = `${err}`;
      }
    }
  }

  // --- Feature 1: Keyboard Navigation & Hotkeys ---
  function initKeyboardHotkeys() {
    window.addEventListener('keydown', (e) => {
      // Hotkey Protection: ignore inside inputs
      const tag = (e.target && e.target.tagName ? e.target.tagName.toLowerCase() : '');
      if (tag === 'input' || tag === 'textarea' || tag === 'select') {
        return;
      }

      // Allow Cmd+O for open directory
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'o') {
        e.preventDefault();
        chooseFolder();
        return;
      }

      if (e.metaKey || e.ctrlKey || e.altKey) return;

      const visible = getVisiblePhotosList();
      if (visible.length === 0) return;

      const currentIdx = visible.findIndex((p) => p.path === (state.selectedPhoto ? state.selectedPhoto.path : null));
      const key = e.key.toLowerCase();

      switch (key) {
        case 'arrowdown':
        case 'j': {
          e.preventDefault();
          const nextIdx = currentIdx < visible.length - 1 ? currentIdx + 1 : currentIdx;
          selectPhoto(visible[nextIdx]);
          break;
        }
        case 'arrowup':
        case 'k': {
          e.preventDefault();
          const prevIdx = currentIdx > 0 ? currentIdx - 1 : 0;
          selectPhoto(visible[prevIdx]);
          break;
        }
        case 'home': {
          e.preventDefault();
          selectPhoto(visible[0]);
          break;
        }
        case 'end': {
          e.preventDefault();
          selectPhoto(visible[visible.length - 1]);
          break;
        }
        case ' ': {
          e.preventDefault();
          if (els.previewContainer && els.previewImg.style.display !== 'none') {
            if (state.zoom.level > 1.0) resetZoom();
            else {
              state.zoom.level = 2.0;
              applyZoomTransform();
            }
          }
          break;
        }
        case '1':
        case '2':
        case '3':
        case '4':
        case '5': {
          if (state.selectedPhoto) {
            e.preventDefault();
            manuallySetRating(state.selectedPhoto, parseInt(key, 10));
          }
          break;
        }
        case '0':
        case 'x': {
          if (state.selectedPhoto) {
            e.preventDefault();
            manuallySetRating(state.selectedPhoto, -1);
          }
          break;
        }
      }
    });
  }

  function manuallySetRating(photo, newRating) {
    if (!photo) return;
    // Counter deltas must be symmetric: pending / decode_failed items are
    // counted in neither total, everything already scored is counted.
    const wasCounted = photo.status !== 'pending' && photo.status !== 'decode_failed';
    const oldRating = photo.rating;
    photo.rating = newRating;
    photo.status = 'scored';
    if (newRating <= 0) {
      photo.veto = 'manual_reject';
    } else {
      photo.veto = '';
    }

    const wasKeep = wasCounted && oldRating > 0;
    const wasReject = wasCounted && oldRating <= 0;
    const isKeep = newRating > 0;
    const isReject = newRating <= 0;
    state.keepCount = Math.max(0, state.keepCount + (isKeep ? 1 : 0) - (wasKeep ? 1 : 0));
    state.rejectCount = Math.max(0, state.rejectCount + (isReject ? 1 : 0) - (wasReject ? 1 : 0));

    els.countKeep.textContent = state.keepCount;
    els.countReject.textContent = state.rejectCount;

    updateTableRow(photo);
    selectPhoto(photo, false);
    appendLog(`[Manual] ${photo.name} rating set to ${newRating > 0 ? `${newRating}★` : 'REJECT'}`);

    // Trigger 2: user modified rating -> incremental debounced save
    markPhotoDirty(photo);
  }

  // --- Metadata Persistence Engine (Save Triggers 1, 2, 3, 4) ---
  function markPhotoDirty(photo) {
    if (!photo) return;
    state.dirtyPhotos.add(photo);
    updateSaveButtonState();
    scheduleIncrementalSave(300);
  }

  function updateSaveButtonState() {
    if (!els.btnSaveMetadata) return;
    const dirtyCount = state.dirtyPhotos.size;
    if (dirtyCount > 0) {
      els.btnSaveMetadata.disabled = false;
      if (els.saveBadge) {
        els.saveBadge.style.display = 'inline-block';
        els.saveBadge.textContent = dirtyCount;
      }
      if (els.btnSaveMetadataText) {
        els.btnSaveMetadataText.textContent = `${I18N.t('topbar.btn_save')} (${dirtyCount})`;
      }
    } else {
      els.btnSaveMetadata.disabled = true;
      if (els.saveBadge) els.saveBadge.style.display = 'none';
      if (els.btnSaveMetadataText) {
        els.btnSaveMetadataText.textContent = I18N.t('topbar.btn_save');
      }
    }
  }

  let _saveDebounceTimer = null;
  function scheduleIncrementalSave(delayMs = 300) {
    if (_saveDebounceTimer) clearTimeout(_saveDebounceTimer);
    _saveDebounceTimer = setTimeout(() => {
      flushSaveMetadata();
    }, delayMs);
  }

  async function flushSaveMetadata() {
    if (state.dirtyPhotos.size === 0) {
      if (state.exitPending) {
        await invokeTauri('exit_app');
      }
      return;
    }
    if (state.isSaving) return;
    state.isSaving = true;

    const itemsToSave = [];
    for (const p of state.dirtyPhotos) {
      itemsToSave.push({
        path: p.path,
        rating: p.rating,
        crop: p.crop || null,
      });
    }

    try {
      await invokeTauri('save_metadata', { items: itemsToSave });
      for (const it of itemsToSave) {
        const obj = state.photoMap.get(it.path);
        if (obj) state.dirtyPhotos.delete(obj);
      }
      appendLog(`[Save] ${itemsToSave.length} metadata records safely persisted`);
    } catch (err) {
      appendLog(`[Save Error] Failed to persist metadata: ${err}`);
    } finally {
      state.isSaving = false;
      updateSaveButtonState();
      if (state.exitPending) {
        await invokeTauri('exit_app');
      }
    }
  }


  // --- Feature 7: Context Menu & File Integration ---
  function initContextMenu() {
    if (!els.contextMenu) return;

    // Trigger on table row right-click
    els.tableBody.addEventListener('contextmenu', (e) => {
      const row = e.target.closest('tr');
      if (!row || row.classList.contains('tau-empty-row') || row.classList.contains('tau-group-header-row')) return;
      e.preventDefault();

      const path = row.getAttribute('data-path');
      const item = state.photoMap.get(path);
      if (!item) return;

      state.contextPhoto = item;
      selectPhoto(item, false);

      // Clamp context menu inside window bounds
      const menuW = 190;
      const menuH = 180;
      const winW = window.innerWidth;
      const winH = window.innerHeight;

      let x = e.clientX;
      let y = e.clientY;
      if (x + menuW > winW) x = Math.max(0, winW - menuW - 8);
      if (y + menuH > winH) y = Math.max(0, winH - menuH - 8);

      els.contextMenu.style.left = `${x}px`;
      els.contextMenu.style.top = `${y}px`;
      els.contextMenu.style.display = 'block';
    });

    // Close on click outside or Esc
    window.addEventListener('click', () => {
      els.contextMenu.style.display = 'none';
    });

    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') els.contextMenu.style.display = 'none';
    });

    // Action handlers
    els.ctxShowInFolder.addEventListener('click', async () => {
      if (!state.contextPhoto) return;
      try {
        await invokeTauri('show_in_folder', { path: state.contextPhoto.path });
      } catch (err) {
        appendLog(`[Show in folder error] ${err}`);
      }
    });

    els.ctxCopyPath.addEventListener('click', () => {
      if (!state.contextPhoto) return;
      navigator.clipboard.writeText(state.contextPhoto.path).then(() => {
        appendLog(`[Clipboard] Copied path: ${state.contextPhoto.path}`);
      });
    });

    els.ctxCopyName.addEventListener('click', () => {
      if (!state.contextPhoto) return;
      navigator.clipboard.writeText(state.contextPhoto.name).then(() => {
        appendLog(`[Clipboard] Copied name: ${state.contextPhoto.name}`);
      });
    });

    els.ctxMarkKeep5.addEventListener('click', () => {
      if (state.contextPhoto) manuallySetRating(state.contextPhoto, 5);
    });

    els.ctxMarkKeep3.addEventListener('click', () => {
      if (state.contextPhoto) manuallySetRating(state.contextPhoto, 3);
    });

    els.ctxMarkReject.addEventListener('click', () => {
      if (state.contextPhoto) manuallySetRating(state.contextPhoto, -1);
    });
  }

  // --- Feature 5: Drag & Drop Folder Import ---
  function initDragAndDrop() {
    const overlay = els.dragDropOverlay;
    if (!overlay) return;

    let dragCounter = 0;

    const showOverlay = () => {
      overlay.style.display = 'flex';
    };

    const hideOverlay = () => {
      dragCounter = 0;
      overlay.style.display = 'none';
    };

    window.addEventListener('dragenter', (e) => {
      e.preventDefault();
      dragCounter++;
      showOverlay();
    });

    window.addEventListener('dragover', (e) => {
      e.preventDefault();
      if (e.dataTransfer) {
        e.dataTransfer.dropEffect = 'copy';
      }
      showOverlay();
    });

    window.addEventListener('dragleave', (e) => {
      e.preventDefault();
      dragCounter--;
      if (dragCounter <= 0) {
        hideOverlay();
      }
    });

    window.addEventListener('drop', (e) => {
      e.preventDefault();
      hideOverlay();

      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        const file = e.dataTransfer.files[0];
        const dir = file.path || file.name;
        if (dir) handleDirectoryLoaded(dir);
      }
    });

    overlay.addEventListener('dragover', (e) => {
      e.preventDefault();
      if (e.dataTransfer) {
        e.dataTransfer.dropEffect = 'copy';
      }
    });

    overlay.addEventListener('drop', (e) => {
      e.preventDefault();
      hideOverlay();
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        const file = e.dataTransfer.files[0];
        const dir = file.path || file.name;
        if (dir) handleDirectoryLoaded(dir);
      }
    });

    // Tauri 2 Official Webview Window DragDrop Event Hook
    try {
      const tauriWindow = window.__TAURI__?.webviewWindow?.getCurrentWebviewWindow?.()
        || window.__TAURI__?.window?.getCurrentWindow?.();
      if (tauriWindow && typeof tauriWindow.onDragDropEvent === 'function') {
        tauriWindow.onDragDropEvent((event) => {
          const type = event?.payload?.type;
          if (type === 'enter' || type === 'over') {
            showOverlay();
          } else if (type === 'leave') {
            hideOverlay();
          } else if (type === 'drop') {
            hideOverlay();
            const paths = event?.payload?.paths;
            if (Array.isArray(paths) && paths.length > 0) {
              handleDirectoryLoaded(paths[0]);
            }
          }
        });
      }
    } catch (err) {
      console.warn('Tauri onDragDropEvent binding exception', err);
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
        if (state.selectedPhoto) selectPhoto(state.selectedPhoto, false);
        updateSpeedAndEta();
        updateConfigSummary();
        updateViewModeUI();
        updateRunButtonState();
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

    if (els.btnSaveMetadata) {
      els.btnSaveMetadata.addEventListener('click', () => {
        appendLog('[Manual Save] User triggered metadata save button.');
        flushSaveMetadata();
      });
    }

    if (els.btnToggleConfig) {
      els.btnToggleConfig.addEventListener('click', toggleConfigPanel);
    }

    if (els.btnResetDefaults) {
      els.btnResetDefaults.addEventListener('click', resetToDefaults);
    }

    if (els.btnViewMode) {
      els.btnViewMode.addEventListener('click', toggleViewMode);
    }

    els.inputDir.addEventListener('change', () => {
      const p = els.inputDir.value.trim();
      if (p) handleDirectoryLoaded(p);
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

    // Table Click Delegation (Rows and Group Headers)
    els.tableBody.addEventListener('click', (e) => {
      // 1. Group Header Click to Toggle Expand/Collapse
      const ghRow = e.target.closest('.tau-group-header-row');
      if (ghRow) {
        const groupId = ghRow.getAttribute('data-group-id');
        if (groupId) {
          state.collapsedGroups.add(`_toggled_${groupId}`);
          if (state.collapsedGroups.has(groupId)) {
            state.collapsedGroups.delete(groupId);
          } else {
            state.collapsedGroups.add(groupId);
          }
          renderTable();
        }
        return;
      }

      // 2. Photo Row Click to Select
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

    initSplitter();
    initPanZoom();
    initKeyboardHotkeys();
    initContextMenu();
    initDragAndDrop();
    setupEventListeners();
    loadSavedParams();
  }

  window.addEventListener('DOMContentLoaded', initUI);
})();
