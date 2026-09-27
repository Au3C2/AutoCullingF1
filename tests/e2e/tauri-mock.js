// tests/e2e/tauri-mock.js — injected via page.addInitScript BEFORE any page
// script. Defines a complete window.__TAURI__ bridge so that:
//   - ui/app.js runs against deterministic fixtures (real engine output)
//   - ui/mock-backend.js stands down (it bails when window.__TAURI__ exists)
//   - every IPC call is counted for dedupe/anti-storm assertions
//
// Fixtures are fetched lazily from /fixtures/previews.json on first invoke.

function tauriMockFactory() {
  const listeners = new Map();
  const stats = { preview: 0, highres: 0, scan: 0, save_metadata: 0, other: 0 };
  let fixturesPromise = null;
  let fixtures = null;

  function ensureFixtures() {
    if (!fixturesPromise) {
      fixturesPromise = fetch('/fixtures/previews.json')
        .then((r) => r.json())
        .then((data) => {
          fixtures = data;
          return data;
        });
    }
    return fixturesPromise;
  }

  function emitEvent(name, payload) {
    const list = listeners.get(name) || [];
    if (window.__TAURI_MOCK_DEBUG__) {
      console.log(`[tauri-mock] emit ${name} -> ${list.length} listener(s)`);
    }
    for (const cb of [...list]) {
      try {
        if (window.__TAURI_MOCK_DEBUG__) console.log(`[tauri-mock] invoking cb for ${name}`);
        cb({ event: name, payload });
        if (window.__TAURI_MOCK_DEBUG__) console.log(`[tauri-mock] cb for ${name} returned`);
      } catch (e) { console.error('[tauri-mock] listener error', e); }
    }
  }

  window.__TAURI_MOCK_STATS__ = stats;
  window.__TAURI_MOCK_EMIT__ = emitEvent;
  window.__TAURI_MOCK_LISTENERS__ = listeners;

  window.__TAURI__ = {
    core: {
      convertFileSrc: (filePath) => filePath, // highres fixture URLs are already http paths
      invoke: async (cmd, args = {}) => {
        const fx = await ensureFixtures();

        if (cmd === 'scan') {
          stats.scan++;
          const photos = fx.photos;
          const paths = photos.map((p) => p.path);
          setTimeout(() => {
            emitEvent('scanned', {
              count: photos.length,
              total: photos.length,
              paths,
              items: photos.map((p) => ({
                path: p.path, name: p.name, timestamp: null, timeStr: null, exts: [],
              })),
            });
          }, 10);
          return { count: photos.length };
        }

        if (cmd === 'preview') {
          stats.preview++;
          const photo = fx.photos.find((p) => p.path === args.path)
            || fx.photos.find((p) => p.path.endsWith('/' + args.path));
          if (!photo) return { path: args.path, data: null, png: null };
          return {
            path: args.path,
            data: photo.data,
            png: photo.data,
            width: photo.width,
            height: photo.height,
            boxes: photo.boxes,
            crop: photo.crop,
          };
        }

        if (cmd === 'request_highres') {
          stats.highres++;
          const photo = fx.photos.find((p) => p.path === args.path)
            || fx.photos.find((p) => p.path.endsWith('/' + args.path));
          const genId = args.genId ?? args.gen_id ?? 0;
          if (!photo || !photo.highresUrl) {
            setTimeout(() => emitEvent('highres_discarded', { gen_id: genId, orig_path: args.path }), 20);
            return null;
          }
          setTimeout(() => {
            emitEvent('highres_ready', {
              gen_id: genId,
              path: photo.highresUrl,
              orig_path: args.path,
              format: 'jpg',
              width: photo.highresWidth,
              height: photo.highresHeight,
              tier: 'fixture',
            });
          }, 30);
          return null;
        }

        if (cmd === 'save_metadata') {
          stats.save_metadata++;
          setTimeout(() => emitEvent('save_done', { count: (args.items || []).length, status: 'ok' }), 10);
          return null;
        }

        if (cmd === 'get_optimal_workers') return 4;
        stats.other++;
        return null;
      },
    },
    event: {
      listen: async (eventName, handler) => {
        if (!listeners.has(eventName)) listeners.set(eventName, []);
        listeners.get(eventName).push(handler);
        return () => {
          const list = listeners.get(eventName) || [];
          const i = list.indexOf(handler);
          if (i >= 0) list.splice(i, 1);
        };
      },
    },
  };
}

module.exports = { tauriMockFactory };
