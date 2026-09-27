// tests/e2e/server.js — static host for the Playwright E2E suite.
//
// Serves:
//   /              -> ui/ (the real app shell, real app.js/style.css)
//   /fixtures/*    -> generated rendering fixtures (previews.json, highres/)
//
// The Tauri bridge itself is injected per-page via tests/e2e/tauri-mock.js
// (page.addInitScript), which makes ui/mock-backend.js stand down.

const http = require('http');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const UI_DIR = path.join(ROOT, 'ui');
const FIXTURES_DIR = __dirname + '/fixtures';

const PORT = Number(process.env.E2E_PORT || 8231);

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

function send(res, code, body, type) {
  res.writeHead(code, { 'Content-Type': type || 'text/plain', 'Cache-Control': 'no-store' });
  res.end(body);
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://127.0.0.1:${PORT}`);
  let filePath = null;

  if (url.pathname.startsWith('/fixtures/') || url.pathname.startsWith('/highres/')) {
    // /highres/* maps into fixtures/highres/ (the generated cache assets)
    const rel = url.pathname.startsWith('/highres/')
      ? path.join('highres', url.pathname.replace(/^\/highres\//, ''))
      : url.pathname.replace(/^\/fixtures\//, '');
    filePath = path.join(FIXTURES_DIR, path.normalize(rel));
  } else {
    const rel = url.pathname === '/' ? 'index.html' : url.pathname.replace(/^\//, '');
    filePath = path.join(UI_DIR, rel);
  }

  if (!filePath.startsWith(FIXTURES_DIR) && !filePath.startsWith(UI_DIR)) {
    return send(res, 403, 'forbidden');
  }

  fs.readFile(filePath, (err, data) => {
    if (err) return send(res, 404, `not found: ${url.pathname}`);
    send(res, 200, data, MIME[path.extname(filePath).toLowerCase()] || 'application/octet-stream');
  });
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`[e2e-server] listening on http://127.0.0.1:${PORT}`);
});
