// tests/e2e/rendering.spec.js — pixel-level regression suite for the
// intelligent high-res zoom / preview pipeline.
//
// Every rendering input is deterministic (fixtures = real engine output), so
// the five user-reported symptom classes are machine-decidable:
//   1. black preview / partial black blocks  -> tile-wise compare vs fixture
//   2. detection boxes disappearing          -> stroke-color pixel count
//   3. SVG geometry drift                    -> boundingBox coincidence
//   4. per-frame preview IPC storm           -> mock IPC counters
//   5. asset/decode failures                 -> pageerror/console monitoring

const { test, expect } = require('@playwright/test');
const { tauriMockFactory } = require('./tauri-mock');
const { findBlackHoles, fixturePngBuffer, countPixelsNear } = require('./helpers');

const BOX_COLORS = [
  [46, 204, 113],  // backend-drawn detection rect (green, burned into the PNG)
  [0, 229, 255],   // SVG car box stroke (cyan)
  [251, 191, 36],  // SVG non-car box stroke (gold)
];

async function loadApp(page, fixtures) {
  await page.addInitScript({ content: `(${tauriMockFactory.toString()})()` });
  const consoleErrors = [];
  page.on('pageerror', (err) => consoleErrors.push(String(err)));
  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text());
  });
  await page.goto('/');
  // Drive the scan flow directly through the engine-protocol events (the same
  // payloads the real engine emits after a folder scan). Deterministic and
  // independent of WebKit drag-drop plumbing.
  const items = fixtures.photos.map((p) => ({
    path: p.path, name: p.name, timestamp: null, timeStr: null, exts: [],
  }));
  const scannedPayload = {
    dir: 'E2E://fixtures',
    count: items.length,
    total: items.length,
    paths: items.map((i) => i.path),
    items,
  };
  await page.evaluate((payload) => window.__TAURI_MOCK_EMIT__('scanned', payload), scannedPayload);
  await page.evaluate((payload) => window.__TAURI_MOCK_EMIT__('scan_meta', payload), {
    dir: 'E2E://fixtures', items,
  });
  await expect(page.locator('#photoTable tbody tr[data-path]').first()).toBeVisible({ timeout: 15_000 });
  return consoleErrors;
}

async function previewState(page) {
  return page.evaluate(() => {
    const img = document.getElementById('previewImg');
    const viewport = document.getElementById('previewViewport');
    const canvas = document.getElementById('previewBoxCanvas');
    const highres = document.getElementById('previewHighResImg');
    return {
      naturalWidth: img?.naturalWidth || 0,
      naturalHeight: img?.naturalHeight || 0,
      imgComplete: img ? img.complete : false,
      viewportW: viewport?.offsetWidth || 0,
      viewportH: viewport?.offsetHeight || 0,
      boxStats: window.__AC_BOX_OVERLAY_STATS__ || null,
      highresNaturalWidth: highres?.naturalWidth || 0,
      highresOpacity: highres ? getComputedStyle(highres).opacity : 'n/a',
      previewRequestedPath: null,
      stats: window.__TAURI_MOCK_STATS__,
    };
  });
}

async function clickRow(page, index) {
  await page.locator('#photoTable tbody tr[data-path]').nth(index).click();
}

test.describe('preview rendering pipeline', () => {
  let fixtures;

  test.beforeAll(async ({ request }) => {
    const res = await request.get('/fixtures/previews.json');
    expect(res.ok()).toBeTruthy();
    fixtures = await res.json();
  });

  test('selecting a photo paints the full preview (no black holes / partial blocks)', async ({ page }) => {
    const consoleErrors = await loadApp(page, fixtures);

    for (const index of [0, 1, 8, 16]) {
      const photo = fixtures.photos[index];
      await clickRow(page, index);

      const state = await previewState(page);
      expect(state.imgComplete, `img.complete for ${photo.name}`).toBeTruthy();
      expect(state.naturalWidth).toBe(photo.width);
      expect(state.viewportW, `viewport hugs photo for ${photo.name}`).toBeGreaterThan(0);

      // Geometry: viewport must match the photo aspect (contain-fit rect).
      // NOTE: #previewImg carries the zoom CSS transform (auto crop-focus),
      // so its boundingBox is transformed — compare untransformed layers.
      const vpBox = await page.locator('#previewViewport').boundingBox();
      const canvasBox = await page.locator('#previewBoxCanvas').boundingBox();
      expect(Math.abs(vpBox.width / vpBox.height - photo.width / photo.height)).toBeLessThan(0.02);
      expect(Math.abs(canvasBox.x - vpBox.x)).toBeLessThan(2);
      expect(Math.abs(canvasBox.y - vpBox.y)).toBeLessThan(2);
      expect(Math.abs(canvasBox.width - vpBox.width)).toBeLessThan(2);

      // Pixel truth: normalize to 100% (auto crop-focus may have zoomed the
      // view) so the visible frame equals the full fixture image, then
      // compare tile-wise — any black/unpainted region fails.
      await page.fill('#zoomLevelInput', '100');
      await page.keyboard.press('Enter');
      await page.waitForTimeout(80);
      const shot = await page.locator('#previewImg').screenshot();
      const { mae, blackHoles } = findBlackHoles(shot, fixturePngBuffer(photo.data));
      expect(blackHoles, `${photo.name}: black/unpainted tiles ${JSON.stringify(blackHoles)}`).toHaveLength(0);
      expect(mae, `${photo.name}: mean abs tile diff`).toBeLessThan(30);

      if (photo.boxes.length > 0) {
        // Canvas overlay: detection boxes must be drawn over the painted photo
        expect(state.boxStats, `${photo.name}: box overlay stats present`).toBeTruthy();
        expect(state.boxStats.boxes).toBe(photo.boxes.length);
        const canvasShot = await page.locator('#previewBoxCanvas').screenshot();
        const boxPixels = BOX_COLORS.reduce((sum, c) => sum + countPixelsNear(canvasShot, c, 55), 0);
        expect(boxPixels, `${photo.name}: box stroke pixels visible`).toBeGreaterThan(80);
      } else if (photo.crop) {
        // Crop-only photo: dashed crop rect must be drawn
        expect(state.boxStats.crop).toBe(true);
      }
    }

    expect(consoleErrors.filter((e) => /EncodingError|Loading error/.test(e))).toHaveLength(0);
  });

  test('rapid photo switching does not storm preview IPC and always settles painted', async ({ page }) => {
    await loadApp(page, fixtures);

    // 10 rapid switches through keyboard navigation (< 60ms apart)
    await clickRow(page, 0);
    await expect(page.locator('#previewImg')).toHaveJSProperty('naturalWidth', fixtures.photos[0].width);
    for (let i = 1; i <= 10; i++) {
      await page.keyboard.press('j');
      await page.waitForTimeout(50);
    }

    const state = await previewState(page);
    // Anti-storm: one IPC per photo selection, not one per frame event /
    // repeated re-entry. 11 selections -> <= 12 preview calls.
    expect(state.stats.preview).toBeLessThanOrEqual(12);

    // Final photo must be fully painted
    const finalIndex = 10 % fixtures.photos.length;
    const photo = fixtures.photos[finalIndex];
    const shot = await page.locator('#previewImg').screenshot();
    const { blackHoles } = findBlackHoles(shot, fixturePngBuffer(photo.data));
    expect(blackHoles, `black tiles after rapid switching: ${JSON.stringify(blackHoles)}`).toHaveLength(0);
  });

  test('zooming to 250% fades in the high-res layer and keeps detection boxes visible', async ({ page }) => {
    await loadApp(page, fixtures);

    // Pick a photo that has both boxes and a highres fixture asset
    const index = fixtures.photos.findIndex((p) => p.boxes.length > 0 && p.highresUrl);
    expect(index, 'fixture set must contain a boxed photo with highres asset').toBeGreaterThanOrEqual(0);
    const photo = fixtures.photos[index];

    await clickRow(page, index);
    await expect(page.locator('#previewImg')).toHaveJSProperty('naturalWidth', photo.width);

    await page.fill('#zoomLevelInput', '250');
    await page.keyboard.press('Enter');

    // High-res layer must actually load through the (mocked) asset protocol
    await expect(page.locator('#previewHighResImg')).toHaveJSProperty('naturalWidth', photo.highresWidth, { timeout: 10_000 });
    await expect(page.locator('#previewHighResImg')).toHaveJSProperty('naturalHeight', photo.highresHeight);
    await expect(page.locator('#previewHighResImg')).toHaveCSS('opacity', '1');

    // The photo must STILL be fully painted under the high-res overlay
    const shot = await page.locator('#previewViewport').screenshot();
    const { blackHoles } = findBlackHoles(shot, fixturePngBuffer(photo.data));
    expect(blackHoles, `black tiles at 250% zoom: ${JSON.stringify(blackHoles)}`).toHaveLength(0);

    // Detection boxes must remain drawn on the canvas overlay, on top of the
    // high-res layer, after the zoom change
    const state = await previewState(page);
    expect(state.boxStats).toBeTruthy();
    expect(state.boxStats.boxes).toBe(photo.boxes.length);
    const canvasShot = await page.locator('#previewBoxCanvas').screenshot();
    const boxPixels = BOX_COLORS.reduce((sum, c) => sum + countPixelsNear(canvasShot, c, 55), 0);
    expect(boxPixels, 'box stroke pixels visible at 250%').toBeGreaterThan(80);
  });

  test('returning from 250% to 100% restores the low-res view (no lingering black)', async ({ page }) => {
    await loadApp(page, fixtures);
    const index = fixtures.photos.findIndex((p) => p.boxes.length > 0 && p.highresUrl);
    const photo = fixtures.photos[index];

    await clickRow(page, index);
    await page.fill('#zoomLevelInput', '250');
    await page.keyboard.press('Enter');
    await expect(page.locator('#previewHighResImg')).toHaveJSProperty('naturalWidth', photo.highresWidth);

    await page.fill('#zoomLevelInput', '100');
    await page.keyboard.press('Enter');
    await expect(page.locator('#previewHighResImg')).toHaveCSS('opacity', '0');

    const shot = await page.locator('#previewImg').screenshot();
    const { blackHoles } = findBlackHoles(shot, fixturePngBuffer(photo.data));
    expect(blackHoles).toHaveLength(0);
  });
});
