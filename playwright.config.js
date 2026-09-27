// playwright.config.js — L2 pixel-level rendering regression suite.
// WebKit mirrors the macOS Tauri WKWebView engine; Chromium mirrors the
// Windows WebView2 engine family.

const { defineConfig, devices } = require('@playwright/test');

module.exports = defineConfig({
  testDir: 'tests/e2e',
  timeout: 60_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: `http://127.0.0.1:${process.env.E2E_PORT || 8231}`,
    viewport: { width: 1440, height: 900 },
    actionTimeout: 10_000,
  },
  webServer: {
    command: 'node tests/e2e/server.js',
    url: `http://127.0.0.1:${process.env.E2E_PORT || 8231}`,
    reuseExistingServer: !process.env.CI,
    timeout: 15_000,
  },
  projects: [
    { name: 'webkit', use: { ...devices['Desktop WebKit'] } },
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
});
