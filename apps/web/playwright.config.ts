import { defineConfig, devices } from "@playwright/test";

const port = Number(process.env.WEB_PORT ?? 5173);

/**
 * Smoke tests only.
 *
 * These need a running API with an ingested corpus behind them, so they are not part of
 * the pull-request gate -- a browser test that fails because nobody seeded a database
 * teaches people to ignore red, and a gate people ignore protects nothing. Run them by
 * hand with `make web-test` against a live stack.
 */
export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: `http://localhost:${port}`,
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "npm run dev",
    url: `http://localhost:${port}`,
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
