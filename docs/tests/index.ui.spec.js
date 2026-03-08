const fs = require("fs");
const path = require("path");
const { test, expect } = require("@playwright/test");

const outputDir = path.join(process.cwd(), "output", "playwright");
const indexFile = path.join(process.cwd(), "docs", "index.html");
const indexFileUrl = `file://${indexFile}`;
const chromeExecutable = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

test.use({
  browserName: "chromium",
  launchOptions: {
    executablePath: chromeExecutable,
  },
});

test.beforeAll(() => {
  fs.mkdirSync(outputDir, { recursive: true });
});

test("local file view stays minimal and navigable", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.goto(indexFileUrl);

  await expect(page.locator(".site-heading")).toBeVisible();
  await expect(page.locator(".section-nav")).toBeVisible();
  await expect(page.locator(".usage-note")).toBeVisible();
  await expect(page.locator(".hero-copy h1")).toContainText("Bus Delay Patterns in Tübingen");

  await page.selectOption("#line-select", "1");
  await expect(page.locator("#spotlight-title")).toHaveText("Line 1 in focus");
  await expect(page.locator("#map-frame")).toHaveAttribute("src", /lines\/all\/network_1\.html$/);

  await page.selectOption("#period-select", "pre");
  await expect(page.locator("#plot-cdf-pdf")).toHaveAttribute("src", /plots\/pre\/delay_cdf_pdf_combo\.png$/);

  await expect(page.locator(".live-details")).not.toHaveAttribute("open", "");
  await page.locator(".live-details summary").click();
  await expect(page.locator(".live-details")).toHaveAttribute("open", "");
  await expect(page.locator("#live-monitor-frame")).toHaveAttribute("src", /efa-bw\.de\/rtMonitor/);

  await page.screenshot({
    path: path.join(outputDir, "index-file-desktop.png"),
    fullPage: true,
  });
});

test("mobile layout keeps navigation accessible", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(indexFileUrl);

  await expect(page.locator(".section-nav")).toBeVisible();
  await expect(page.locator(".section-nav a")).toHaveCount(4);

  await page.screenshot({
    path: path.join(outputDir, "index-file-mobile.png"),
    fullPage: true,
  });
});
