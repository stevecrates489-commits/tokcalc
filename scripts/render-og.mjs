/**
 * Render tokcalc OG image via Playwright snapshot of an HTML template.
 * Outputs:
 *   - /home/z/my-project/public/og.png (2400x1260 @ 2x scale, for all social platforms)
 *
 * The og-icon-256.png is generated separately if needed (see git history for the
 * original render script with the icon step).
 *
 * Usage: node scripts/render-og.mjs
 */
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TEMPLATE = path.join(__dirname, 'og-template.html');
const OUT_OG = path.join(__dirname, '..', 'public', 'og.png');

async function main() {
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1200, height: 630 },
    deviceScaleFactor: 2, // 2x for crisp retina display
  });

  const page = await context.newPage();
  await page.goto('file://' + TEMPLATE, { waitUntil: 'networkidle' });
  // Give web fonts a moment to settle (Inter / JetBrains Mono from Google Fonts CDN)
  await page.waitForTimeout(1000);
  await page.screenshot({
    path: OUT_OG,
    type: 'png',
    clip: { x: 0, y: 0, width: 1200, height: 630 },
  });
  console.log('✓ OG image:', OUT_OG);

  await browser.close();
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
