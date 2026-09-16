// Regression for transparent pinholes in overlapping, oriented paint contours.
// Run after regenerating sample/output/viewport1.svg. Requires Playwright and
// ImageMagick; PLAYWRIGHT_MODULE may point to a local Playwright installation.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const [svgPath = 'sample/output/viewport1.svg', outputDirectory = '/tmp/picvec-viewport-browser'] = process.argv.slice(2);
fs.mkdirSync(outputDirectory, { recursive: true });
const svg = fs.readFileSync(svgPath, 'utf8');
const browser = await chromium.launch({ headless: true });
const report = { chromium: await browser.version(), svg_sha256: createHash('sha256').update(svg).digest('hex'), samples: [] };
try {
  // This crop is the dark mountain in the reported screenshot, with no snow
  // or source transparency. Coordinates belong only to this sample regression.
  const cases = [
    ...[1, 3.3, 8].map(scale => ({ name: 'mountain', scale, x: 0, y: 100, w: 100, h: 100 })),
    { name: 'full', scale: 4, x: 0, y: 0, w: 640, h: 426 },
  ];
  for (const { name, scale, x, y, w, h } of cases) {
    const width = Math.round(w * scale), height = Math.round(h * scale);
    const cropped = svg.replace(/<svg\b[^>]*>/, root => root
      .replace(/\bwidth="[^"]*"/, `width="${width}"`)
      .replace(/\bheight="[^"]*"/, `height="${height}"`)
      .replace(/\bviewBox="[^"]*"/, `viewBox="${x} ${y} ${w} ${h}"`));
    const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
    await page.setContent(`<style>html,body{margin:0}img{display:block}</style><img src="data:image/svg+xml;base64,${Buffer.from(cropped).toString('base64')}">`);
    await page.locator('img').evaluate(image => image.decode());
    const png = await page.screenshot({ path: path.join(outputDirectory, `${name}-${scale}-rgba.png`), omitBackground: true });
    const converted = spawnSync('convert', ['png:-', '-depth', '8', 'rgba:-'], { input: png, maxBuffer: 64 * 1024 * 1024 });
    assert.equal(converted.status, 0, converted.stderr.toString());
    const rgba = converted.stdout;
    assert.equal(rgba.length, width * height * 4);
    let pinholes = 0, lowCoveragePixels = 0;
    // Ignore only the crop boundary. Shared-edge antialiasing may be partially
    // transparent; the bug creates actual holes with near-zero coverage.
    const inset = Math.ceil((name === 'full' ? 2 : 1) * scale);
    for (let y = inset; y < height - inset; y++) for (let x = inset; x < width - inset; x++) {
      const alpha = rgba[(y * width + x) * 4 + 3];
      if (alpha < 128) lowCoveragePixels++;
      if (alpha < 32) pinholes++;
    }
    for (const background of ['#ffffff', '#151515']) {
      await page.evaluate(background => { document.body.style.background = background; }, background);
      await page.screenshot({ path: path.join(outputDirectory, `${name}-${scale}-${background.slice(1)}.png`) });
    }
    report.samples.push({ case: name, scale, pinholes, lowCoveragePixels, passed: pinholes === 0 });
    await page.close();
  }
} finally {
  await browser.close();
}
report.passed = report.samples.every(sample => sample.passed);
fs.writeFileSync(path.join(outputDirectory, 'report.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
assert.ok(report.passed, 'Transparent pinholes appeared in viewport1');
