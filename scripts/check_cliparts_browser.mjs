// Optional regression for the Chromium SVG-image renderer used by VS Code.
// Coordinates identify test data, never production conversion branches.
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const [svgPath, outputDirectory, sourcePath = 'sample/input/cliparts.png'] = process.argv.slice(2);
if (!svgPath || !outputDirectory) throw new Error('Usage: node scripts/check_cliparts_browser.mjs SVG OUTPUT_DIRECTORY [SOURCE_PNG]');
fs.mkdirSync(outputDirectory, { recursive: true });
function run(command, args, input) {
  const result = spawnSync(command, args, { input, maxBuffer: 128 * 1024 * 1024 });
  if (result.status !== 0) throw new Error(`${command}: ${result.stderr}`);
  return result.stdout;
}
const [sourceWidth, sourceHeight] = run('identify', ['-format', '%w %h', sourcePath]).toString().split(' ').map(Number);
const source = run('convert', [sourcePath, '-depth', '8', 'rgba:-']);
const svg = fs.readFileSync(svgPath, 'utf8');
const cases = [
  { name: 'oval', crop: [440, 295, 130, 95], core: [467, 308, 80, 61] },
  { name: 'face', crop: [255, 425, 135, 150], core: [277, 453, 96, 94] },
  { name: 'hand', crop: [290, 640, 75, 95], core: [300, 652, 53, 74] },
];
function flatOpaque(x, y) {
  if (x < 1 || y < 1 || x + 1 >= sourceWidth || y + 1 >= sourceHeight) return false;
  const min = [255, 255, 255], max = [0, 0, 0];
  for (let yy = y - 1; yy <= y + 1; yy++) for (let xx = x - 1; xx <= x + 1; xx++) {
    const i = (yy * sourceWidth + xx) * 4;
    if (source[i + 3] !== 255) return false;
    for (let k = 0; k < 3; k++) { min[k] = Math.min(min[k], source[i + k]); max[k] = Math.max(max[k], source[i + k]); }
  }
  return max.every((v, k) => v - min[k] <= 8);
}
const browser = await chromium.launch({ headless: true });
const report = { chromium: await browser.version(), svg_sha256: createHash('sha256').update(svg).digest('hex'), samples: [] };
try {
  for (const item of cases) for (const scale of [1, 3.3, 8]) for (const background of ['#151515', '#ffffff']) {
    const [x, y, w, h] = item.crop, [cx, cy, cw, ch] = item.core;
    const width = Math.round(w * scale), height = Math.round(h * scale), sx = width / w, sy = height / h;
    const cropped = svg.replace(/<svg\b[^>]*>/, root => root.replace(/\bwidth="[^"]*"/, `width="${width}"`).replace(/\bheight="[^"]*"/, `height="${height}"`).replace(/\bviewBox="[^"]*"/, `viewBox="${x} ${y} ${w} ${h}"`));
    const stem = path.join(outputDirectory, `${item.name}-${scale}-${background.slice(1)}`);
    fs.writeFileSync(`${stem}.svg`, cropped);
    const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
    await page.setContent(`<style>html,body{margin:0;background:${background}}img{display:block}</style><img src="data:image/svg+xml;base64,${Buffer.from(cropped).toString('base64')}">`);
    await page.locator('img').evaluate(async image => { await image.decode(); });
    const png = await page.screenshot({ path: `${stem}-chromium.png` });
    await page.close();
    const actual = run('convert', ['png:-', '-depth', '8', 'rgba:-'], png);
    run('rsvg-convert', ['-b', background, `${stem}.svg`, '-o', `${stem}-reference.png`]);
    const reference = run('convert', [`${stem}-reference.png`, '-depth', '8', 'rgba:-']);
    const errors = []; let total = 0;
    for (let py = 0; py < height; py++) for (let px = 0; px < width; px++) {
      const xx = x + (px + 0.5) / sx, yy = y + (py + 0.5) / sy;
      const vertical = yy >= cy && yy <= cy + ch && Math.min(Math.abs(xx - cx), Math.abs(xx - cx - cw)) * sx <= 1.25;
      const horizontal = xx >= cx && xx <= cx + cw && Math.min(Math.abs(yy - cy), Math.abs(yy - cy - ch)) * sy <= 1.25;
      if (!(vertical || horizontal) || !flatOpaque(Math.floor(xx), Math.floor(yy))) continue;
      const i = (py * width + px) * 4;
      const delta = [0, 1, 2].map(k => Math.abs(actual[i + k] - reference[i + k]));
      errors.push(Math.max(...delta)); total += delta.reduce((a, b) => a + b, 0) / 3;
    }
    errors.sort((a, b) => a - b);
    const result = { case: item.name, scale, background, pixels: errors.length, mean_rgb_difference: total / errors.length, p95_max_channel_difference: errors[Math.floor(errors.length * 0.95)], max_channel_difference: errors.at(-1) };
    result.passed = errors.length >= 20 && result.mean_rgb_difference <= 2 && result.p95_max_channel_difference <= 4 && result.max_channel_difference <= 12;
    report.samples.push(result);
  }
} finally { await browser.close(); }
report.passed = report.samples.every(sample => sample.passed);
fs.writeFileSync(path.join(outputDirectory, 'report.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
if (!report.passed) process.exitCode = 1;
