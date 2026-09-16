import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { deflateSync } from 'node:zlib';
import init, { convert_image } from '../docs/pkg/picvec.js';

const wasm = await readFile(new URL('../docs/pkg/picvec_bg.wasm', import.meta.url));
const bindings = await readFile(new URL('../docs/pkg/picvec.js', import.meta.url));
const { version } = JSON.parse(await readFile(new URL('../docs/pkg/version.json', import.meta.url), 'utf8'));
assert.equal(version, createHash('sha256').update(bindings).update(wasm).digest('hex'),
  'The browser cache version must match the generated bindings and Wasm.');
await init({ module_or_path: wasm });
const png128 = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAGUlEQVR4nGO4pGHTQAlmGDVg1IBRA4aLAQBZNrYQKNcSzgAAAABJRU5ErkJggg==', 'base64');
const png0 = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAG0lEQVR4nGO4pGHDQAmmSPOoAaMGjBowmAwAAAoJNhApr+wgAAAAAElFTkSuQmCC', 'base64');
const svg = convert_image(png128, 1024, false);
assert.match(svg, /<svg/);
assert.match(svg, /fill-opacity=/);
assert.doesNotMatch(svg, /<mask|\smask=/);
const transparent = convert_image(png0, 1024, false);
assert.match(transparent, /<svg/);
assert.doesNotMatch(transparent, /<(?:path|rect|circle|ellipse|polygon|polyline|line)\b/);
assert.throws(() => convert_image(new Uint8Array([1, 2, 3]), 1024, false));
assert.throws(() => convert_image(png128, 0, false));
assert.throws(() => convert_image(png128, 8192, false));
// Errors must not poison the module: a subsequent conversion still works.
assert.equal(convert_image(png128, 1024, false), svg);
// Tiny heights keep the processing-size boundary tests inexpensive in Wasm.
function transparentPng(width, height) {
  function chunk(type, data) {
    const body = Buffer.concat([Buffer.from(type), data]);
    let crc = 0xffffffff;
    for (const byte of body) {
      crc ^= byte;
      for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
    }
    const length = Buffer.alloc(4);
    length.writeUInt32BE(data.length);
    const checksum = Buffer.alloc(4);
    checksum.writeUInt32BE((crc ^ 0xffffffff) >>> 0);
    return Buffer.concat([length, body, checksum]);
  }
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0);
  header.writeUInt32BE(height, 4);
  header[8] = 8;
  header[9] = 6;
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk('IHDR', header),
    chunk('IDAT', deflateSync(Buffer.alloc((width * 4 + 1) * height))),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}
for (const [width, height, limit, expectedWidth, expectedHeight] of [
  [16, 16, 512, 16, 16],
  [16, 16, 1024, 16, 16],
  [16, 16, 2048, 16, 16],
  [512, 2, 512, 512, 2],
  [1024, 4, 512, 512, 2],
  [4, 1024, 512, 2, 512],
  [1024, 2, 1024, 1024, 2],
  [2048, 4, 1024, 1024, 2],
  [4, 2048, 1024, 2, 1024],
  [2048, 2, 2048, 2048, 2],
  [4096, 4, 2048, 2048, 2],
  [4, 4096, 2048, 2, 2048],
]) {
  const output = convert_image(transparentPng(width, height), limit, false);
  const root = output.match(/<svg\b[^>]*>/)[0];
  assert.match(root, new RegExp(`\\bwidth="${expectedWidth}"`));
  assert.match(root, new RegExp(`\\bheight="${expectedHeight}"`));
}
assert.throws(() => convert_image(png128, 256, false));
assert.throws(() => convert_image(png128, 16384, false));
console.log('Wasm smoke tests passed (alpha, errors, reuse, original size and proportional resizing at 512/1024/2048).');
