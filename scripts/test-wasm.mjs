import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import init, { convert_image } from '../docs/pkg/picvec.js';

await init({ module_or_path: await readFile(new URL('../docs/pkg/picvec_bg.wasm', import.meta.url)) });
const png128 = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAGUlEQVR4nGO4pGHTQAlmGDVg1IBRA4aLAQBZNrYQKNcSzgAAAABJRU5ErkJggg==', 'base64');
const png0 = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAG0lEQVR4nGO4pGHDQAmmSPOoAaMGjBowmAwAAAoJNhApr+wgAAAAAElFTkSuQmCC', 'base64');
const svg = convert_image(png128, 64, false);
assert.match(svg, /<svg/);
assert.match(svg, /fill-opacity=/);
assert.doesNotMatch(svg, /<mask|\smask=/);
const transparent = convert_image(png0, 64, false);
assert.match(transparent, /<svg/);
assert.doesNotMatch(transparent, /<(?:path|rect|circle|ellipse|polygon|polyline|line)\b/);
assert.throws(() => convert_image(new Uint8Array([1, 2, 3]), 64, false));
assert.throws(() => convert_image(png128, 0, false));
assert.throws(() => convert_image(png128, 2048, false));
// Errors must not poison the module: a subsequent conversion still works.
assert.equal(convert_image(png128, 64, false), svg);
console.log('Wasm smoke tests passed (conversion, alpha, transparent input, errors, reuse).');
