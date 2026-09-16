const $ = (id) => document.getElementById(id);
let selectedFile;
let originalURL;
let resultURL;
let worker;
let timer;
let selection = 0;
let busy = false;
const supported = 'WebAssembly' in window && 'Worker' in window;

function status(message, error = false) {
  $('status').textContent = message;
  $('status').parentElement.classList.toggle('error', error);
}
function clearResult() {
  if (resultURL) URL.revokeObjectURL(resultURL);
  resultURL = undefined;
  $('result').removeAttribute('src');
  $('result').hidden = true;
  $('result-placeholder').hidden = false;
  $('download').hidden = true;
  $('download').removeAttribute('href');
  $('result-info').textContent = '';
}
function stop() {
  worker?.terminate();
  worker = undefined;
  clearInterval(timer);
  busy = false;
  $('cancel').hidden = true;
  $('convert').disabled = !selectedFile;
  $('size').disabled = false;
  $('background').disabled = false;
}
async function selectFile(file) {
  if (!file || !supported) return;
  const current = ++selection;
  stop();
  selectedFile = undefined;
  $('convert').disabled = true;
  clearResult();
  $('elapsed').textContent = '';
  $('filename').textContent = 'Made for a closer look.';
  $('source-info').textContent = '';
  $('original').hidden = true;
  $('original').removeAttribute('src');
  $('original-placeholder').hidden = false;
  if (originalURL) URL.revokeObjectURL(originalURL);
  originalURL = undefined;
  if (file.size > 20 * 1024 * 1024) {
    status('Choose an image smaller than 20 MiB.', true);
    return;
  }
  status('Checking image…');
  try {
    // Read only the header here. Rust checks dimensions and decoder limits before
    // full decode; avoid decoding an oversized image just to display a preview.
    const bytes = new Uint8Array(await file.slice(0, 32).arrayBuffer());
    if (current !== selection) return;
    const png = [137, 80, 78, 71, 13, 10, 26, 10].every((value, i) => bytes[i] === value);
    const jpeg = bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255;
    if (!png && !jpeg) throw new Error('Please choose a PNG or JPEG image.');
    selectedFile = file;
    $('filename').textContent = file.name;
    $('source-info').textContent = formatBytes(file.size);
    $('convert').disabled = false;
    status('Ready to convert.');
    // Preview is loaded after Wasm validates and converts the input.
    $('original-placeholder').textContent = 'Preview appears after conversion';
  } catch (error) {
    if (current === selection) status(error.message, true);
  }
}
function formatBytes(size) {
  return size >= 1024 * 1024 ? `${(size / 1024 / 1024).toFixed(1)} MiB` : `${(size / 1024).toFixed(1)} KiB`;
}
$('file').addEventListener('change', (event) => selectFile(event.target.files[0]));
const zone = $('drop-zone');
for (const type of ['dragenter', 'dragover']) zone.addEventListener(type, (event) => {
  event.preventDefault();
  zone.classList.add('dragging');
});
for (const type of ['dragleave', 'drop']) zone.addEventListener(type, (event) => {
  event.preventDefault();
  zone.classList.remove('dragging');
});
zone.addEventListener('drop', (event) => selectFile(event.dataTransfer.files[0]));
// Dropping outside the upload area must not navigate away from a conversion.
window.addEventListener('dragover', (event) => event.preventDefault());
window.addEventListener('drop', (event) => event.preventDefault());
$('cancel').addEventListener('click', () => {
  stop();
  status('Conversion cancelled. You can try again with a smaller processing size.');
});
$('form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!selectedFile || busy) return;
  clearResult();
  busy = true;
  const file = selectedFile;
  $('convert').disabled = true;
  $('cancel').hidden = false;
  $('size').disabled = true;
  $('background').disabled = true;
  const started = performance.now();
  $('elapsed').textContent = '0s';
  timer = setInterval(() => { $('elapsed').textContent = `${Math.floor((performance.now() - started) / 1000)}s`; }, 1000);
  status('Loading the converter…');
  try {
    const active = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    worker = active;
    const fail = (message) => {
      if (worker !== active) return;
      stop();
      status(message, true);
    };
    active.onerror = () => fail('The converter could not run. Check your connection and reload. If you host this site, build docs/pkg first.');
    active.onmessage = ({ data }) => {
      if (worker !== active) return;
      if (data.type === 'started') {
        status('Converting… Detailed images may take several minutes.');
      } else if (data.type === 'error') {
        fail(`Conversion failed: ${data.message}`);
      } else if (data.type === 'result') {
        const blob = new Blob([data.svg], { type: 'image/svg+xml' });
        resultURL = URL.createObjectURL(blob);
        $('result').src = resultURL;
        $('result').hidden = false;
        $('result-placeholder').hidden = true;
        $('result-info').textContent = formatBytes(blob.size);
        if (originalURL) URL.revokeObjectURL(originalURL);
        originalURL = URL.createObjectURL(file);
        $('original').src = originalURL;
        $('original').hidden = false;
        $('original-placeholder').hidden = true;
        $('download').href = resultURL;
        $('download').download = `${file.name.replace(/\.[^.]+$/, '') || 'image'}.svg`;
        $('download').hidden = false;
        $('elapsed').textContent = `${((performance.now() - started) / 1000).toFixed(1)}s`;
        stop();
        status('Your SVG is ready. Download it and make it your own.');
      }
    };
    const bytes = await file.arrayBuffer();
    if (worker !== active) return;
    active.postMessage({ bytes, size: Number($('size').value), removeBackground: $('background').checked }, [bytes]);
  } catch (error) {
    stop();
    status(`Could not start conversion: ${error.message}`, true);
  }
});
if (!supported) {
  $('file').disabled = true;
  status('This converter needs a browser with WebAssembly and Web Worker support.', true);
}
