const $ = (id) => document.getElementById(id);
let selectedFile;
let originalURL;
let resultURL;
let worker;
let timer;
let selection = 0;
const supported = 'WebAssembly' in window && 'Worker' in window;

function status(message, error = false) {
  $('status').textContent = message;
  $('status').parentElement.classList.toggle('error', error);
}
function setLoading(loading) {
  $('result-loading').hidden = !loading;
  $('result-panel').setAttribute('aria-busy', String(loading));
  $('result-placeholder').hidden = loading || !$('result').hidden;
}
function clearResult() {
  if (resultURL) URL.revokeObjectURL(resultURL);
  resultURL = undefined;
  $('result').removeAttribute('src');
  $('result').hidden = true;
  $('result-placeholder').hidden = false;
  $('result-placeholder').textContent = 'Your converted SVG will appear here.';
  $('download').hidden = true;
  $('download').removeAttribute('href');
  $('result-info').textContent = '';
}
function stop() {
  worker?.terminate();
  worker = undefined;
  clearInterval(timer);
  $('cancel').hidden = true;
  setLoading(false);
}
async function selectFile(file) {
  if (!file || !supported) return;
  const current = ++selection;
  const options = { size: Number($('size').value), removeBackground: $('background').checked };
  stop();
  selectedFile = undefined;
  clearResult();
  $('elapsed').textContent = '';
  $('filename').textContent = 'No image selected.';
  $('source-info').textContent = '';
  $('original').hidden = true;
  $('original').removeAttribute('src');
  $('original-placeholder').hidden = false;
  $('original-placeholder').textContent = 'Choose a PNG or JPEG to begin.';
  if (originalURL) URL.revokeObjectURL(originalURL);
  originalURL = undefined;
  if (file.size > 20 * 1024 * 1024) {
    status('Choose an image smaller than 20 MiB.', true);
    return;
  }
  setLoading(true);
  $('cancel').hidden = false;
  status('Checking image…');
  try {
    // Check the format before handing the local file to the browser preview.
    // The Wasm decoder independently validates source dimensions and limits.
    const bytes = new Uint8Array(await file.slice(0, 32).arrayBuffer());
    if (current !== selection) return;
    const png = [137, 80, 78, 71, 13, 10, 26, 10].every((value, i) => bytes[i] === value);
    const jpeg = bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255;
    if (!png && !jpeg) throw new Error('Please choose a PNG or JPEG image.');
    selectedFile = file;
    $('filename').textContent = file.name;
    $('source-info').textContent = formatBytes(file.size);
    originalURL = URL.createObjectURL(file);
    $('original').src = originalURL;
    $('original').hidden = false;
    $('original-placeholder').hidden = true;
    await convertImage(options);
  } catch (error) {
    if (current !== selection) return;
    stop();
    status(error.message, true);
  }
}
function formatBytes(size) {
  return size >= 1024 * 1024 ? `${(size / 1024 / 1024).toFixed(1)} MiB` : `${(size / 1024).toFixed(1)} KiB`;
}
$('file').addEventListener('change', (event) => {
  const file = event.target.files[0];
  event.target.value = '';
  void selectFile(file);
});
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
  ++selection;
  stop();
  $('result-placeholder').textContent = 'Conversion cancelled.';
  status('Conversion cancelled. Choose an image to try again.');
});
async function convertImage(options) {
  if (!selectedFile) return;
  stop();
  clearResult();
  setLoading(true);
  const file = selectedFile;
  $('cancel').hidden = false;
  const started = performance.now();
  $('elapsed').textContent = '0s';
  timer = setInterval(() => { $('elapsed').textContent = `${Math.floor((performance.now() - started) / 1000)}s`; }, 1000);
  status('Loading the converter…');
  let active;
  try {
    active = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    worker = active;
    const fail = (message) => {
      if (worker !== active) return;
      stop();
      $('result-placeholder').textContent = 'Conversion failed.';
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
        $('download').href = resultURL;
        $('download').download = `${file.name.replace(/\.[^.]+$/, '') || 'image'}.svg`;
        $('download').hidden = false;
        $('elapsed').textContent = `${((performance.now() - started) / 1000).toFixed(1)}s`;
        stop();
        status('Conversion complete. Click Download SVG to save.');
      }
    };
    const bytes = await file.arrayBuffer();
    if (worker !== active) return;
    active.postMessage({ bytes, ...options }, [bytes]);
  } catch (error) {
    if (active && worker !== active) return;
    stop();
    $('result-placeholder').textContent = 'Conversion failed.';
    status(`Could not start conversion: ${error.message}`, true);
  }
}
$('form').addEventListener('submit', (event) => event.preventDefault());
$('original').addEventListener('error', () => {
  $('original').hidden = true;
  $('original-placeholder').hidden = false;
  $('original-placeholder').textContent = 'Could not preview this image.';
});
if (!supported) {
  $('file').disabled = true;
  status('This converter needs a browser with WebAssembly and Web Worker support.', true);
}
