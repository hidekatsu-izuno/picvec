import init, { convert_image } from './pkg/picvec.js';

self.onmessage = async ({ data }) => {
  try {
    await init();
    self.postMessage({ type: 'started' });
    const svg = convert_image(new Uint8Array(data.bytes), data.size, data.removeBackground);
    self.postMessage({ type: 'result', svg });
  } catch (error) {
    self.postMessage({ type: 'error', message: String(error?.message ?? error) });
  }
};
