const version = new URL(self.location.href).searchParams.get('v');
const bindingsURL = new URL('./pkg/picvec.js', import.meta.url);
const wasmURL = new URL('./pkg/picvec_bg.wasm', import.meta.url);
bindingsURL.searchParams.set('v', version);
wasmURL.searchParams.set('v', version);

self.onmessage = async ({ data }) => {
  try {
    const { default: init, convert_image } = await import(bindingsURL.href);
    await init({ module_or_path: wasmURL });
    self.postMessage({ type: 'started' });
    const svg = convert_image(new Uint8Array(data.bytes), data.size, data.removeBackground);
    self.postMessage({ type: 'result', svg });
  } catch (error) {
    self.postMessage({ type: 'error', message: String(error?.message ?? error) });
  }
};
