/**
 * AudioWorklet processor – captures mic audio on a dedicated audio thread,
 * downsamples to 16 kHz, converts to 16-bit PCM, and posts the ArrayBuffer
 * to the main thread for WebSocket transmission.
 *
 * Runs every 128 frames (~2.67 ms at 48 kHz).  We accumulate samples and
 * flush roughly every 20 ms to keep chunk size small for low latency.
 */
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // Accumulate ~20 ms of audio at the native sample rate before sending
    this._sendSize = Math.floor(sampleRate * 0.02);
    this._buf = new Float32Array(this._sendSize);
    this._pos = 0;
    this._ratio = sampleRate / 16000;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || !input[0].length) return true;

    const src = input[0]; // mono channel
    let srcOff = 0;

    while (srcOff < src.length) {
      const space = this._sendSize - this._pos;
      const toCopy = Math.min(space, src.length - srcOff);
      this._buf.set(src.subarray(srcOff, srcOff + toCopy), this._pos);
      this._pos += toCopy;
      srcOff += toCopy;

      if (this._pos >= this._sendSize) {
        this._flush();
      }
    }

    return true;
  }

  /** Downsample the accumulated buffer to 16 kHz PCM-16 and post it. */
  _flush() {
    const outLen = Math.round(this._pos / this._ratio);
    const pcm16 = new Int16Array(outLen);

    for (let i = 0; i < outLen; i++) {
      const idx = Math.min(Math.round(i * this._ratio), this._pos - 1);
      const s = Math.max(-1, Math.min(1, this._buf[idx]));
      pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }

    // Transfer (zero-copy) the underlying ArrayBuffer to the main thread
    this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
    this._pos = 0;
  }
}

registerProcessor("pcm-processor", PCMProcessor);
