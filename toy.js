// Toy diffusion model on 2-D points, in plain JS (no libraries).
// Shared by the app (window.Toy) and by tools/pretrain_toy.js (Node), so the pre-trained
// fallback weights come from exactly this code.
(function (root) {
  'use strict';

  // ─── Constants ──────────────────────────────────
  const T = 100;                       // diffusion steps
  const N_FREQ = 4;                    // time embedding: sin/cos at 4 frequencies
  const IN = 2 + 2 * N_FREQ;           // x, y + time features
  const H = 64;                        // hidden width
  const OUT = 2;                       // predicted noise (x, y)
  const SIZES = [[IN, H], [H, H], [H, OUT]];
  const N_PARAMS = SIZES.reduce((s, [a, b]) => s + a * b + b, 0);
  const DATA_N = 2048, BATCH = 256, ITERS_PER_EPOCH = DATA_N / BATCH;

  const betas = new Float64Array(T), alphas = new Float64Array(T), alphaBar = new Float64Array(T);
  for (let t = 0; t < T; t++) {
    betas[t] = 1e-4 + (0.08 - 1e-4) * t / (T - 1);
    alphas[t] = 1 - betas[t];
    alphaBar[t] = (t ? alphaBar[t - 1] : 1) * alphas[t];
  }

  // ─── Random numbers (seeded, so runs are repeatable) ───
  function rng(seed) {
    let s = seed >>> 0 || 1;
    return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
  }
  function gaussian(r) {
    let u = 0; while (u === 0) u = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
  }

  // ─── Shapes (points along an outline, slightly jittered) ───
  function curvePoint(shape, u) {               // u in [0,1)
    const a = u * 2 * Math.PI;
    if (shape === 'heart') {
      return [16 * Math.sin(a) ** 3 / 17, (13 * Math.cos(a) - 5 * Math.cos(2 * a) - 2 * Math.cos(3 * a) - Math.cos(4 * a)) / 17 + 0.15];
    }
    if (shape === 'fish') {                     // body ellipse (70%) + tail triangle (30%)
      if (u < 0.7) { const b = u / 0.7 * 2 * Math.PI; return [0.15 + 0.6 * Math.cos(b), 0.38 * Math.sin(b)]; }
      const v = (u - 0.7) / 0.3, p = [[-0.45, 0], [-0.95, 0.38], [-0.95, -0.38]], seg = Math.min(2, Math.floor(v * 3)), f = v * 3 - seg;
      const A = p[seg], B = p[(seg + 1) % 3];
      return [A[0] + (B[0] - A[0]) * f, A[1] + (B[1] - A[1]) * f];
    }
    // letter Q: ring (80%) + tail stroke (20%)
    if (u < 0.8) { const b = u / 0.8 * 2 * Math.PI; return [0.62 * Math.cos(b), 0.05 + 0.72 * Math.sin(b)]; }
    const f = (u - 0.8) / 0.2;
    return [0.2 + 0.6 * f, -0.35 - 0.5 * f];
  }
  function sampleShape(shape, n, seed) {
    const r = rng(seed), pts = new Float64Array(n * 2);
    for (let i = 0; i < n; i++) {
      const [x, y] = curvePoint(shape, r());
      pts[2 * i] = x + 0.03 * gaussian(r);
      pts[2 * i + 1] = y + 0.03 * gaussian(r);
    }
    return pts;
  }

  // ─── Model: 3-layer MLP, ReLU ───
  function initParams(seed) {
    const r = rng(seed), layers = [];
    for (const [a, b] of SIZES) {
      const W = new Float64Array(a * b), bias = new Float64Array(b), s = Math.sqrt(2 / a);
      for (let i = 0; i < W.length; i++) W[i] = gaussian(r) * s;
      layers.push({ W, b: bias, a, n: b });
    }
    return layers;
  }
  function embed(x, y, t, out, o) {
    out[o] = x; out[o + 1] = y;
    const tt = t / T;
    for (let k = 0; k < N_FREQ; k++) {
      const w = Math.PI * (1 << k) * tt;
      out[o + 2 + 2 * k] = Math.sin(w); out[o + 3 + 2 * k] = Math.cos(w);
    }
  }
  // forward for a batch; keeps activations for backprop
  function forward(layers, X, n) {
    const acts = [X];
    let cur = X;
    layers.forEach((L, li) => {
      const Y = new Float64Array(n * L.n), last = li === layers.length - 1;
      for (let i = 0; i < n; i++) {
        const xo = i * L.a, yo = i * L.n;
        for (let j = 0; j < L.n; j++) {
          let s = L.b[j];
          for (let k = 0; k < L.a; k++) s += cur[xo + k] * L.W[k * L.n + j];
          Y[yo + j] = last ? s : (s > 0 ? s : 0);
        }
      }
      acts.push(Y); cur = Y;
    });
    return acts;
  }

  // ─── Trainer (Adam), stepped from outside so the UI can stay responsive ───
  function createTrainer({ shape = 'heart', seed = 1 } = {}) {
    const layers = initParams(seed), r = rng(seed * 7 + 3), data = sampleShape(shape, DATA_N, seed + 11);
    const m = layers.map(L => ({ W: new Float64Array(L.W.length), b: new Float64Array(L.b.length) }));
    const v = layers.map(L => ({ W: new Float64Array(L.W.length), b: new Float64Array(L.b.length) }));
    const X = new Float64Array(BATCH * IN), target = new Float64Array(BATCH * 2);
    let step = 0;

    function iterate(lr) {
      for (let i = 0; i < BATCH; i++) {          // one training example = noisy point + t → the noise we added
        const d = Math.floor(r() * DATA_N), t = Math.floor(r() * T), ab = alphaBar[t];
        const e1 = gaussian(r), e2 = gaussian(r);
        embed(Math.sqrt(ab) * data[2 * d] + Math.sqrt(1 - ab) * e1, Math.sqrt(ab) * data[2 * d + 1] + Math.sqrt(1 - ab) * e2, t, X, i * IN);
        target[2 * i] = e1; target[2 * i + 1] = e2;
      }
      const acts = forward(layers, X, BATCH), out = acts[acts.length - 1];
      let loss = 0, g = new Float64Array(BATCH * 2);
      for (let i = 0; i < BATCH * 2; i++) { const d = out[i] - target[i]; loss += d * d; g[i] = 2 * d / (BATCH * 2); }
      loss /= BATCH * 2;
      step++;
      const b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** step, c2 = 1 - b2 ** step;
      for (let li = layers.length - 1; li >= 0; li--) {
        const L = layers[li], A = acts[li], gW = new Float64Array(L.W.length), gb = new Float64Array(L.n);
        const gIn = li ? new Float64Array(BATCH * L.a) : null;
        for (let i = 0; i < BATCH; i++) {
          const ao = i * L.a, go = i * L.n;
          for (let j = 0; j < L.n; j++) {
            const gj = g[go + j]; if (gj === 0) continue;
            gb[j] += gj;
            for (let k = 0; k < L.a; k++) {
              gW[k * L.n + j] += A[ao + k] * gj;
              if (gIn) gIn[ao + k] += L.W[k * L.n + j] * gj;
            }
          }
        }
        if (gIn) for (let i = 0; i < gIn.length; i++) if (A[i] <= 0) gIn[i] = 0;   // ReLU derivative
        for (const [P, G, M, V] of [[L.W, gW, m[li].W, v[li].W], [L.b, gb, m[li].b, v[li].b]]) {
          for (let i = 0; i < P.length; i++) {
            M[i] = b1 * M[i] + (1 - b1) * G[i];
            V[i] = b2 * V[i] + (1 - b2) * G[i] * G[i];
            P[i] -= lr * (M[i] / c1) / (Math.sqrt(V[i] / c2) + 1e-8);
          }
        }
        g = gIn;
      }
      return loss;
    }

    // one epoch = DATA_N / BATCH iterations; returns the mean loss
    function epoch(lr) {
      let s = 0;
      for (let i = 0; i < ITERS_PER_EPOCH; i++) s += iterate(lr);
      return s / ITERS_PER_EPOCH;
    }
    return { epoch, layers, data, get step() { return step; } };
  }

  // ─── Sampling: start from a Gaussian cloud, remove predicted noise T times ───
  function createSampler(layers, n, seed) {
    const r = rng(seed), x = new Float64Array(n * 2), X = new Float64Array(n * IN);
    for (let i = 0; i < n * 2; i++) x[i] = gaussian(r);
    let t = T - 1;
    function step() {
      if (t < 0) return false;
      for (let i = 0; i < n; i++) embed(x[2 * i], x[2 * i + 1], t, X, i * IN);
      const acts = forward(layers, X, n), eps = acts[acts.length - 1];
      const c = betas[t] / Math.sqrt(1 - alphaBar[t]), s = 1 / Math.sqrt(alphas[t]), sig = t > 0 ? Math.sqrt(betas[t]) : 0;
      for (let i = 0; i < n * 2; i++) x[i] = s * (x[i] - c * eps[i]) + sig * gaussian(r);
      t--;
      return true;
    }
    return { step, points: x, get t() { return t; } };
  }

  // ─── Weights (de)serialisation: Float32 → base64 ───
  function packWeights(layers) {
    const flat = new Float32Array(N_PARAMS); let o = 0;
    for (const L of layers) { flat.set(L.W, o); o += L.W.length; flat.set(L.b, o); o += L.b.length; }
    const bytes = new Uint8Array(flat.buffer); let s = '';
    for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
    return typeof btoa === 'function' ? btoa(s) : Buffer.from(bytes).toString('base64');
  }
  function unpackWeights(b64) {
    const bin = typeof atob === 'function' ? atob(b64) : Buffer.from(b64, 'base64').toString('binary');
    const bytes = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const flat = new Float32Array(bytes.buffer), layers = []; let o = 0;
    for (const [a, b] of SIZES) {
      const W = Float64Array.from(flat.subarray(o, o + a * b)); o += a * b;
      const bias = Float64Array.from(flat.subarray(o, o + b)); o += b;
      layers.push({ W, b: bias, a, n: b });
    }
    return layers;
  }

  const Toy = { T, N_PARAMS, DATA_N, BATCH, alphaBar, createTrainer, createSampler, sampleShape, initParams, packWeights, unpackWeights, rng, gaussian };
  if (typeof module !== 'undefined' && module.exports) module.exports = Toy; else root.Toy = Toy;
})(typeof window !== 'undefined' ? window : globalThis);
