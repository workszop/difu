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
  const SIZES = [[IN, H], [H, H], [H, H], [H, 2]];   // 3 hidden layers, output = predicted noise (x, y)
  const N_PARAMS = SIZES.reduce((s, [a, b]) => s + a * b + b, 0);
  const DATA_N = 2048, BATCH = 128, ITERS_PER_EPOCH = DATA_N / BATCH;
  const NEAR = 0.08;                   // a sampled point "is on the shape" within this distance

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
  // share of points within NEAR of the shape (how good the samples are)
  const refs = {};
  function quality(pts, shape) {
    const ref = refs[shape] || (refs[shape] = sampleShape(shape, 1200, 99)), lim = NEAR * NEAR;
    let near = 0;
    for (let i = 0; i < pts.length; i += 2) {
      for (let j = 0; j < ref.length; j += 2) {
        const dx = pts[i] - ref[j], dy = pts[i + 1] - ref[j + 1];
        if (dx * dx + dy * dy < lim) { near++; break; }
      }
    }
    return near / (pts.length / 2);
  }

  // ─── Model: MLP with ReLU; W stored [out][in] so the inner loop is contiguous ───
  function initParams(seed) {
    const r = rng(seed);
    return SIZES.map(([a, n]) => {
      const W = new Float32Array(a * n), s = Math.sqrt(2 / a);
      for (let i = 0; i < W.length; i++) W[i] = gaussian(r) * s;
      return { a, n, W, b: new Float32Array(n) };
    });
  }
  function embed(x, y, t, out, o) {
    out[o] = x; out[o + 1] = y;
    const tt = t / T;
    for (let k = 0; k < N_FREQ; k++) {
      const w = Math.PI * (1 << k) * tt;
      out[o + 2 + 2 * k] = Math.sin(w); out[o + 3 + 2 * k] = Math.cos(w);
    }
  }
  function forward(layers, X, n) {
    const acts = [X];
    let cur = X;
    layers.forEach((L, li) => {
      const Y = new Float32Array(n * L.n), last = li === layers.length - 1;
      for (let i = 0; i < n; i++) {
        const xo = i * L.a, yo = i * L.n;
        for (let j = 0; j < L.n; j++) {
          let s = L.b[j]; const wo = j * L.a;
          for (let k = 0; k < L.a; k++) s += cur[xo + k] * L.W[wo + k];
          Y[yo + j] = last || s > 0 ? s : 0;
        }
      }
      acts.push(Y); cur = Y;
    });
    return acts;
  }
  function predictNoise(layers, x, t, n, X) {
    for (let i = 0; i < n; i++) embed(x[2 * i], x[2 * i + 1], t, X, i * IN);
    const acts = forward(layers, X, n);
    return acts[acts.length - 1];
  }

  // ─── Trainer (Adam), stepped from outside so the UI can stay responsive ───
  function createTrainer({ shape = 'heart', seed = 1 } = {}) {
    const layers = initParams(seed), r = rng(seed * 7 + 3), data = sampleShape(shape, DATA_N, seed + 11);
    const opt = layers.map(L => ({ mW: new Float32Array(L.W.length), vW: new Float32Array(L.W.length), mb: new Float32Array(L.n), vb: new Float32Array(L.n) }));
    const X = new Float32Array(BATCH * IN), target = new Float32Array(BATCH * 2);
    let step = 0;

    function iterate(lr) {
      for (let i = 0; i < BATCH; i++) {          // one training example = noisy point + t → the noise we added
        const d = Math.floor(r() * DATA_N), t = Math.floor(r() * T), ab = alphaBar[t];
        const e1 = gaussian(r), e2 = gaussian(r);
        embed(Math.sqrt(ab) * data[2 * d] + Math.sqrt(1 - ab) * e1, Math.sqrt(ab) * data[2 * d + 1] + Math.sqrt(1 - ab) * e2, t, X, i * IN);
        target[2 * i] = e1; target[2 * i + 1] = e2;
      }
      const acts = forward(layers, X, BATCH), out = acts[acts.length - 1];
      let loss = 0, g = new Float32Array(BATCH * 2);
      for (let i = 0; i < BATCH * 2; i++) { const d = out[i] - target[i]; loss += d * d; g[i] = 2 * d / (BATCH * 2); }
      step++;
      const c1 = 1 - 0.9 ** step, c2 = 1 - 0.999 ** step;
      for (let li = layers.length - 1; li >= 0; li--) {
        const L = layers[li], A = acts[li], o = opt[li];
        const gW = new Float32Array(L.W.length), gb = new Float32Array(L.n), gIn = li ? new Float32Array(BATCH * L.a) : null;
        for (let i = 0; i < BATCH; i++) {
          const ao = i * L.a, go = i * L.n;
          for (let j = 0; j < L.n; j++) {
            const gj = g[go + j]; if (gj === 0) continue;
            gb[j] += gj; const wo = j * L.a;
            for (let k = 0; k < L.a; k++) {
              gW[wo + k] += A[ao + k] * gj;
              if (gIn) gIn[ao + k] += L.W[wo + k] * gj;
            }
          }
        }
        if (gIn) for (let i = 0; i < gIn.length; i++) if (A[i] <= 0) gIn[i] = 0;   // ReLU derivative
        for (const [P, G, M, V] of [[L.W, gW, o.mW, o.vW], [L.b, gb, o.mb, o.vb]]) {
          for (let i = 0; i < P.length; i++) {
            M[i] = 0.9 * M[i] + 0.1 * G[i];
            V[i] = 0.999 * V[i] + 0.001 * G[i] * G[i];
            P[i] -= lr * (M[i] / c1) / (Math.sqrt(V[i] / c2) + 1e-8);
          }
        }
        g = gIn;
      }
      return loss / (BATCH * 2);
    }

    // one epoch = DATA_N / BATCH iterations; returns the mean loss
    function epoch(lr) {
      let s = 0;
      for (let i = 0; i < ITERS_PER_EPOCH; i++) s += iterate(lr);
      return s / ITERS_PER_EPOCH;
    }
    return { epoch, layers, data, get step() { return step; } };
  }

  // ─── Sampling: start from a Gaussian cloud, remove predicted noise T times (DDPM) ───
  function createSampler(layers, n, seed) {
    const r = rng(seed), x = new Float32Array(n * 2), X = new Float32Array(n * IN);
    for (let i = 0; i < n * 2; i++) x[i] = gaussian(r);
    let t = T - 1;
    function step() {
      if (t < 0) return false;
      const eps = predictNoise(layers, x, t, n, X);
      const c = betas[t] / Math.sqrt(1 - alphaBar[t]), s = 1 / Math.sqrt(alphas[t]), sig = t > 0 ? Math.sqrt(betas[t]) : 0;
      for (let i = 0; i < n * 2; i++) x[i] = s * (x[i] - c * eps[i]) + sig * gaussian(r);
      t--;
      return true;
    }
    return { step, points: x, get t() { return t; } };
  }
  // fast preview while training: deterministic DDIM over a few of the T steps
  function quickSample(layers, n, seed, steps = 20) {
    const r = rng(seed), x = new Float32Array(n * 2), X = new Float32Array(n * IN);
    for (let i = 0; i < n * 2; i++) x[i] = gaussian(r);
    const ts = Array.from({ length: steps }, (_, i) => Math.round((T - 1) * (1 - i / steps)));
    ts.forEach((t, i) => {
      const eps = predictNoise(layers, x, t, n, X), a = alphaBar[t], tp = ts[i + 1], ap = tp === undefined ? 1 : alphaBar[tp];
      for (let k = 0; k < n * 2; k++) {
        const x0 = (x[k] - Math.sqrt(1 - a) * eps[k]) / Math.sqrt(a);
        x[k] = Math.sqrt(ap) * x0 + Math.sqrt(1 - ap) * eps[k];
      }
    });
    return x;
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
    const flat = new Float32Array(bytes.buffer); let o = 0;
    return SIZES.map(([a, n]) => {
      const W = flat.slice(o, o + a * n); o += a * n;
      const b = flat.slice(o, o + n); o += n;
      return { a, n, W, b };
    });
  }

  const Toy = { T, N_PARAMS, DATA_N, BATCH, ITERS_PER_EPOCH, alphaBar, createTrainer, createSampler, quickSample, quality, sampleShape, initParams, packWeights, unpackWeights, rng, gaussian };
  if (typeof module !== 'undefined' && module.exports) module.exports = Toy; else root.Toy = Toy;
})(typeof window !== 'undefined' ? window : globalThis);
