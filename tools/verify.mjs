// DOM-contract + invariant checks for the app (1440px, 390px, keyboard, reduced motion, offline, file://).
// Needs Playwright: PLAYWRIGHT=/path/to/playwright/index.mjs CHROME=/usr/bin/google-chrome node tools/verify.mjs <port> [gen|train|phone|reduced|offline]
// Serve the repo root first: python3 -m http.server <port>
const { chromium } = await import(process.env.PLAYWRIGHT || 'playwright');
const PORT = process.argv[2], ONLY = process.argv[3] || 'all';
const OUT = process.env.OUT || '/tmp/';
const APP = `http://localhost:${PORT}/index.html`;
const FILE = new URL('../index.html', import.meta.url).href;
const results = [];
const check = (name, ok, info = '') => { results.push({ name, ok, info }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${info ? '  - ' + info : ''}`); };
const browser = await chromium.launch({ executablePath: process.env.CHROME || undefined, args: ['--no-sandbox'] });

async function open(url, opts = {}) {
  const ctx = await browser.newContext({ viewport: opts.viewport || { width: 1440, height: 900 }, reducedMotion: opts.reduced ? 'reduce' : 'no-preference', deviceScaleFactor: 1 });
  if (opts.offline) await ctx.route(u => !u.href.startsWith('http://localhost') && !u.href.startsWith('file:') && !u.href.startsWith('data:'), r => r.abort());
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  await page.goto(url);
  await page.waitForFunction(() => document.documentElement.classList.contains('app-ready'), null, { timeout: 8000 });
  return { ctx, page, errors };
}
const gen = page => page.evaluate(() => ({ ...document.querySelector('#gen').dataset }));
const toy = page => page.evaluate(() => ({ ...document.querySelector('#toy').dataset }));
const imgsOk = page => page.evaluate(() => [...document.querySelectorAll('#img-now, #img-pred:not([hidden])')].every(i => i.complete && i.naturalWidth > 0));

if (ONLY === 'all' || ONLY === 'gen') {
  const { ctx, page, errors } = await open(APP);
  let g = await gen(page);
  check('gen contract attrs present', ['case', 'step', 'seed', 'guidance'].every(k => g[k] !== undefined && g[k] !== ''), JSON.stringify(g));
  check('starts at step 0 (noise)', g.step === '0' && +g.noise >= 99, `step=${g.step} noise=${g.noise}`);
  check('pred empty at step 0', await page.evaluate(() => !document.querySelector('#pred-empty').hidden && document.querySelector('#img-pred').hidden));
  await page.waitForTimeout(400);
  check('frame images load', await imgsOk(page));
  await page.screenshot({ path: OUT + 'app_gen_1440_step0.png' });
  await page.keyboard.press('ArrowRight'); await page.keyboard.press('ArrowRight');
  g = await gen(page); check('ArrowRight moves step', g.step === '2', `step=${g.step}`);
  await page.keyboard.press('ArrowLeft'); g = await gen(page); check('ArrowLeft moves step', g.step === '1');
  await page.keyboard.press(' '); g = await gen(page); check('space starts playback', g.playing === 'true');
  await page.waitForFunction(() => document.querySelector('#gen').dataset.playing === 'false', null, { timeout: 10000 });
  g = await gen(page); check('playback ends at last step', g.step === g.steps && g.noise === '0', `step=${g.step}/${g.steps} noise=${g.noise}`);
  const last = await page.evaluate(() => document.querySelector('#img-now').getAttribute('src'));
  check('last step shows the final image c{N}', /\/c\d{2}\.(webp|avif)$/.test(last) && last.includes(`/c${String(g.steps).padStart(2, '0')}.`), last);
  await page.waitForTimeout(300);
  check('images load at last step', await imgsOk(page));
  await page.screenshot({ path: OUT + 'app_gen_1440_final.png' });
  // seed → guidance mid, new image
  await page.click('#seeds button:nth-child(2)'); g = await gen(page);
  check('seed switch', g.seed === '42' && g.guidance === 'mid' && g.case.includes('s42'), g.case);
  check('settings change replays from noise', g.playing === 'true' && +g.step <= 1, `step=${g.step} playing=${g.playing}`);
  check('seed aha highlighted', await page.evaluate(() => document.querySelector('.aha.on')?.dataset.aha === 'seed'));
  await page.click('#guids button:nth-child(3)'); g = await gen(page);
  check('guidance switch resets seed', g.guidance === 'high' && g.seed === '11', g.case);
  check('steps switch hidden for the fruit bowl', await page.evaluate(() => document.querySelector('#grp-steps').hidden));
  await page.click('#prompts button:nth-child(1)');
  check('steps switch shown for the cat, all 3 options', await page.evaluate(() => !document.querySelector('#grp-steps').hidden && [...document.querySelectorAll('#stepsSeg button')].every(b => !b.hidden)));
  await page.click('#stepsSeg button:nth-child(1)'); g = await gen(page);
  check('4-step case', g.case === 'kot_s11_gmid_n4' && g.steps === '4', g.case);
  await page.click('#stepsSeg button:nth-child(3)'); g = await gen(page);
  check('50-step case + strip of 51', g.steps === '50' && await page.evaluate(() => document.querySelectorAll('#strip .thumb').length === 51));
  await page.click('#strip .thumb:nth-child(11)'); g = await gen(page); check('thumb click jumps', g.step === '10');
  // every case: step 0 + last frames exist and decode
  const bad = await page.evaluate(async () => {
    const out = [];
    for (const c of window.DIFFUSION_DATA.cases) for (const f of ['c00', `c${String(c.steps).padStart(2, '0')}`, `p${String(c.steps).padStart(2, '0')}`]) {
      const im = new Image(); im.src = `assets/gen/${c.id}/${f}.${window.DIFFUSION_DATA.format || 'webp'}`;
      try { await im.decode(); } catch { out.push(c.id + '/' + f); }
    }
    return out;
  });
  check('all cases: first/last frames decode in browser', bad.length === 0, bad.slice(0, 3).join(','));
  // language
  await page.click('[data-lang=en]');
  check('EN switch', await page.evaluate(() => document.documentElement.lang === 'en' && document.querySelector('.app-title').textContent === 'From noise to picture'));
  await page.click('[data-lang=pl]');
  check('no console errors (gen)', errors.length === 0, errors.join(' | '));
  await ctx.close();
}

if (ONLY === 'all' || ONLY === 'train') {
  const { ctx, page, errors } = await open(APP + '#train');
  let d = await toy(page);
  check('toy contract attrs', d.phase === 'idle' && d.epoch === '0', JSON.stringify(d));
  check('forward card data-t', await page.evaluate(() => document.querySelector('#fwd').dataset.t === '0.40'));
  await page.fill('#fwd-slider', '100');
  check('t=1 → 100% noise', await page.evaluate(() => document.querySelector('#fwd').dataset.noise === '100'));
  await page.fill('#fwd-slider', '40');
  // untrained sampling
  await page.click('#btn-sample');
  await page.waitForFunction(() => document.querySelector('#toy').dataset.phase === 'done', null, { timeout: 15000 });
  d = await toy(page); check('raw sampling works', d.source === 'untrained', d.source);
  const t0 = Date.now();
  await page.click('#btn-train');
  await page.waitForFunction(() => document.querySelector('#toy').dataset.phase === 'training', null, { timeout: 3000 });
  await page.waitForTimeout(1500);
  await page.screenshot({ path: OUT + 'app_train_1440_training.png', fullPage: false });
  await page.waitForFunction(() => document.querySelector('#toy').dataset.phase === 'sampling', null, { timeout: 90000 });
  const trainS = (Date.now() - t0) / 1000;
  d = await toy(page);
  check('live training under ~30 s', trainS < 32, `${trainS.toFixed(1)} s`);
  check('loss after training < loss at start', +d.loss < +d.lossStart, `${d.lossStart} → ${d.loss}`);
  check('all epochs done', d.epoch === String(await page.evaluate(() => TOY_EPOCHS)), d.epoch);
  await page.waitForFunction(() => document.querySelector('#toy').dataset.phase === 'done', null, { timeout: 15000 });
  d = await toy(page); check('sampling done, t=0', d.sampleT === '0');
  d = await toy(page); check('≥90% generated dots on the shape (data-quality)', +d.quality >= 90, d.quality + '%');
  check('quality rose during training', await page.evaluate(() => state.toy.quality.length >= 10 && state.toy.quality[0][1] < 0.2));
  await page.locator('#toy').scrollIntoViewIfNeeded();
  await page.screenshot({ path: OUT + 'app_train_1440_toy.png' });
  // pretrained fish
  await page.click('#shapes button:nth-child(2)');
  await page.click('#btn-pre');
  await page.waitForFunction(() => document.querySelector('#toy').dataset.phase === 'done', null, { timeout: 15000 });
  d = await toy(page); check('pretrained fallback', d.source === 'pretrained' && d.shape === 'fish' && +d.loss < +d.lossStart, `${d.lossStart} → ${d.loss}`);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: OUT + 'app_train_1440_top.png' });
  await page.screenshot({ path: OUT + 'app_train_1440_full.png', fullPage: true });
  check('no console errors (train)', errors.length === 0, errors.join(' | '));
  await ctx.close();
}

if (ONLY === 'all' || ONLY === 'phone') {
  for (const v of ['gen', 'train']) {
    const { ctx, page, errors } = await open(APP + '#' + v, { viewport: { width: 390, height: 844 } });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    check(`390px ${v}: no horizontal scroll`, overflow <= 0, `overflow=${overflow}`);
    await page.screenshot({ path: OUT + `app_${v}_390.png` });
    await page.screenshot({ path: OUT + `app_${v}_390_full.png`, fullPage: true });
    check(`390px ${v}: no errors`, errors.length === 0, errors.join(' | '));
    await ctx.close();
  }
}

if (ONLY === 'all' || ONLY === 'reduced') {
  const { ctx, page } = await open(APP, { reduced: true });
  let g = await gen(page);
  check('reduced motion: final state at once', g.step === g.steps, `${g.step}/${g.steps}`);
  await page.click('#play'); g = await gen(page);
  check('reduced motion: play jumps to the end', g.step === g.steps && g.playing === 'false');
  await page.click('[data-tab=train]');
  await page.click('#btn-pre');
  check('reduced motion: sampling final at once', (await toy(page)).phase === 'done');
  await ctx.close();
}

if (ONLY === 'all' || ONLY === 'offline') {
  for (const url of [APP, FILE]) {
    const { ctx, page, errors } = await open(url, { offline: true });
    await page.waitForTimeout(500);
    const ok = await imgsOk(page);
    await page.click('[data-tab=train]');
    await page.waitForTimeout(300);
    const photo = await page.evaluate(() => { const d = document.querySelector('#cv-noisy').getContext('2d').getImageData(160, 160, 1, 1).data; return d[3] === 255; });
    check(`offline ${url.startsWith('file') ? 'file://' : 'http'}: frames + photo canvas work`, ok && photo, errors.filter(e => !/fonts|ERR_FAILED|net::/.test(e)).join(' | '));
    await ctx.close();
  }
}

await browser.close();
const failed = results.filter(r => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} passed`);
process.exit(failed.length ? 1 : 0);
