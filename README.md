# Od szumu do obrazu / From noise to picture

Interactive material for part 1b ("Generatywne AI") of the AI workshop: how a diffusion model
generates an image, and how it is trained. Polish first, English second. edulab look.
Trainer drives it on a projector (10-15 min), participants can open it on their phones.

Live: https://workszop.github.io/difu/

Open `index.html` – no build step, no server, no API calls, works offline and from `file://`.

## Views

**Generowanie** – real per-step outputs of Stable Diffusion 1.5, precomputed. For every step:
"Teraz" (the current noisy image) and "Co model przewiduje" (the model's guess of the final
image, x0). 3 prompts (kot w kapeluszu, latarnia morska o zachodzie, miska z owocami); seed, guidance and steps switches; slider, play/pause, step strip.

**Trening** – one training example on a real photo (add noise at t, the target is the noise);
a tiny diffusion model (9,154 parameters, 3-layer MLP in plain JS in `toy.js`) trained live in the browser on a
2-D dot shape (heart / fish / letter Q). A table maps it to Stable Diffusion (one dot = one example, the shape = what correct examples look like). During training a preview every 25 epochs shows what the model would generate, with two charts: the loss (flattens fast, the noise is random so it cannot reach zero) and the share of dots on the shape (keeps rising, ~1% → 94-97%). Then 1000 dots are generated from noise;
pre-trained weights as a fallback for slow devices; a sourced scale table; a fold on training data
and copyright.

## Model and settings (Generowanie)

| | |
|---|---|
| Model | [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Licence | [CreativeML OpenRAIL-M](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/LICENSE.md) – allows this use; its use restrictions apply |
| Scheduler | DDIM, eta = 0, `timestep_spacing="trailing"` (step 0 = pure noise) |
| Resolution | generated 512×512, stored 320×320 AVIF (noisy frames q35, predictions q50) |
| Default run | 20 steps; "kot w kapeluszu" also at 4 and 50 steps |
| Seeds | 11, 42, 2024 (at guidance 7.5) |
| Guidance | low 2.0 / medium 7.5 / high 15.0 (at seed 11) |
| Negative prompt | `blurry, low quality, watermark, text artifacts` |
| Run | diffusers 0.42.1, torch 2.11, NVIDIA A100 (Colab), 2026-11-06 |
| Size | 17 cases, 5.9 MB |

Seed and guidance vary one at a time (not every seed × every guidance), which keeps the set under
15 MB. `GRID = "full"` in the notebook makes all combinations at about twice the size.

## Regenerating the assets

1. **Frames:** open `tools/generate_frames.ipynb` in Colab (GPU runtime) → Run all → download
   `diffusion_frames.zip` into the repo root. Edit the notebook through `tools/build_notebook.py`
   (`python3 tools/build_notebook.py` rewrites the `.ipynb`).
2. **Pack:** `unzip diffusion_frames.zip -d tools/out && python3 tools/pack_frames.py` converts to
   AVIF, writes `assets/gen/` + `assets/gen/cases.js`, and checks the invariants (every frame present,
   step 0 noisier than the last step, noise schedule ends at 0, budget).
3. **Toy weights:** `node tools/pretrain_toy.js` → `assets/toy-weights.js` (same code and settings
   as the live training).

## Verification

`tools/verify.mjs` checks the DOM contract and invariants with Playwright: start at noise, last
step shows the final image, arrows / space, seed / guidance / steps switches, every case decodes,
PL/EN, live training < ~30 s with loss falling, ≥ 90 % of dots on the shape and the share rising during training, pre-trained
fallback, 390 px without horizontal scroll, reduced motion shows the final state at once, offline
over http and `file://`.

```bash
python3 -m http.server 8817 &
PLAYWRIGHT=/path/to/playwright/index.mjs CHROME=/usr/bin/google-chrome node tools/verify.mjs 8817
```

DOM contract: `#gen[data-view data-case data-step data-steps data-seed data-guidance data-prompt
data-playing data-noise]`, `#toy[data-phase(idle|training|sampling|done) data-epoch data-loss
data-loss-start data-shape data-source data-sample-t]`, `#fwd[data-t data-noise]`.

## Sources for the scale table

1. [CompVis Stable Diffusion v1 README](https://github.com/CompVis/stable-diffusion#stable-diffusion-v1) – 860M UNet, 123M text encoder
2. [SD v1-5 model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) – LAION-2B(en) subsets, steps per checkpoint, 32×8 A100, batch 2048, 150,000 A100 hours (whole v1 family, authors' estimate)
3. [LAION-5B](https://laion.ai/blog/laion-5b/) – 5.85 billion pairs, 2.3 billion in English

The "Skąd się biorą dane?" fold names lawsuits and AI Act Art. 53 duties in general terms;
check the legal status before each workshop.

## Credits

Cat photo: Marko Milivojevic (pixnio.com) via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Domestic_shorthair_cat_portrait_in_grass.jpg), CC0.
Icons: [Lucide](https://lucide.dev) (ISC), inlined.
