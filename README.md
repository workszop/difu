# Od szumu do obrazu / From noise to picture

Interactive material for part 1b ("Generatywne AI") of the AI workshop: how a diffusion model
generates an image, and how it is trained. Polish first, English second. edulab look.
Trainer drives it on a projector (10-15 min), participants can open it on their phones.

Live: https://workszop.github.io/difu/

Open `index.html` – no build step, no server, no API calls, works offline and from `file://`.

## Views

**Generowanie** – real per-step outputs of Stable Diffusion 1.5, precomputed. For every step:
"Teraz" (the current noisy image) and "Co model przewiduje" (the model's guess of the final
image, x0). 3 prompts (kot w kapeluszu, logo piekarni, miska z owocami); seed, guidance and steps switches; slider, play/pause, step strip.

**Trening** – one training example on a real photo (add noise at t, the target is the noise);
a simple illustrated walk through how Stable Diffusion works (text encoder, latent map, denoising loop, decoder) on real frames; a fold on training data and copyright.

## Model and settings (Generowanie)

| | |
|---|---|
| Model | [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| Licence | [CreativeML OpenRAIL-M](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/LICENSE.md) – allows this use; its use restrictions apply |
| Scheduler | DDIM, eta = 0, `timestep_spacing="trailing"` (step 0 = pure noise) |
| Resolution | generated 512×512, stored 50×50 WebP, shown pixelated (`image-rendering: pixelated`) so the noise reads as noise |
| Default run | 20 steps; "kot w kapeluszu" also at 4 and 50 steps |
| Seeds | 11, 42, 2024 (at guidance 7.5) |
| Guidance | low 2.0 / medium 7.5 / high 15.0 (at seed 11) |
| Negative prompt | `blurry, low quality, watermark, text artifacts` |
| Run | diffusers 0.42.1, torch 2.14, NVIDIA A100 (Colab), 2026-11-07 |
| Size | 17 cases, 0.69 MB |

Seed and guidance vary one at a time (not every seed × every guidance), which keeps the set under
15 MB. `GRID = "full"` in the notebook makes all combinations at about twice the size.

## Regenerating the assets

1. **Frames:** open `tools/generate_frames.ipynb` in Colab (GPU runtime) → Run all → download
   `diffusion_frames.zip` into the repo root. Edit the notebook through `tools/build_notebook.py`
   (`python3 tools/build_notebook.py` rewrites the `.ipynb`).
2. **Install:** the zip already holds the final `assets/gen/` (WebP frames + `cases.js`); the notebook
   checks the invariants (every frame present, step 0 noisier than the last step, noise schedule ends
   at 0, budget). `rm -r assets/gen && unzip diffusion_frames.zip && rm assets/gen/_meta.json`.
   (`tools/pack_frames.py` is the older AVIF route for large frames; not needed at 50 px.)

## Verification

`tools/verify.mjs` checks the DOM contract and invariants with Playwright: start at noise, last
step shows the final image, arrows / space, seed / guidance / steps switches, every case decodes,
PL/EN, the training example (noise slider), 390 px without horizontal scroll, reduced motion shows the final state at once, offline
over http and `file://`.

```bash
python3 -m http.server 8817 &
PLAYWRIGHT=/path/to/playwright/index.mjs CHROME=/usr/bin/google-chrome node tools/verify.mjs 8817
```

DOM contract: `#gen[data-view data-case data-step data-steps data-seed data-guidance data-prompt
data-playing data-noise]`, `#fwd[data-t data-noise]`.


## Credits

Cat photo: Marko Milivojevic (pixnio.com) via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Domestic_shorthair_cat_portrait_in_grass.jpg), CC0.
Icons: [Lucide](https://lucide.dev) (ISC), inlined.
