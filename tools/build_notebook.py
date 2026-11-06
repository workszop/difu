#!/usr/bin/env python3
"""Writes tools/generate_frames.ipynb (the Colab notebook) from the cell sources below.

Run:  python3 tools/build_notebook.py
Edit the cells here, not the .ipynb, so the notebook stays reviewable in git.
"""
import json
from pathlib import Path

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")})


def code(src):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip("\n")})


md(r"""
# Od szumu do obrazu – frame generator (Colab)

Generates the precomputed frames for the app's **Generowanie** view.
For every case and every denoising step it saves:

* `cNN.webp` – the current noisy image (latent after step NN, decoded by the VAE),
* `pNN.webp` – the model's prediction of the final image (x0) made during step NN.

**How to run:** `Runtime → Change runtime type → T4 GPU`, then `Runtime → Run all`.
About 15–20 minutes on a free T4. The last cell downloads `diffusion_frames.zip`.
Unzip it into the repo root (it contains `assets/gen/…` and `assets/gen/cases.js`).

If Colab disconnects, run all again: finished cases are skipped.

Model: `stable-diffusion-v1-5/stable-diffusion-v1-5` (CreativeML OpenRAIL-M).
Licence: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/LICENSE.md
""")

code(r"""
!pip install -q -U diffusers transformers accelerate
""")

md("## Settings")

code(r"""
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MODEL_LICENSE = "CreativeML OpenRAIL-M"
MODEL_LICENSE_URL = "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/LICENSE.md"

GEN_SIZE = 512          # SD 1.5 native resolution
OUT_SIZE = 320          # saved frame size (px); lower it if the budget check fails
WEBP_QUALITY = 62
BUDGET_MB = 15.0

DEFAULT_STEPS = 20
STEP_VARIANTS = [4, 50]          # extra runs for the first prompt (20 is the default run)
SEEDS = [11, 42, 2024]
GUIDANCE = {"low": 2.0, "mid": 7.5, "high": 15.0}

# "cross": seeds at mid guidance + guidance levels at the first seed (5 runs per prompt).
# "full": every seed x every guidance (9 runs per prompt) - about 2x the size.
GRID = "cross"

NEGATIVE = "blurry, low quality, watermark, text artifacts"

# id, Polish caption (UI), English caption (UI), prompt sent to the model
PROMPTS = [
    ("kot",      "kot w kapeluszu",              "a cat in a hat",
     "a photo of a ginger cat wearing a small straw hat, studio light, sharp focus"),
    ("latarnia", "latarnia morska o zachodzie",  "a lighthouse at sunset",
     "a lighthouse on a rocky cliff at sunset, warm sky, calm sea, oil painting"),
    ("logo",     "logo piekarni",                "a bakery logo",
     "a simple flat vector logo for a bakery, a loaf of bread and wheat, two colours, white background"),
    ("rower",    "czerwony rower w deszczu",     "a red bicycle in the rain",
     "a red bicycle leaning on a wall on a rainy city street, reflections, photo"),
    ("owoce",    "miska z owocami",              "a bowl of fruit",
     "a ceramic bowl with apples, pears and grapes on a wooden table, soft daylight, photo"),
    ("robot",    "robot czytający książkę",      "a robot reading a book",
     "a small friendly robot reading a book in a cosy library, illustration"),
]

OUT_DIR = "/content/out/assets/gen"
""")

md("## Load the model")

code(r"""
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

assert torch.cuda.is_available(), "No GPU - set Runtime > Change runtime type > T4 GPU"
DEVICE, DTYPE = "cuda", torch.float16

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID, torch_dtype=DTYPE, safety_checker=None, requires_safety_checker=False
).to(DEVICE)
# DDIM (deterministic, eta=0). "trailing" spacing starts from the last timestep (999),
# so step 0 is pure noise and very short runs (4 steps) still end at t≈0.
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
pipe.set_progress_bar_config(disable=True)
print(torch.cuda.get_device_name(0), "| diffusers", __import__("diffusers").__version__)
""")

md(r"""
## Denoising loop that keeps every step

This is the standard text-to-image loop (UNet + classifier-free guidance + scheduler),
written out by hand so we can keep the scheduler's `pred_original_sample` (the x0 guess),
which the stock pipeline throws away.
""")

code(r"""
import os, json, io
from PIL import Image

sched = pipe.scheduler
ALPHAS = sched.alphas_cumprod  # ᾱ_t on CPU


@torch.no_grad()
def decode(latents):
    imgs = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    imgs = (imgs / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu().numpy()
    return [Image.fromarray((im * 255).round().astype("uint8")) for im in imgs]


def save(img, path):
    img.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS).save(path, "WEBP", quality=WEBP_QUALITY, method=6)


@torch.no_grad()
def run_case(case_dir, prompt, seed, guidance, steps):
    os.makedirs(case_dir, exist_ok=True)
    cond, uncond = pipe.encode_prompt(prompt, DEVICE, 1, True, negative_prompt=NEGATIVE)
    embeds = torch.cat([uncond, cond])

    sched.set_timesteps(steps, device=DEVICE)
    g = torch.Generator(DEVICE).manual_seed(seed)
    lat_hw = GEN_SIZE // pipe.vae_scale_factor
    latents = torch.randn((1, pipe.unet.config.in_channels, lat_hw, lat_hw),
                          generator=g, device=DEVICE, dtype=DTYPE) * sched.init_noise_sigma

    timesteps = [int(t) for t in sched.timesteps]
    # noise share of the *current* image after k steps: 1 - ᾱ of the timestep it now sits at
    alphabar = [float(ALPHAS[timesteps[0]])]

    save(decode(latents)[0], f"{case_dir}/c00.webp")           # step 0: pure noise, no guess yet
    for k, t in enumerate(sched.timesteps, start=1):
        inp = sched.scale_model_input(torch.cat([latents] * 2), t)
        eps = pipe.unet(inp, t, encoder_hidden_states=embeds, return_dict=False)[0]
        eps_u, eps_c = eps.chunk(2)
        eps = eps_u + guidance * (eps_c - eps_u)
        out = sched.step(eps, t, latents)
        latents = out.prev_sample
        cur, pred = decode(torch.cat([latents, out.pred_original_sample]))
        save(cur, f"{case_dir}/c{k:02d}.webp")
        save(pred, f"{case_dir}/p{k:02d}.webp")
        alphabar.append(float(ALPHAS[timesteps[k]]) if k < steps else 1.0)  # last frame: clean
    return {"timesteps": timesteps, "alphabar": [round(a, 5) for a in alphabar]}
""")

md("## Run all cases")

code(r"""
import time

def case_list():
    cases = []
    s0 = SEEDS[0]
    for i, (pid, pl, en, prompt) in enumerate(PROMPTS):
        combos = set()
        if GRID == "full":
            combos = {(s, g) for s in SEEDS for g in GUIDANCE}
        else:
            combos = {(s, "mid") for s in SEEDS} | {(s0, g) for g in GUIDANCE}
        for s, g in sorted(combos, key=lambda c: (SEEDS.index(c[0]), list(GUIDANCE).index(c[1]))):
            cases.append(dict(prompt_id=pid, seed=s, guidance=g, steps=DEFAULT_STEPS))
        if i == 0:
            for n in STEP_VARIANTS:
                cases.append(dict(prompt_id=pid, seed=s0, guidance="mid", steps=n))
    for c in cases:
        c["id"] = f"{c['prompt_id']}_s{c['seed']}_g{c['guidance']}_n{c['steps']}"
    return cases

CASES = case_list()
print(len(CASES), "cases")
meta_path = f"{OUT_DIR}/_meta.json"
os.makedirs(OUT_DIR, exist_ok=True)
meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

t0 = time.time()
for i, c in enumerate(CASES, 1):
    if c["id"] in meta:
        continue
    prompt = next(p[3] for p in PROMPTS if p[0] == c["prompt_id"])
    meta[c["id"]] = run_case(f"{OUT_DIR}/{c['id']}", prompt, c["seed"], GUIDANCE[c["guidance"]], c["steps"])
    json.dump(meta, open(meta_path, "w"))
    print(f"{i:3d}/{len(CASES)} {c['id']:32s} {time.time() - t0:6.0f}s")
print("done")
""")

md("## Check invariants, write `cases.js`")

code(r"""
import glob, numpy as np

errors = []
total = 0
for c in CASES:
    d = f"{OUT_DIR}/{c['id']}"
    n = c["steps"]
    want = [f"c{k:02d}.webp" for k in range(n + 1)] + [f"p{k:02d}.webp" for k in range(1, n + 1)]
    missing = [w for w in want if not os.path.exists(f"{d}/{w}")]
    if missing:
        errors.append(f"{c['id']}: missing {missing[:3]}…")
    total += sum(os.path.getsize(f) for f in glob.glob(f"{d}/*.webp"))
    # step 0 must look like noise: high pixel-to-pixel variation; last step must not
    a0 = np.asarray(Image.open(f"{d}/c00.webp").convert("L"), dtype=float)
    aN = np.asarray(Image.open(f"{d}/c{n:02d}.webp").convert("L"), dtype=float)
    rough = lambda a: np.abs(np.diff(a, axis=1)).mean()
    c["roughness"] = [round(rough(a0), 1), round(rough(aN), 1)]
    if not rough(a0) > rough(aN):
        errors.append(f"{c['id']}: step 0 is not rougher than the last step {c['roughness']}")
    c.update(meta[c["id"]])

mb = total / 1e6
print(f"total {mb:.1f} MB in {len(CASES)} cases (budget {BUDGET_MB} MB)")
if mb > BUDGET_MB:
    errors.append(f"over budget: {mb:.1f} MB > {BUDGET_MB} MB - lower OUT_SIZE or WEBP_QUALITY")
print("\n".join(errors) or "all invariants OK")

manifest = {
    "model": MODEL_ID, "license": MODEL_LICENSE, "license_url": MODEL_LICENSE_URL,
    "scheduler": "DDIM (eta=0, timestep_spacing=trailing)", "gen_size": GEN_SIZE,
    "out_size": OUT_SIZE, "negative_prompt": NEGATIVE, "default_steps": DEFAULT_STEPS,
    "seeds": SEEDS, "guidance": GUIDANCE, "grid": GRID,
    "diffusers": __import__("diffusers").__version__, "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0), "generated": time.strftime("%Y-%m-%d"),
    "total_mb": round(mb, 2),
    "prompts": [dict(id=p[0], pl=p[1], en=p[2], prompt=p[3]) for p in PROMPTS],
    "cases": CASES,
}
with open(f"{OUT_DIR}/cases.js", "w") as f:
    f.write("// Generated by tools/generate_frames.ipynb - do not edit by hand\n")
    f.write("window.DIFFUSION_DATA = " + json.dumps(manifest, ensure_ascii=False, indent=1) + ";\n")
print("wrote cases.js")
""")

md("## Download")

code(r"""
import shutil
shutil.make_archive("/content/diffusion_frames", "zip", "/content/out")
from google.colab import files
files.download("/content/diffusion_frames.zip")
""")

nb = {
    "cells": [{**c, "source": [l + "\n" for l in c["source"].split("\n")[:-1]] + [c["source"].split("\n")[-1]]}
              for c in CELLS],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4, "nbformat_minor": 0,
}
out = Path(__file__).with_name("generate_frames.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("wrote", out)
