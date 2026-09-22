"""Assemble a Hugging Face release folder: weights + tokenizer + selectia_config.json + the inference subset of the
package + the model licence + the card.

    python scripts/stage_release.py runs/selectia_full/model runs/release/selectia-2.6b [eval_results.json] [MODEL_CARD.md]

The weights and `selectia_config.json` are covered by the LFM Open License v1.0 (MODEL-LICENSE), not the Apache-2.0
code licence, so the licence text ships next to them.
"""
import os, shutil, sys

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(f"{dst}/selectia", exist_ok=True)
for f in os.listdir(src):
    if os.path.isfile(f"{src}/{f}"):
        shutil.copy2(f"{src}/{f}", f"{dst}/{f}")
for m in ["__init__", "prompt", "model", "systemone", "infer", "metrics"]:       # everything needed to run; no training code
    shutil.copy2(f"selectia/{m}.py", f"{dst}/selectia/{m}.py")
shutil.copy2(sys.argv[4] if len(sys.argv) > 4 else "MODEL_CARD.md", f"{dst}/README.md")
for lic in ("MODEL-LICENSE", "CODE-LICENSE"):
    if os.path.exists(lic):
        shutil.copy2(lic, f"{dst}/{lic}")
if len(sys.argv) > 3:
    shutil.copy2(sys.argv[3], f"{dst}/eval_results.json")
assert os.path.exists(f"{dst}/selectia_config.json"), "write selectia_config.json (temperature, version, schema_first, isolated_levels, yesmom) into the model folder first"
print("staged", dst, sorted(os.listdir(dst)))
