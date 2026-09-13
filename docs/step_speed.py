"""Render docs/step_speed.png from the measured numbers in docs/step_speed.json."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = json.loads((Path(__file__).parent / "step_speed.json").read_text())
S, T = D["single_fly"], D["batched"]
c = D["style"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=c["surface"], gridspec_kw={"width_ratios": [1.25, 1]})
for ax in (a, b):
    ax.set_facecolor(c["surface"])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(colors=c["text2"], length=0)
    ax.grid(axis="x", color=c["grid"], lw=0.8); ax.set_axisbelow(True)
names = [r["name"] for r in S][::-1]; vals = [r["steps_per_s"] for r in S][::-1]
cols = [c["accent"] if r["name"] == S[-1]["name"] else c["muted"] for r in S][::-1]
a.barh(names, vals, color=cols, height=0.62)
for i, v in enumerate(vals): a.text(v + 60, i, f"{v:,}", va="center", color=c["text"], fontsize=10.5)
a.set_xlim(0, max(vals) * 1.18); a.set_xlabel("steps / s  (one fly)", color=c["text2"])
a.set_title("Single-fly step, same stimulus", loc="left", color=c["text"], fontweight="bold")
a.tick_params(axis="y", labelcolor=c["text"])
n = [str(r["n_flies"]) for r in T]; fs = [r["fly_steps_per_s"] for r in T]
b.grid(axis="x", visible=False); b.grid(axis="y", color=c["grid"], lw=0.8)
b.bar(n, fs, color=c["accent"], width=0.62)
for i, (r, v) in enumerate(zip(T, fs)):
    b.text(i, v + 900, f"{v:,}\n{r['steps_per_s']:,} steps/s", ha="center", color=c["text"], fontsize=9.5)
b.set_ylim(0, max(fs) * 1.22); b.set_xlabel("flies per batch", color=c["text2"]); b.set_ylabel("fly-steps / s", color=c["text2"])
b.set_title("Batched (event_fused), throughput", loc="left", color=c["text"], fontweight="bold")
fig.suptitle(D["title"], x=0.01, ha="left", color=c["text"], fontsize=14, fontweight="bold", y=0.985)
fig.text(0.01, 0.905, D["subtitle"], color=c["text2"], fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.87))
fig.savefig(Path(__file__).parent / "step_speed.png", dpi=160, facecolor=c["surface"])
