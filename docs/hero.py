"""Render docs/hero.png: soma cloud of the whole CNS coloured by spikes from a short
EventFusedLIF run. Knobs in docs/hero.json."""
import json
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, torch
import pyarrow.feather as feather
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from event_lif import EventFusedLIF

ROOT = Path(__file__).resolve().parent.parent
H = json.loads((ROOT / "docs" / "hero.json").read_text())
P = json.loads((ROOT / "params.json").read_text())
dev = torch.device("cuda")

M = sp.load_npz(ROOT / "cache" / "connectome_signed.npz").tocsr().astype(np.float32)
n = M.shape[0]
meta = pd.read_parquet(ROOT / "cache" / "meta.parquet")
bodies = np.load(ROOT / "cache" / "bodies.npy")
ann = feather.read_table(ROOT / P["data"]["dir"] / P["data"]["annotations"], columns=["bodyId", "somaLocation"]).to_pandas()
ann = ann.set_index("bodyId").reindex(bodies)
xyz = np.array([list(v) if v is not None and len(v) == 3 else [np.nan] * 3 for v in ann["somaLocation"]], dtype=float)

st = H["stimulus"]; lif = P["lif"]
rng = np.random.default_rng(st["seed"])
cand = np.flatnonzero(meta["superclass"].isin(st["superclass"]).values)
targets = torch.tensor(rng.choice(cand, st["n_targets"], replace=False), device=dev)
p_spike = st["rate_hz"] * lif["dt_ms"] / 1000.0
g = torch.Generator(device=dev).manual_seed(st["seed"])
total = H["warmup_steps"] + H["steps"]
stim = (torch.rand(total, st["n_targets"], generator=g, device=dev) < p_spike).float() * st["amplitude"]

N = H["n_flies"]
net = EventFusedLIF(M, lif, dev)
v = torch.zeros(n, N, device=dev); s = torch.zeros_like(v); ext = torch.zeros_like(v)
stims = [stim] + [(torch.rand(total, st["n_targets"], generator=torch.Generator(device=dev).manual_seed(st["seed"] + k), device=dev) < p_spike).float() * st["amplitude"] for k in range(1, N)]
stim = torch.stack(stims, 2)  # [total, n_targets, N]
counts = torch.zeros(n, N, device=dev)
for t in range(total):
    ext.zero_(); ext[targets] = stim[t]
    v, s = net.step(v, s, ext)
    if t >= H["warmup_steps"]: counts += s
counts = counts.cpu().numpy()
print(f"spikes/fly={counts.sum(0).astype(int).tolist()}")

V, C = H["view"], H["style"]
ok = ~np.isnan(xyz[:, 0])
x, y = xyz[ok, V["x_axis"]], xyz[ok, V["y_axis"]] * (-1 if V["flip_y"] else 1)
vmax = counts.max()

def cloud(ax, c, ps, acs):
    ax.set_facecolor(C["surface"]); ax.set_axis_off()
    ax.scatter(x[c == 0], y[c == 0], s=ps, c=C["silent"], lw=0, rasterized=True)
    o = np.argsort(c[c > 0])
    ax.scatter(x[c > 0][o], y[c > 0][o], s=acs, c=c[c > 0][o], cmap=C["cmap"], vmin=-vmax * 0.4, vmax=vmax, lw=0)
    ax.set_aspect("equal")

fig = plt.figure(figsize=(13, 7.4), facecolor=C["surface"])
ax = fig.add_axes([0.03, 0.15, 0.42, 0.70]); cloud(ax, counts[ok, 0], V["point_size"], V["active_size"])
gr, gc = H["grid"]
for k in range(N):
    r, cc = divmod(k, gc)
    a = fig.add_axes([0.52 + cc * 0.115, 0.60 - r * 0.145, 0.11, 0.14]); cloud(a, counts[ok, k], 0.08, 0.5)
fig.text(0.24, 0.035, H["single_label"], color=C["text"], fontsize=12, ha="center", va="bottom", linespacing=1.4)
fig.text(0.75, 0.035, H["batch_label"], color=C["text"], fontsize=12, ha="center", va="bottom", linespacing=1.4)
fig.text(0.02, 0.93, H["title"], color=C["text"], fontsize=34, fontweight="bold")
fig.text(0.02, 0.885, H["subtitle"], color=C["text2"], fontsize=12.5)
fig.text(0.02, 0.005, H["caption"].format(steps=H["steps"]), color=C["text2"], fontsize=9.5)
fig.savefig(ROOT / "docs" / "hero.png", dpi=150, facecolor=C["surface"])
