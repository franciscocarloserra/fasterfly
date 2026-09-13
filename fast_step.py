"""Native LIF step-speed probe on the signed MaleCNS connectome (no browser).

Runs the same stimulus through several step implementations and reports
steps/s plus total spikes, so a faster variant can be checked against the
baseline for equivalence.

    ./venv/bin/python fast_step.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parent
P = json.loads((ROOT / "fast_params.json").read_text())
LIF = json.loads((ROOT / "params.json").read_text())["lif"]
dev = torch.device(P["device"] if torch.cuda.is_available() else "cpu")

# -- load -------------------------------------------------------------------
M = sp.load_npz(ROOT / "cache" / "connectome_signed.npz").tocsr().astype(np.float32)
n = M.shape[0]
meta = pd.read_parquet(ROOT / "cache" / "meta.parquet")
st = P["stimulus"]
rng = np.random.default_rng(st["seed"])
cand = np.flatnonzero(meta["superclass"].isin(st["superclass"]).values)
targets = torch.tensor(rng.choice(cand, st["n_targets"], replace=False), device=dev)

decay = float(np.exp(-LIF["dt_ms"] / LIF["tau_m_ms"]))
gain, v_th, v_reset = LIF["synaptic_gain"], LIF["v_threshold"], LIF["v_reset"]
p_spike = st["rate_hz"] * LIF["dt_ms"] / 1000.0

# Pre-generated stimulus so every variant sees identical input.
g = torch.Generator(device=dev).manual_seed(st["seed"])
total = P["warmup_steps"] + P["steps"]
stim = (torch.rand(total, st["n_targets"], generator=g, device=dev) < p_spike).float() * st["amplitude"]


def make_ext(t):
    ext = torch.zeros(n, device=dev)
    ext[targets] = stim[t]
    return ext


# -- variants: each returns a step(v, s, ext) -> (v, s) -----------------------
def lif_update(v, cur):
    v = v * decay + cur
    fired = v >= v_th
    v = torch.where(fired, torch.full_like(v, v_reset), v)
    return v, fired.float()


def variant_coo():
    A = torch.sparse_coo_tensor(
        torch.tensor(np.vstack(M.tocoo().nonzero()), dtype=torch.long),
        torch.tensor(M.data), M.shape).coalesce().to(dev)

    def step(v, s, ext):
        return lif_update(v, torch.sparse.mm(A, s.unsqueeze(1)).squeeze(1) * gain + ext)
    return step


def variant_csr(idx_dtype=torch.int64, val_dtype=torch.float32):
    A = torch.sparse_csr_tensor(
        torch.tensor(M.indptr, dtype=idx_dtype), torch.tensor(M.indices, dtype=idx_dtype),
        torch.tensor(M.data, dtype=val_dtype), M.shape).to(dev)

    def step(v, s, ext):
        return lif_update(v, torch.mv(A, s.to(val_dtype)).float() * gain + ext)
    return step


def variant_csr_graph():
    """Same as csr but the whole step captured once in a CUDA graph."""
    step = variant_csr()
    v0, s0, e0 = (torch.zeros(n, device=dev) for _ in range(3))
    sv = torch.cuda.Stream()
    sv.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(sv):
        for _ in range(3):
            step(v0, s0, e0)
    torch.cuda.current_stream().wait_stream(sv)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        v1, s1 = step(v0, s0, e0)

    def gstep(v, s, ext):
        v0.copy_(v); s0.copy_(s); e0.copy_(ext)
        graph.replay()
        return v1.clone(), s1.clone()
    return gstep


def variant_event():
    """Event-driven: only gather the outgoing synapses of neurons that fired."""
    T = M.T.tocsr()  # row = presynaptic neuron, cols = its targets
    indptr = torch.tensor(T.indptr, dtype=torch.int64, device=dev)
    cols = torch.tensor(T.indices, dtype=torch.int64, device=dev)
    vals = torch.tensor(T.data, device=dev)

    def step(v, s, ext):
        fired = torch.nonzero(s, as_tuple=True)[0]
        cur = ext.clone()
        if fired.numel():
            start, end = indptr[fired], indptr[fired + 1]
            cnt = end - start
            offs = torch.repeat_interleave(start - torch.cumsum(cnt, 0) + cnt, cnt)
            idx = torch.arange(int(cnt.sum()), device=dev) + offs
            cur.index_add_(0, cols[idx], vals[idx] * gain)
        return lif_update(v, cur)
    return step


try:  # Triton's JIT resolves `tl` from module globals, so the import must live here.
    import triton
    import triton.language as tl
except ImportError:
    triton = None


def variant_fused():
    """Single Triton kernel: CSR gather + LIF update, one program per block of rows."""

    @triton.jit
    def lif_kernel(indptr_ptr, indices_ptr, values_ptr, s_ptr, ext_ptr, v_ptr, v_out_ptr, s_out_ptr,
                   n_rows, decay, gain, v_th, v_reset, ROWS: tl.constexpr, BLOCK: tl.constexpr):
        row0 = tl.program_id(0) * ROWS
        for r in range(ROWS):
            row = row0 + r
            if row < n_rows:
                start = tl.load(indptr_ptr + row)
                end = tl.load(indptr_ptr + row + 1)
                acc = tl.zeros([BLOCK], dtype=tl.float32)
                for k0 in range(start, end, BLOCK):
                    ks = k0 + tl.arange(0, BLOCK)
                    m = ks < end
                    idx = tl.load(indices_ptr + ks, mask=m, other=0)
                    val = tl.load(values_ptr + ks, mask=m, other=0.0)
                    sv = tl.load(s_ptr + idx, mask=m, other=0.0)
                    acc += val * sv
                cur = gain * tl.sum(acc, axis=0) + tl.load(ext_ptr + row)
                v = tl.load(v_ptr + row) * decay + cur
                fired = v >= v_th
                tl.store(v_out_ptr + row, tl.where(fired, v_reset, v))
                tl.store(s_out_ptr + row, tl.where(fired, 1.0, 0.0))

    indptr = torch.tensor(M.indptr, dtype=torch.int32, device=dev)
    indices = torch.tensor(M.indices, dtype=torch.int32, device=dev)
    values = torch.tensor(M.data, dtype=torch.float32, device=dev)
    ROWS, BLOCK = P.get("fused_rows", 4), P.get("fused_block", 128)
    grid = (triton.cdiv(n, ROWS),)

    def step(v, s, ext):
        v_out, s_out = torch.empty_like(v), torch.empty_like(s)
        lif_kernel[grid](indptr, indices, values, s, ext, v, v_out, s_out,
                         n, decay, gain, v_th, v_reset, ROWS=ROWS, BLOCK=BLOCK)
        return v_out, s_out
    return step


def variant_event_fused():
    """Event-driven Triton: one program per presynaptic neuron scatters its
    out-synapses with atomics (early return if silent), then an elementwise LIF kernel."""

    @triton.jit
    def scatter_kernel(indptr_ptr, cols_ptr, vals_ptr, s_ptr, cur_ptr, gain, BLOCK: tl.constexpr):
        pre = tl.program_id(0)
        if tl.load(s_ptr + pre) != 0.0:
            start = tl.load(indptr_ptr + pre)
            end = tl.load(indptr_ptr + pre + 1)
            for k0 in range(start, end, BLOCK):
                ks = k0 + tl.arange(0, BLOCK)
                m = ks < end
                col = tl.load(cols_ptr + ks, mask=m, other=0)
                val = tl.load(vals_ptr + ks, mask=m, other=0.0)
                tl.atomic_add(cur_ptr + col, gain * val, mask=m)

    @triton.jit
    def lif_kernel(v_ptr, cur_ptr, v_out_ptr, s_out_ptr, n_rows, decay, v_th, v_reset, BLOCK: tl.constexpr):
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = rows < n_rows
        v = tl.load(v_ptr + rows, mask=m, other=0.0) * decay + tl.load(cur_ptr + rows, mask=m, other=0.0)
        fired = v >= v_th
        tl.store(v_out_ptr + rows, tl.where(fired, v_reset, v), mask=m)
        tl.store(s_out_ptr + rows, tl.where(fired, 1.0, 0.0), mask=m)

    T = M.T.tocsr()
    indptr = torch.tensor(T.indptr, dtype=torch.int32, device=dev)
    cols = torch.tensor(T.indices, dtype=torch.int32, device=dev)
    vals = torch.tensor(T.data, dtype=torch.float32, device=dev)
    BLOCK = P.get("event_block", 128)
    grid_lif = (triton.cdiv(n, BLOCK),)

    def scatter(s, ext):
        cur = ext.clone()
        scatter_kernel[(n,)](indptr, cols, vals, s, cur, gain, BLOCK=BLOCK)
        return cur

    def step(v, s, ext):
        cur = scatter(s, ext)
        v_out, s_out = torch.empty_like(v), torch.empty_like(s)
        lif_kernel[grid_lif](v, cur, v_out, s_out, n, decay, v_th, v_reset, BLOCK=BLOCK)
        return v_out, s_out
    step.scatter, step.outdeg = scatter, np.diff(T.indptr)
    return step


def variant_event_fused_batched():
    """Same kernels as event_fused but state is [n, n_flies]; here wrapped for a single fly."""
    from event_lif import EventFusedLIF
    B = P.get("batched", {})
    lif = EventFusedLIF(M, LIF, dev, block=P.get("event_block", 128), fly_block=B.get("fly_block", 16))

    def step(v, s, ext):
        v_out, s_out = lif.step(v.unsqueeze(1), s.unsqueeze(1), ext.unsqueeze(1))
        return v_out.squeeze(1), s_out.squeeze(1)
    step.lif = lif
    return step


def validate_scatter(step):
    """Compare the scatter kernel against torch.mv(A_csr, s) * gain + ext on 4 spike patterns."""
    A = torch.sparse_csr_tensor(torch.tensor(M.indptr), torch.tensor(M.indices), torch.tensor(M.data), M.shape).to(dev)
    r = torch.Generator(device=dev).manual_seed(1)
    top = torch.tensor(np.argsort(-step.outdeg)[:1000], device=dev)
    cases = {"all_ones": torch.ones(n, device=dev),
             "top1000_outdeg": torch.zeros(n, device=dev).index_fill_(0, top, 1.0),
             "rand_1pct": (torch.rand(n, generator=r, device=dev) < 0.01).float(),
             "rand_50pct": (torch.rand(n, generator=r, device=dev) < 0.50).float()}
    ext = make_ext(0)
    for name, s in cases.items():
        ref, got = torch.mv(A, s) * gain + ext, step.scatter(s, ext)
        res = {"max_abs_diff": float((ref - got).abs().max()), "allclose": bool(torch.allclose(ref, got, atol=1e-4, rtol=1e-4))}
        print(f"validate {name:15s} max|diff|={res['max_abs_diff']:.3e} allclose={res['allclose']}")
        RUN["scatter"][name] = res


def pack(raster):
    """[steps, n] bool tensor -> packed uint8 numpy (exact-equality comparisons on CPU)."""
    return np.packbits(raster.cpu().numpy(), axis=1)


def stim_for(seed):
    g = torch.Generator(device=dev).manual_seed(int(seed))
    return (torch.rand(total, st["n_targets"], generator=g, device=dev) < p_spike).float() * st["amplitude"]


def run_batched(lif, seeds, capture):
    """Run the batched kernel with one stimulus seed per fly. Returns (steps/s, [packed raster per fly] or None)."""
    N = len(seeds)
    S = torch.stack([stim_for(k) for k in seeds], dim=2)  # [total, n_targets, N]
    v, s = torch.zeros(n, N, device=dev), torch.zeros(n, N, device=dev)
    ext = torch.zeros(n, N, device=dev)
    rows = []  # per step: packed [N, n/8] on CPU (a full [steps, n, N] bool raster would not fit the GPU at N=64)
    for t in range(P["warmup_steps"]):
        ext[targets] = S[t]
        v, s = lif.step(v, s, ext)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for t in range(P["warmup_steps"], total):
        ext[targets] = S[t]
        v, s = lif.step(v, s, ext)
        if capture:
            rows.append(pack(s.bool().T))
    torch.cuda.synchronize()
    sps = P["steps"] / (time.perf_counter() - t0)
    return sps, ([np.stack([r[k] for r in rows]) for k in range(N)] if capture else None)


def run_single_ref(seed):
    """Reference single-fly raster (csr variant) for stimulus `seed`, packed."""
    step, S = variant_csr(), stim_for(seed)
    v, s = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
    rast = torch.zeros(P["steps"], n, dtype=torch.bool, device=dev)
    for t in range(total):
        ext = torch.zeros(n, device=dev)
        ext[targets] = S[t]
        v, s = step(v, s, ext)
        if t >= P["warmup_steps"]:
            rast[t - P["warmup_steps"]] = s.bool()
    return pack(rast)


VARIANTS = {
    "coo": variant_coo, "csr": variant_csr, "csr_graph": variant_csr_graph, "event": variant_event,
    "fused": variant_fused, "event_fused": variant_event_fused, "event_fused_batched": variant_event_fused_batched,
    "csr_i32": lambda: variant_csr(torch.int32),
    "csr_i32_f16": lambda: variant_csr(torch.int32, torch.float16),
}

# -- run --------------------------------------------------------------------
print(f"neurons={n} synapses={M.nnz} device={dev} steps={P['steps']}")
validate, rasters = P.get("validate", False), {}
RUN = {"scatter": {}, "variants": {}}  # this run's results, keyed into validation.json by rate_hz
for name in P["variants"]:
    if name == "csr_graph" and dev.type != "cuda":
        continue
    step = VARIANTS[name]()
    if validate and name == "event_fused":
        validate_scatter(step)
    v, s = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
    spikes = 0.0
    for t in range(P["warmup_steps"]):
        v, s = step(v, s, make_ext(t))
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for t in range(P["warmup_steps"], total):
        v, s = step(v, s, make_ext(t))
        spikes += s.sum()
        if validate:
            rasters.setdefault(name, torch.zeros(P["steps"], n, dtype=torch.bool, device=dev))[t - P["warmup_steps"]] = s.bool()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    sps = P["steps"] / dt
    print(f"{name:10s} {sps:8.0f} steps/s  {sps * LIF['dt_ms'] / 1000:5.2f}x realtime  "
          f"spikes={float(spikes):.0f}  ({float(spikes) / P['steps'] / n * 100:.2f}%/step)")
    RUN["variants"][name] = {"steps_per_s": round(sps), "spikes": float(spikes), "pct_per_step": float(spikes) / P["steps"] / n * 100}

if validate and "csr" in rasters and "event_fused" in rasters:
    diff = rasters["csr"] ^ rasters["event_fused"]
    dn = torch.tensor(meta["type"].isin(["DNp01", "DNp02", "DNp04", "DNp11"]).values, device=dev)
    RUN["raster_diff"] = {"events": int(diff.sum()), "dn_events": int(diff[:, dn].sum())}
    print(f"raster diff csr vs event_fused: {RUN['raster_diff']['events']} events, on DN rows: {RUN['raster_diff']['dn_events']}")

B = P.get("batched")
if B and B.get("enabled", True):
    from event_lif import EventFusedLIF
    bout = ROOT / B["out_dir"]
    bout.mkdir(parents=True, exist_ok=True)
    plog = open(bout / "progress.log", "a")

    def log(msg):
        print(msg)
        plog.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        plog.flush()

    lif = EventFusedLIF(M, LIF, dev, block=P.get("event_block", 128), fly_block=B.get("fly_block", 16))
    seed0, sizes = st["seed"], P["n_flies"]
    # Reference for (c): the batched kernel itself at N=1 with seed k (proves flies do not cross).
    # csr is reported separately: atomic summation order can flip a 1-ulp threshold tie (seen on seed 13,
    # neuron 60339, step 242) after which the network diverges, so csr bit-identity is not guaranteed per seed.
    ref = {k: run_batched(lif, [seed0 + k], capture=True)[1][0] for k in range(max(sizes))}
    csr_ref = {k: run_single_ref(seed0 + k) for k in range(max(sizes))}
    csr_diff = {k: int(np.unpackbits(ref[k] ^ csr_ref[k], axis=1).sum()) for k in ref}
    log(f"batched: {len(ref)} single-fly references computed; seeds whose N=1 kernel raster differs from csr: "
        f"{ {k: d for k, d in csr_diff.items() if d} }")
    RUN["csr_diff_events_per_seed"] = csr_diff
    if validate:  # (a) N=1 vs csr and event_fused rasters from the single-fly loop
        _, r1 = run_batched(lif, [seed0], capture=True)
        RUN["batched_a"] = {name: bool(np.array_equal(pack(rasters[name]), r1[0])) for name in ("csr", "event_fused") if name in rasters}
        RUN["batched_a"]["csr_ref"] = bool(np.array_equal(csr_ref[0], r1[0]))
        log(f"(a) N=1 identical to: {RUN['batched_a']}")
    RUN["batched"] = {}
    for N in sizes:
        same = np.array_equal
        _, same_r = run_batched(lif, [seed0] * N, capture=True)    # (b) identical ext per fly
        _, diff_r = run_batched(lif, [seed0 + k for k in range(N)], capture=True)  # (c) seed k per fly
        b_ok = all(same(r, csr_ref[0]) for r in same_r)
        c_ok = all(same(diff_r[k], ref[k]) for k in range(N))
        c_csr = sum(1 for k in range(N) if same(diff_r[k], csr_ref[k]))
        sps, _ = run_batched(lif, [seed0 + k for k in range(N)], capture=False)
        RUN["batched"][N] = {"steps_per_s": round(sps), "fly_steps_per_s": round(sps * N), "b_identical_ext": b_ok, "c_seed_per_fly": c_ok,
                             "c_flies_identical_to_csr": c_csr, "spikes_per_fly": [int(np.unpackbits(r, axis=1).sum()) for r in diff_r]}
        log(f"N={N:3d}  {sps:8.0f} steps/s  {sps * N:9.0f} fly-steps/s  (b) same-ext ok={b_ok}  (c) seed-per-fly ok={c_ok}  csr-identical flies {c_csr}/{N}")
    lines = ["# event_fused_batched", "", f"stimulus: rate_hz={st['rate_hz']} steps={P['steps']} seed0={seed0}, fly k uses seed0+k.",
             "(b): every fly column == csr single-fly raster (seed0). (c): fly k == batched kernel at N=1 with seed k; csr column counts flies also bit-identical to csr.",
             "", "| n_flies | steps/s | fly-steps/s | (b) identical ext | (c) seed per fly | flies == csr |", "|---|---|---|---|---|---|"]
    lines += [f"| {N} | {r['steps_per_s']} | {r['fly_steps_per_s']} | {r['b_identical_ext']} | {r['c_seed_per_fly']} | {r['c_flies_identical_to_csr']}/{N} |" for N, r in RUN["batched"].items()]
    if "batched_a" in RUN:
        lines += ["", f"(a) N=1 raster bit-identical to: {RUN['batched_a']}"]
    lines += ["", f"seeds whose N=1 kernel raster differs from csr (events): { {k: d for k, d in csr_diff.items() if d} }",
              "cause: atomic summation order flips a 1-ulp threshold tie (seed 13: neuron 60339, step 242, v=1.000000119 vs v_th=1.0), then the network diverges; kernel is deterministic run-to-run."]
    lines += ["", "single-fly variants this run: " + ", ".join(f"{k}={v['steps_per_s']} steps/s" for k, v in RUN["variants"].items())]
    (bout / "summary.md").write_text("\n".join(lines) + "\n")
    (bout / "validation.json").write_text(json.dumps({"params": P, "run": RUN}, indent=2, default=str))
    log(f"wrote {bout / 'summary.md'}")

if validate and P.get("out_dir"):  # persist: params copy, validation.json (merged per rate_hz), summary.md
    out = ROOT / P["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "params.json").write_text(json.dumps(P, indent=2))
    vf = out / "validation.json"
    V = json.loads(vf.read_text()) if vf.exists() else {}
    V[str(st["rate_hz"])] = RUN
    vf.write_text(json.dumps(V, indent=2))
    lines = ["# event_fused vs csr", "", "| rate_hz | variant | steps/s | spikes | %/step |", "|---|---|---|---|---|"]
    for rate, R in V.items():
        for name, r in R["variants"].items():
            lines.append(f"| {rate} | {name} | {r['steps_per_s']} | {r['spikes']:.0f} | {r['pct_per_step']:.3f} |")
    for rate, R in V.items():
        lines += ["", f"## validation @ rate_hz={rate}"]
        lines += [f"- scatter {k}: max|diff|={v['max_abs_diff']:.3e} allclose={v['allclose']}" for k, v in R["scatter"].items()]
        if "raster_diff" in R:
            lines.append(f"- raster diff: {R['raster_diff']['events']} events, {R['raster_diff']['dn_events']} on DNp01/02/04/11")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
