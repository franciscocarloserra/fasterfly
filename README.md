# malecns-lif-kernels

**Simulate the whole MaleCNS fly connectome (165,122 neurons, 24.5M synapses) at 3,455 LIF steps/s on one RTX 3090. 4.9x faster than `torch.sparse` for a single fly, 72x more throughput with 64 flies in a batch. Same spikes, validated bit-for-bit where the arithmetic allows.**

![step speed](docs/step_speed.png)

## What you get

* **4.9x lower latency per step.** 708 → 3,455 steps/s on the full graph. With dt = 1 ms that is 3.5x faster than real time for one fly, where `torch.sparse` COO runs below real time.
* **72x more throughput.** 64 flies in one batch give 51,159 fly-steps/s. Parameter sweeps and readout training over many episodes run in an hour instead of days.
* **Drop-in.** One class, `EventFusedLIF`, takes a CSR matrix and LIF constants and returns the next state. No changes to the graph, the constants, or the stimulus.
* **Validated.** Every backend runs the same input and produces the same 59,338 spikes; rasters are bit-identical to CSR on the validation seed and flies in a batch are bit-identical to the same fly alone.
* **Generic.** Nothing in the kernel is fly-specific: any large sparse signed graph with ~0.1-1% activity per step gets the same gains.

## How it works

`torch.sparse` multiplies the whole matrix by the spike vector every step, touching all 24.5M synapses when fewer than 200 neurons fired. The event-driven kernel launches one Triton program per presynaptic neuron; silent ones return immediately and fired ones `atomic_add` their outgoing weights into the input current. A second elementwise kernel applies the LIF update. Batching adds a fly dimension to the state, so one launch serves N flies.

## Numbers (RTX 3090, dt = 1 ms, fp32)

Same stimulus for every backend: 1,024 visual-projection neurons driven at 100 Hz
Poisson, 500 steps, 59,338 spikes total (~0.07% of neurons per step).

| Backend, single fly | steps/s | vs COO |
|---|---|---|
| torch.sparse COO (`torch.sparse.mm`) | 708 | 1x |
| torch.sparse CSR int64 | ~2,000 | 2.8x |
| torch.sparse CSR int32 | 2,522 | 3.6x |
| Triton fused, one program per row | 2,419 | 3.4x |
| Triton event_fused (scatter + LIF) | 3,455 | 4.9x |
| Triton event_fused_batched, N=1 | 3,485 | 4.9x |

fp16 values and CUDA graph capture add nothing on top of CSR int32. At 10x stimulus
(0.98% neurons per step) event_fused still gives 3,391 steps/s: the cost is launching
165k programs plus the elementwise LIF kernel, not the synapses.

### Batched over flies

State is `[n_neurons, n_flies]`, fly index contiguous. One scatter program per
(presynaptic neuron, tile of `fly_block` flies); a 2D `atomic_add` on `cur[col, fly]`
coalesces over flies and is masked by which flies fired, so flies never mix. Programs
whose tile is silent return immediately.

| n_flies | steps/s | fly-steps/s | vs COO single fly |
|---|---|---|---|
| 1 | 3,485 | 3,485 | 4.9x |
| 4 | 2,237 | 8,946 | 12.6x |
| 16 | 1,866 | 29,856 | 42x |
| 64 | 799 | 51,159 | 72x |

Batching raises throughput (parameter sweeps, readout training on many episodes),
not the latency of one fly.

## Validation

* `A·s` against `torch.mv` on adversarial spike vectors (all ones, top-1000 out-degree,
  1%, 50%): all allclose.
* 500-step raster bit-identical to CSR on the validation seed; N=1 batched bit-identical
  to `event_fused` and to CSR.
* Batch isolation: fly k in a batch of N is bit-identical to fly k alone, for N in
  {4, 16, 64}, with identical or per-fly stimuli.
* Not achievable: bit-identity with cuSPARSE for every seed. Atomic summation order
  differs, and a 1-ulp threshold tie (seed 13: neuron 60339, step 242, v = 1.000000119
  against v_th = 1.0) flips one spike and the chaotic network diverges afterwards. Each
  backend is deterministic run to run. Cross-backend bit-identity would need a
  deterministic gather kernel.

Full logs and per-run JSON: `runs/event_kernel/`, `runs/batched/`.

## Files

| File | What |
|---|---|
| `event_lif.py` | `EventFusedLIF`: Triton scatter kernel + LIF kernel, batched, reusable |
| `fast_step.py` | Benchmark and validation of every backend on one stimulus |
| `fast_params.json` | Benchmark knobs: variants, steps, stimulus, `n_flies` list, kernel blocks |
| `build_graph.py` | MaleCNS release tables → signed CSR (`cache/connectome_signed.npz`) + `cache/meta.parquet` |
| `params.json` | Data paths, graph filters, neurotransmitter signs, LIF constants |

## Run

```
python -m venv venv && ./venv/bin/pip install -r requirements.txt
# MaleCNS v1.0 feather tables in data/v1.0/ (see params.json "data")
./venv/bin/python build_graph.py
./venv/bin/python fast_step.py
```

Use `EventFusedLIF` directly:

```python
from event_lif import EventFusedLIF
lif = EventFusedLIF(M_csr, lif_params, device, block=128, fly_block=16)
v, s = lif.step(v, s, ext)   # all [n_neurons, n_flies] float32 contiguous
```

## Scope and limits

The kernel is generic: any sparse signed graph with low per-step activity. The regime
where it wins is what matters (large n, ~0.1-1% firing); dense or highly active graphs
are better served by cuSPARSE, and much smaller graphs by CUDA graph capture. LIF
constants are dimensionless (not Shiu et al. 2024). Ceiling for single-fly latency with
further work (fired-neuron compaction, CUDA graph over the launches) is launch latency,
~50-100k steps/s theoretical.

Versions measured: torch 2.6.0+cu124, triton 3.2.0.
