# event_fused_batched

stimulus: rate_hz=100.0 steps=500 seed0=0, fly k uses seed0+k.
(b): every fly column == csr single-fly raster (seed0). (c): fly k == batched kernel at N=1 with seed k; csr column counts flies also bit-identical to csr.

| n_flies | steps/s | fly-steps/s | (b) identical ext | (c) seed per fly | flies == csr |
|---|---|---|---|---|---|
| 1 | 3485 | 3485 | True | True | 1/1 |
| 4 | 2237 | 8946 | True | True | 4/4 |
| 16 | 1866 | 29856 | True | True | 15/16 |
| 64 | 799 | 51159 | True | True | 63/64 |

(a) N=1 raster bit-identical to: {'csr': True, 'event_fused': True, 'csr_ref': True}

seeds whose N=1 kernel raster differs from csr (events): {13: 6942}
cause: atomic summation order flips a 1-ulp threshold tie (seed 13: neuron 60339, step 242, v=1.000000119 vs v_th=1.0), then the network diverges; kernel is deterministic run-to-run.

single-fly variants this run: csr=1608 steps/s, event_fused=3380 steps/s, event_fused_batched=2341 steps/s

# server backend validation (Brain.advance)

ticks=20 steps_per_tick=11 seed=0

| n_flies | backend | steps/s (server counter, median) | cl total | cr total | identical to coo |
|---|---|---|---|---|---|
| 1 | coo | 777 | 754 | 653 | True |
| 1 | event_fused | 2967 | 754 | 653 | True |
| 16 | coo | 261 | 11745 | 11400 | True |
| 16 | event_fused | 1915 | 11747 | 11402 | False |
  N=16: |cl|+|cr| count difference summed over ticks and flies = 4

Stage 2 notes: table measured after aligning the COO LIF update to `v*decay + cur` (it used `v += -v*(1-decay) + cur`, different rounding; before that the N=16 difference was 46 counts, see todelete/summary.md). Residual 4-count difference at N=16 (3 of 16 flies, first divergence at tick >= 7) is the same class as stage 1: atomic vs cuSPARSE summation order flipping a threshold tie, then divergence. Each backend is self-consistent (fly k at N=16 == same fly alone at N=1, both backends). N=1 identical over 20 ticks. Default set to parallel.step_backend = "event_fused".
