"""Batched event-driven LIF step on the signed connectome (Triton).

State layout is [n_neurons, n_flies], fly index contiguous: for a fired
presynaptic neuron the scatter writes cur[col, fly] for every fired fly, so a
2D block (synapse x fly) hits F contiguous floats per postsynaptic column and
the fly dimension coalesces. Atomics are indexed by (col, fly), so flies never
mix. Silent (pre, fly) pairs are masked out of the atomic, and a presynaptic
neuron silent in every fly returns early.

    lif = EventFusedLIF(M_csr, lif_params, device, block=128, fly_block=16)
    v, s = lif.step(v, s, ext)      # all [n, n_flies] float32 contiguous
"""
import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def scatter_kernel(indptr_ptr, cols_ptr, vals_ptr, s_ptr, cur_ptr, gain, n_flies,
                   BLOCK: tl.constexpr, FB: tl.constexpr):
    pre = tl.program_id(0)
    f0 = tl.program_id(1) * FB
    flies = f0 + tl.arange(0, FB)
    fm = flies < n_flies
    s = tl.load(s_ptr + pre * n_flies + flies, mask=fm, other=0.0)
    fired = (s != 0.0) & fm
    if tl.sum(fired.to(tl.int32), axis=0) != 0:
        start = tl.load(indptr_ptr + pre)
        end = tl.load(indptr_ptr + pre + 1)
        for k0 in range(start, end, BLOCK):
            ks = k0 + tl.arange(0, BLOCK)
            m = ks < end
            col = tl.load(cols_ptr + ks, mask=m, other=0)
            val = tl.load(vals_ptr + ks, mask=m, other=0.0)
            ptrs = cur_ptr + col[:, None] * n_flies + flies[None, :]
            w = (gain * val)[:, None] * tl.where(fired, 1.0, 0.0)[None, :]
            tl.atomic_add(ptrs, w, mask=m[:, None] & fired[None, :])


@triton.jit
def lif_kernel(v_ptr, cur_ptr, v_out_ptr, s_out_ptr, numel, decay, v_th, v_reset, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < numel
    v = tl.load(v_ptr + i, mask=m, other=0.0) * decay + tl.load(cur_ptr + i, mask=m, other=0.0)
    fired = v >= v_th
    tl.store(v_out_ptr + i, tl.where(fired, v_reset, v), mask=m)
    tl.store(s_out_ptr + i, tl.where(fired, 1.0, 0.0), mask=m)


class EventFusedLIF:
    def __init__(self, M_csr, lif, dev, block=128, fly_block=16):
        T = M_csr.T.tocsr()  # row = presynaptic neuron, cols = its targets
        self.n = M_csr.shape[0]
        self.indptr = torch.tensor(T.indptr, dtype=torch.int32, device=dev)
        self.cols = torch.tensor(T.indices, dtype=torch.int32, device=dev)
        self.vals = torch.tensor(T.data, dtype=torch.float32, device=dev)
        self.outdeg = np.diff(T.indptr)
        self.decay = float(np.exp(-lif["dt_ms"] / lif["tau_m_ms"]))
        self.gain, self.v_th, self.v_reset = float(lif["synaptic_gain"]), float(lif["v_threshold"]), float(lif["v_reset"])
        self.block, self.fly_block = int(block), int(fly_block)

    def scatter(self, s, ext):
        """cur = A @ s * gain + ext, s/ext [n, F] contiguous."""
        n, F = s.shape
        cur = ext.contiguous().clone()
        grid = (n, triton.cdiv(F, self.fly_block))
        scatter_kernel[grid](self.indptr, self.cols, self.vals, s.contiguous(), cur, self.gain, F,
                             BLOCK=self.block, FB=self.fly_block)
        return cur

    def step(self, v, s, ext):
        cur = self.scatter(s, ext)
        v = v.contiguous()
        v_out, s_out = torch.empty_like(v), torch.empty_like(v)
        numel = v.numel()
        lif_kernel[(triton.cdiv(numel, self.block),)](v, cur, v_out, s_out, numel, self.decay, self.v_th, self.v_reset,
                                                      BLOCK=self.block)
        return v_out, s_out
