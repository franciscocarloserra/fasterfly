"""Build a signed sparse connectivity matrix from the MaleCNS v1.0 flat connectome.

Reads the feather tables, keeps the bodies selected in params.json, applies a sign
per presynaptic neuron from its consensus neurotransmitter, and caches the result
as a scipy CSR matrix plus the body index.

    python build_graph.py [--params params.json]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import scipy.sparse as sp

ROOT = Path(__file__).parent


def log(msg, t0=None):
    stamp = f"  [{time.time() - t0:6.1f}s]" if t0 else ""
    print(f"{msg}{stamp}", flush=True)


def build(params):
    t0 = time.time()
    d = params["data"]
    data_dir = ROOT / d["dir"]
    cache_dir = ROOT / d["cache"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    ann = feather.read_table(data_dir / d["annotations"]).to_pandas()
    log(f"annotations: {len(ann):,} bodies", t0)

    keep = params["graph"]["status_keep"]
    ann = ann[ann["status"].isin(keep)]
    log(f"kept status {keep}: {len(ann):,} bodies", t0)

    nt = feather.read_table(data_dir / d["neurotransmitters"]).to_pandas()
    nt = nt[["body", "consensus_nt", "predicted_nt_confidence"]].drop_duplicates("body")
    min_conf = params["neurotransmitter"]["min_nt_confidence"]
    if min_conf > 0:
        nt.loc[nt["predicted_nt_confidence"] < min_conf, "consensus_nt"] = "unknown"
    log(f"neurotransmitters: {len(nt):,} bodies", t0)

    w = feather.read_table(data_dir / d["weights"]).to_pandas()
    log(f"edges (raw): {len(w):,}", t0)

    min_w = params["graph"]["min_weight"]
    if min_w > 1:
        w = w[w["weight"] >= min_w]
        log(f"edges (weight >= {min_w}): {len(w):,}", t0)

    # Restrict to the kept bodies and map body IDs to dense indices.
    bodies = np.sort(ann["bodyId"].to_numpy().astype(np.int64))
    idx = pd.Series(np.arange(len(bodies), dtype=np.int64), index=bodies)

    pre = idx.reindex(w["body_pre"].to_numpy()).to_numpy()
    post = idx.reindex(w["body_post"].to_numpy()).to_numpy()
    valid = ~(np.isnan(pre) | np.isnan(post))
    pre, post = pre[valid].astype(np.int64), post[valid].astype(np.int64)
    weight = w["weight"].to_numpy()[valid].astype(np.float32)
    log(f"edges (both endpoints kept): {len(weight):,}", t0)

    # Sign from the presynaptic neuron's neurotransmitter.
    sign_map = params["neurotransmitter"]["sign"]
    default_sign = params["neurotransmitter"]["default_sign"]
    nt_by_body = nt.set_index("body")["consensus_nt"]
    nt_per_node = nt_by_body.reindex(bodies).fillna("unknown").str.lower()
    sign = nt_per_node.map(sign_map).fillna(default_sign).to_numpy().astype(np.float32)
    log(f"signed neurons: +{(sign > 0).sum():,} -{(sign < 0).sum():,} 0={(sign == 0).sum():,}", t0)

    signed_w = weight * sign[pre]

    # CSR laid out as [post, pre] so that a step is M @ spikes.
    M = sp.csr_matrix(
        (signed_w, (post, pre)), shape=(len(bodies), len(bodies)), dtype=np.float32
    )
    M.eliminate_zeros()
    log(f"matrix: {M.shape[0]:,} x {M.shape[1]:,}, nnz={M.nnz:,}", t0)

    sp.save_npz(cache_dir / "connectome_signed.npz", M)
    np.save(cache_dir / "bodies.npy", bodies)
    meta = ann.set_index("bodyId").reindex(bodies)[
        ["type", "class", "superclass", "somaSide"]
    ]
    meta["consensus_nt"] = nt_per_node.to_numpy()
    meta.to_parquet(cache_dir / "meta.parquet")
    log(f"wrote cache to {cache_dir}", t0)
    return M, bodies, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=Path, default=ROOT / "params.json")
    args = ap.parse_args()
    build(json.loads(args.params.read_text()))


if __name__ == "__main__":
    main()
