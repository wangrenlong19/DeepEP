#!/usr/bin/env python3
"""Compare two per-tile checksum tables from ForwardResult.tile_debug.

Each table is a uint8 tensor of 384-byte records indexed by
(expert, gmm_loop): 3 x 128-byte rows of 64 bf16 fp32-accumulated partial
sums — raw Gate [0:128), raw Up [128:256) (both emitted between
WaitForCube and Run) and the SwiGLU result [256:384) (emitted after Run).
Records are deterministic per tile, so a dual run and a single-lane
bisect run of the same shape compare bitwise per record.

usage:
  compare_tile_debug.py DUAL.pt BISECT.pt --expert-per-rank 2 \
      --max-output-size 2048 --n 6144

DUAL.pt / BISECT.pt are torch.save'd `result.tile_debug.cpu()` tensors
(one per rank per run). Mismatched records are grouped by which slots
differ: gate/up-only means the tile arrived corrupted (fixpipe delivery);
y-only (with clean gate/up) means the tile was consumed correctly but the
result diverged afterwards; everything means delivery plus overwrite.
"""

import argparse

import torch

SLOT_BYTES = 128
SLOTS = ("gate", "up", "y")
RECORD_BYTES = 3 * SLOT_BYTES


def load_records(path: str) -> torch.Tensor:
    table = torch.load(path, map_location="cpu")
    assert table.dtype == torch.uint8, f"{path}: expected uint8 bytes, got {table.dtype}"
    assert table.numel() % RECORD_BYTES == 0, f"{path}: size {table.numel()} is not a multiple of {RECORD_BYTES}"
    return table.reshape(-1, RECORD_BYTES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dual", help="dual-run tile_debug tensor (.pt)")
    parser.add_argument("bisect", help="single-lane bisect tile_debug tensor (.pt)")
    parser.add_argument("--expert-per-rank", type=int, required=True)
    parser.add_argument("--max-output-size", type=int, required=True)
    parser.add_argument("--n", type=int, required=True, help="GMM1 N (2 x moe-intermediate)")
    args = parser.parse_args()

    dual = load_records(args.dual)
    bisect = load_records(args.bisect)
    assert dual.shape == bisect.shape, f"table shapes differ: {dual.shape} vs {bisect.shape}"
    if dual.shape[0] == 0:
        print("empty tables (MEGAMOE_BF16_TILE_DEBUG not set?)")
        return

    loops_per_expert = dual.shape[0] // args.expert_per_rank
    print(f"records={dual.shape[0]} experts={args.expert_per_rank} loops_per_expert={loops_per_expert}")

    dual_bf = dual.view(torch.bfloat16).float()
    bisect_bf = bisect.view(torch.bfloat16).float()
    per_slot_neq = (dual_bf != bisect_bf).reshape(dual.shape[0], 3, 64).any(dim=2)

    mismatched = per_slot_neq.any(dim=1).nonzero().flatten()
    print(f"mismatched records: {mismatched.numel()} / {dual.shape[0]}")
    if mismatched.numel() == 0:
        print("tables are bitwise identical — no corruption detected")
        return

    for idx in mismatched.tolist():
        expert, loop = divmod(idx, loops_per_expert)
        slots = "+".join(SLOTS[s] for s in range(3) if per_slot_neq[idx, s])
        print(f"  expert={expert} loop={loop}: {slots}")


if __name__ == "__main__":
    main()
