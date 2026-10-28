#!/usr/bin/env python3
"""Validate committed golden weight-layout and reload-stride metadata."""

import argparse
import json
from pathlib import Path


WGT_SRAM_BYTES = 80 * 1024
ROW_GROUPS = (64, 32, 16, 8, 4, 2, 1)
DENSE_OPS = (0, 2)
ROW_WEIGHT_OPS = (0, 1, 2)


def expected_stride(meta):
    if meta.get("op_type") not in DENSE_OPS:
        return 0
    if meta.get("dma_wgt_size", 0) <= WGT_SRAM_BYTES:
        return 0
    elem_bytes = 2 if meta.get("data_type", 0) == 1 else 1
    k_bytes = (
        meta.get("kernel_h", 1)
        * meta.get("kernel_w", 1)
        * meta["in_c"]
        * elem_bytes
    )
    for channels in ROW_GROUPS:
        if channels * k_bytes <= WGT_SRAM_BYTES:
            return (channels * k_bytes + 3) // 4
    raise ValueError("one output-channel K vector exceeds weight SRAM")


def check_file(path, fix=False):
    metadata = json.loads(path.read_text())
    errors = []
    changed = False
    for index, meta in enumerate(metadata):
        expected_layout = 1 if meta.get("op_type") in ROW_WEIGHT_OPS else 0
        expected_wpo = expected_stride(meta)
        if meta.get("wgt_layout") != expected_layout:
            errors.append(
                f"{path.parent.name} L{index}: wgt_layout="
                f"{meta.get('wgt_layout')} expected {expected_layout}"
            )
            if fix:
                meta["wgt_layout"] = expected_layout
                changed = True
        if meta.get("wgt_per_oc_words", 0) != expected_wpo:
            errors.append(
                f"{path.parent.name} L{index}: wgt_per_oc_words="
                f"{meta.get('wgt_per_oc_words', 0)} expected {expected_wpo}"
            )
            if fix:
                meta["wgt_per_oc_words"] = expected_wpo
                changed = True
    if changed:
        path.write_text(json.dumps(metadata, indent=2) + "\n")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / "golden_dma_e2e"
    errors = []
    for path in sorted(root.glob("*/metadata.json")):
        if "backup" not in path.parent.name:
            errors.extend(check_file(path, args.fix))
    if errors and not args.fix:
        raise SystemExit("\n".join(errors))
    if args.fix:
        print(f"normalized {len(errors)} metadata fields")
    else:
        print("weight metadata check passed")


if __name__ == "__main__":
    main()
