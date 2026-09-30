#!/bin/bash
# Sweep the 64-lane row and the 8x8 mesh over the golden model set.
# Usage: run_util_sweep.sh row|mesh
set -u
cd "$(dirname "$0")"
mode=$1
build=sim_build_util_$mode
[ "$mode" = mesh ] && export FORCE_MESH=1
for m in a b c d e h i j k l; do
    meta=golden/golden_dma_e2e/model_${m}_int8/metadata.json
    [ -f "$meta" ] || { echo "SKIP $m"; continue; }
    UTIL_META=$PWD/$meta timeout 3600 make DUT=npu_compute_tb \
        MODULE=integration.test_array_util SIM_BUILD=$build \
        > /tmp/util2_${m}_${mode}.log 2>&1
    echo "== model_$m $mode"
    grep -E "SUMMARY" /tmp/util2_${m}_${mode}.log
done
