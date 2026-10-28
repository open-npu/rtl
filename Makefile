# Open-NPU RTL — Makefile
# SPDX-License-Identifier: Apache-2.0
#
# Usage:
#   make sim          — Icarus Verilog simulation
#   make sim_verilator — Verilator C++ simulation (faster)
#   make syn          — bounded Yosys structural sanity check
#   make clean        — Remove generated files

# ─── Configuration ───
TOP        ?= npu_top
FREQ_MHZ   ?= 200
ARRAY_SIZE ?= 8
SPAD_KB    ?= 164
ACC_WIDTH  ?= 44

# ─── Paths ───
SRC_DIR    = src
TB_DIR     = tb
SIM_DIR    = sim
INC_DIR    = include
SYN_DIR    = synth

# ─── Source files ───
SRCS       = $(wildcard $(SRC_DIR)/*.v)
TB_SRCS    = $(wildcard $(TB_DIR)/*.v)

# ─── Icarus Verilog ───
IVERILOG   = iverilog
VVP        = vvp
IV_FLAGS   = -g2012 -I$(INC_DIR) -DARRAY_SIZE=$(ARRAY_SIZE) -DSPAD_KB=$(SPAD_KB) -DACC_WIDTH=$(ACC_WIDTH)

.PHONY: sim sim_verilator syn syn-full syn-area lint clean

sim: $(SIM_DIR)/$(TOP).vvp
	$(VVP) $< -lxt2
	@echo "Waveform: $(SIM_DIR)/$(TOP).vcd"

$(SIM_DIR)/$(TOP).vvp: $(SRCS) $(TB_SRCS)
	$(IVERILOG) $(IV_FLAGS) -o $@ -s tb_$(TOP) $^

# ─── Verilator ───
sim_verilator:
	verilator --cc --exe --build -Wall -I$(INC_DIR) \
		-DARRAY_SIZE=$(ARRAY_SIZE) -DSPAD_KB=$(SPAD_KB) \
		--top-module $(TOP) $(SRCS) $(TB_DIR)/tb_$(TOP)_verilator.cpp
	./obj_dir/V$(TOP)

# ─── Yosys Synthesis ───
#
# Stock Yosys expands the compute row's dynamic multi-write arrays into more
# than 50k mux decoders during `proc`.  That is neither bounded enough for CI
# nor representative of a backend where these arrays are SRAM/register-file
# macros.  Keep the default gate honest and bounded: parse the production
# interfaces, elaborate the top-level connectivity, then run check/stat.

syn:
	mkdir -p $(SYN_DIR)
	cd $(SYN_DIR) && yosys -s npu_synth.ys \
		> synth.log 2>&1 \
		|| { tail -40 synth.log; exit 1; }
	@echo "structural sanity passed ($(TOP), production interfaces)"

# Full technology synthesis needs a backend-specific mapping for the compute
# row memories.  Fail immediately instead of appearing to hang in `proc`.
syn-full:
	@echo "ERROR: full stock-Yosys synthesis is unsupported: map the compute row arrays to backend memory macros first." >&2
	@exit 2

# Backward-compatible spelling.
syn-area: syn-full

# ─── Lint (Verilator) ───
lint:
	verilator --lint-only -Wall -Wno-fatal -I$(INC_DIR) \
		-DARRAY_SIZE=$(ARRAY_SIZE) -DSPAD_KB=$(SPAD_KB) -DACC_WIDTH=$(ACC_WIDTH) \
		$(SRCS)

# ─── Clean ───
clean:
	rm -rf $(SIM_DIR)/*.vvp $(SIM_DIR)/*.vcd $(SIM_DIR)/*.lxt
	rm -rf obj_dir
	rm -f $(SYN_DIR)/*.json $(SYN_DIR)/*.log
