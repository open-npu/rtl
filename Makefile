# Open-NPU RTL — Makefile
# SPDX-License-Identifier: Apache-2.0
#
# Usage:
#   make sim          — Icarus Verilog simulation
#   make sim_verilator — Verilator C++ simulation (faster)
#   make syn          — Yosys synthesis (area/timing report)
#   make clean        — Remove generated files

# ─── Configuration ───
TOP        ?= npu_top
FREQ_MHZ   ?= 200
ARRAY_SIZE ?= 8
SPAD_KB    ?= 192
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

.PHONY: sim sim_verilator syn syn-area lint clean

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
# Two targets, because a full tech-map is far too slow to gate a regression on:
# `synth` runs memory_map, which expands the $(SPAD_KB)KB scratchpad into
# ~1.5M flip-flops and then hands that to ABC. That ran >1h without finishing,
# and it is not physically meaningful either — the scratchpad is an SRAM macro
# or BRAM in any real flow, never registers.
#
#   syn      — fast synthesizability gate (~30s). This is what CI should run.
#   syn-area — full tech-map for area numbers, scratchpad blackboxed.
#
# Both write the log first and replay it. Piping yosys into tee would hand make
# tee's exit status, so synthesis errors used to pass silently.

syn:
	mkdir -p $(SYN_DIR)
	yosys -p "read_verilog -sv -I$(INC_DIR) -DARRAY_SIZE=$(ARRAY_SIZE) -DSPAD_KB=$(SPAD_KB) -DACC_WIDTH=$(ACC_WIDTH) $(SRCS); \
		blackbox npu_sram_wide; \
		hierarchy -top $(TOP); \
		proc; \
		opt_clean; \
		check -assert" \
		> $(SYN_DIR)/synth.log 2>&1 \
		|| { tail -40 $(SYN_DIR)/synth.log; exit 1; }
	@echo "synthesizability check passed ($(TOP), ARRAY_SIZE=$(ARRAY_SIZE), SPAD_KB=$(SPAD_KB))"

syn-area:
	mkdir -p $(SYN_DIR)
	yosys -p "read_verilog -sv -I$(INC_DIR) -DARRAY_SIZE=$(ARRAY_SIZE) -DSPAD_KB=$(SPAD_KB) -DACC_WIDTH=$(ACC_WIDTH) $(SRCS); \
		blackbox npu_sram; \
		blackbox npu_sram_wide; \
		hierarchy -top $(TOP); \
		synth -top $(TOP); \
		stat; \
		write_json $(SYN_DIR)/$(TOP).json" \
		> $(SYN_DIR)/area.log 2>&1 \
		|| { tail -40 $(SYN_DIR)/area.log; exit 1; }
	@sed -n '/Printing statistics/,$$p' $(SYN_DIR)/area.log

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
