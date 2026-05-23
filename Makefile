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
ARRAY_SIZE ?= 16
SPAD_KB    ?= 128

# ─── Paths ───
SRC_DIR    = src
TB_DIR     = tb
SIM_DIR    = sim
INC_DIR    = include
SYN_DIR    = syn

# ─── Source files ───
SRCS       = $(wildcard $(SRC_DIR)/*.v)
TB_SRCS    = $(wildcard $(TB_DIR)/*.v)

# ─── Icarus Verilog ───
IVERILOG   = iverilog
VVP        = vvp
IV_FLAGS   = -g2012 -I$(INC_DIR) -DARRAY_SIZE=$(ARRAY_SIZE) -DSPAD_KB=$(SPAD_KB)

.PHONY: sim sim_verilator syn lint clean

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
syn:
	yosys -p "read_verilog -sv $(SRCS); \
		synth -top $(TOP); \
		stat; \
		write_json $(SYN_DIR)/$(TOP).json" \
		2>&1 | tee $(SYN_DIR)/synth.log

# ─── Lint (Verilator) ───
lint:
	verilator --lint-only -Wall -I$(INC_DIR) \
		-DARRAY_SIZE=$(ARRAY_SIZE) -DSPAD_KB=$(SPAD_KB) \
		$(SRCS)

# ─── Clean ───
clean:
	rm -rf $(SIM_DIR)/*.vvp $(SIM_DIR)/*.vcd $(SIM_DIR)/*.lxt
	rm -rf obj_dir
	rm -f $(SYN_DIR)/*.json $(SYN_DIR)/*.log
