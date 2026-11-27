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
SPAD_KB    ?= 172
ACC_WIDTH  ?= 44
DMA_BUS_WIDTH ?= 32
DMA_BUS_PROTO ?= 0

DMA_PROTO_FLAG = $(if $(filter 1 axi4,$(DMA_BUS_PROTO)),-DDMA_AXI4 -DDMA_BUS_PROTO=1,-DDMA_BUS_PROTO=0)
HW_DEFINES     = -DARRAY_SIZE=$(ARRAY_SIZE) -DSPAD_KB=$(SPAD_KB) -DACC_WIDTH=$(ACC_WIDTH) \
                 -DDMA_BUS_WIDTH=$(DMA_BUS_WIDTH) $(DMA_PROTO_FLAG)

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
IV_FLAGS   = -g2012 -I$(INC_DIR) $(HW_DEFINES)

.PHONY: sim sim_verilator syn syn-full syn-area lint test-bus-cfg golden-check clean

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
# Full RTL synthesis check (including real npu_compute, npu_ppu_bank, npu_dw_bank).

syn:
	mkdir -p $(SYN_DIR)
	cd $(SYN_DIR) && yosys \
		$(HW_DEFINES) \
		-s npu_synth.ys \
		> synth.log 2>&1 \
		|| { tail -40 synth.log; exit 1; }
	@echo "Yosys full RTL synthesis passed (npu_top, ARRAY_SIZE=$(ARRAY_SIZE), SPAD_KB=$(SPAD_KB), ACC_WIDTH=$(ACC_WIDTH), DMA_BUS_WIDTH=$(DMA_BUS_WIDTH), DMA_BUS_PROTO=$(DMA_BUS_PROTO))"

syn-full: syn

# Backward-compatible spelling.
syn-area: syn-full

# ─── Lint (Verilator) ───
lint:
	verilator --lint-only -Wall -Wno-fatal -I$(INC_DIR) \
		$(HW_DEFINES) \
		--top-module $(TOP) \
		$(SRCS)

test-bus-cfg:
	mkdir -p $(SIM_DIR)
	@for w in 32 64 128; do \
		for p in 0 1; do \
			pflag=$$( [ $$p -eq 1 ] && echo "-DDMA_AXI4 -DDMA_BUS_PROTO=1" || echo "-DDMA_BUS_PROTO=0" ); \
			$(IVERILOG) -g2012 -I$(INC_DIR) -DARRAY_SIZE=$(ARRAY_SIZE) -DSPAD_KB=$(SPAD_KB) -DACC_WIDTH=$(ACC_WIDTH) \
				-DDMA_BUS_WIDTH=$$w $$pflag \
				-o $(SIM_DIR)/bus_cfg_w$${w}_p$${p}.vvp \
				-s tb_npu_bus_cfg $(SRCS) $(TB_DIR)/bus_cfg/tb_npu_bus_cfg.v && \
			$(VVP) $(SIM_DIR)/bus_cfg_w$${w}_p$${p}.vvp || exit 1; \
		done; \
	done

golden-check:
	python3 $(TB_DIR)/golden/check_weight_metadata.py

# ─── Clean ───
clean:
	rm -rf $(SIM_DIR)/*.vvp $(SIM_DIR)/*.vcd $(SIM_DIR)/*.lxt
	rm -rf obj_dir
	rm -f $(SYN_DIR)/*.json $(SYN_DIR)/*.log
