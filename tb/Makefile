# Open-NPU RTL — cocotb Testbench Makefile
# SPDX-License-Identifier: Apache-2.0
#
# Usage:
#   make                — run all tests on npu_pe (default)
#   make DUT=npu_pe     — explicit DUT selection
#   make SIM=icarus     — use Icarus Verilog (default)
#   make SIM=verilator  — use Verilator

# ─── Configuration ───
DUT        ?= npu_pe
SIM        ?= icarus
TOPLEVEL_LANG ?= verilog

# ─── Hardware parameters (override to test different configs) ───
ARRAY_SIZE ?= 16
SPAD_KB    ?= 128

# ─── Paths ───
RTL_DIR    = $(shell pwd)/../src
INC_DIR    = $(shell pwd)/../include

# ─── Verilog sources ───
# For hierarchical modules, include dependencies. npu_pe is always needed.
VERILOG_SOURCES = $(RTL_DIR)/npu_pe.v
ifneq ($(DUT),npu_pe)
VERILOG_SOURCES += $(RTL_DIR)/$(DUT).v
endif

# ─── Compile flags ───
COMPILE_ARGS = -g2012 -I$(INC_DIR) -DARRAY_SIZE=$(ARRAY_SIZE) -DSPAD_KB=$(SPAD_KB)

# ─── cocotb config ───
TOPLEVEL   = $(DUT)
MODULE     = test_$(DUT)

# ─── Include cocotb Makefile ───
include $(shell cocotb-config --makefiles)/Makefile.sim
