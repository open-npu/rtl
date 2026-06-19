# Open-NPU RTL — cocotb Testbench Makefile
# SPDX-License-Identifier: Apache-2.0
#
# Usage:
#   make                          — run unit test on npu_pe (default)
#   make DUT=npu_pe               — explicit DUT selection (auto-finds module)
#   make DUT=npu_top MODULE=e2e.test_npu_dma_e2e  — explicit module path
#   make SIM=icarus               — use Icarus Verilog (default)
#   make SIM=verilator            — use Verilator
#
# Module auto-resolution:
#   If MODULE is not set, the Makefile searches unit/, integration/, e2e/,
#   robustness/ for test_$(DUT).py and sets MODULE accordingly.

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

# ─── Python path ───
# Add tb/ (for utils.*), tb/golden/ (for gen_* imports), and all test dirs
export PYTHONPATH := $(shell pwd):$(shell pwd)/golden:$(shell pwd)/unit:$(shell pwd)/integration:$(shell pwd)/e2e:$(shell pwd)/robustness

# ─── Module auto-resolution ───
# Search for test_$(DUT).py in subdirectories; fall back to bare name
ifneq ($(MODULE),)
  # Explicit MODULE override — use as-is
else
  _FOUND := $(shell find unit integration e2e robustness -name 'test_$(DUT).py' 2>/dev/null | head -1)
  ifneq ($(_FOUND),)
    # Convert path like e2e/test_npu_dma_e2e.py → e2e.test_npu_dma_e2e
    MODULE = $(subst /,.,$(basename $(_FOUND)))
  else
    MODULE = test_$(DUT)
  endif
endif

# ─── Verilog sources ───
# Hierarchical modules need their dependencies included.
ifeq ($(DUT),npu_top)
VERILOG_SOURCES = $(RTL_DIR)/npu_pe.v \
                  $(RTL_DIR)/npu_systolic.v \
                  $(RTL_DIR)/npu_ppu.v \
                  $(RTL_DIR)/npu_csr.v \
                  $(RTL_DIR)/npu_sram.v \
                  $(RTL_DIR)/npu_dma.v \
                  $(RTL_DIR)/npu_dw_conv.v \
                  $(RTL_DIR)/npu_ctrl.v \
                  $(RTL_DIR)/npu_compute.v \
                  $(RTL_DIR)/npu_top.v
else ifeq ($(DUT),npu_compute_tb)
VERILOG_SOURCES = $(RTL_DIR)/npu_pe.v \
                  $(RTL_DIR)/npu_systolic.v \
                  $(RTL_DIR)/npu_ppu.v \
                  $(RTL_DIR)/npu_sram.v \
                  $(RTL_DIR)/npu_dw_conv.v \
                  $(RTL_DIR)/npu_compute.v \
                  $(shell pwd)/npu_compute_tb.v
else ifeq ($(DUT),npu_compute)
VERILOG_SOURCES = $(RTL_DIR)/npu_pe.v \
                  $(RTL_DIR)/npu_systolic.v \
                  $(RTL_DIR)/npu_ppu.v \
                  $(RTL_DIR)/npu_sram.v \
                  $(RTL_DIR)/npu_dw_conv.v \
                  $(RTL_DIR)/npu_compute.v
else ifeq ($(DUT),npu_systolic)
VERILOG_SOURCES = $(RTL_DIR)/npu_pe.v $(RTL_DIR)/npu_systolic.v
else
VERILOG_SOURCES = $(RTL_DIR)/$(DUT).v
endif

# ─── Compile flags ───
ifeq ($(SIM),verilator)
COMPILE_ARGS = +incdir+$(INC_DIR) -DARRAY_SIZE=$(ARRAY_SIZE) -DSPAD_KB=$(SPAD_KB) \
               -Wno-fatal -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-WIDTHCONCAT --trace
ifdef VCD_TRACE
COMPILE_ARGS += +define+VCD_TRACE
endif
else
COMPILE_ARGS = -g2012 -I$(INC_DIR) -DARRAY_SIZE=$(ARRAY_SIZE) -DSPAD_KB=$(SPAD_KB)
endif

# ─── cocotb config ───
TOPLEVEL   = $(DUT)

# ─── Include cocotb Makefile ───
include $(shell cocotb-config --makefiles)/Makefile.sim
