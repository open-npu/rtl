import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ReadOnly

@cocotb.test()
async def test_dw_trace(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.ppu_mode.value = 3
    dut.ppu_relu_en.value = 0
    dut.ppu_bias_en.value = 0
    dut.ppu_zp_en.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    dut.u_sram_wgt.mem[0].value = 0x00000003
    dut.u_sram_act.mem[0].value = 0x00000007
    dut.u_sram_param.mem[0].value = 0x00000001
    dut.u_sram_param.mem[1].value = 0
    dut.u_sram_param.mem[2].value = 0
    dut.u_sram_param.mem[3].value = 0
    
    # Verify SRAM content
    await RisingEdge(dut.clk)
    raw_wgt = dut.u_sram_wgt.mem[0].value
    dut._log.info(f"VERIFY: u_sram_wgt.mem[0] = {raw_wgt} ({int(raw_wgt):#010x})")
    
    dut.cfg_op_type.value = 1
    dut.cfg_in_c.value = 1
    dut.cfg_out_c.value = 1
    dut.cfg_out_h.value = 1
    dut.cfg_out_w.value = 1
    dut.cfg_kernel_h.value = 1
    dut.cfg_kernel_w.value = 1
    dut.cfg_stride_h.value = 1
    dut.cfg_stride_w.value = 1
    dut.cfg_pad_top.value = 0
    dut.cfg_pad_left.value = 0
    dut.cfg_tile_h.value = 0
    dut.cfg_tile_w.value = 0
    dut.cfg_tile_num_h.value = 1
    dut.cfg_tile_num_w.value = 1
    dut.cfg_in_w.value = 1
    dut.cfg_in_h.value = 1
    
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    for cyc in range(60):
        await RisingEdge(dut.clk)
        await ReadOnly()
        state = int(dut.u_compute.state.value)
        done = int(dut.done.value)
        
        # Log every cycle for key states
        try:
            wgt_load = int(dut.u_dw_conv.wgt_load.value)
        except:
            wgt_load = "?"
        try:
            wgt_valid = int(dut.u_dw_conv.wgt_valid.value)
        except:
            wgt_valid = "?"
        try:
            wgt_idx = int(dut.u_dw_conv.wgt_idx.value)
        except:
            wgt_idx = "?"
        try:
            acc_clear = int(dut.u_dw_conv.acc_clear.value)
        except:
            acc_clear = "?"
        try:
            in_valid = int(dut.u_dw_conv.in_valid.value)
        except:
            in_valid = "?"
        try:
            in_data_val = int(dut.u_dw_conv.in_data.value)
        except:
            in_data_val = "?"
        try:
            compute_idx = int(dut.u_dw_conv.compute_idx.value)
        except:
            compute_idx = "?"
        try:
            weights_0 = int(dut.u_dw_conv.weights[0].value)
        except:
            weights_0 = "X"
        try:
            out_valid = int(dut.u_dw_conv.out_valid.value)
        except:
            out_valid = "?"
        try:
            acc_out_val = int(dut.u_dw_conv.acc_out.value)
        except:
            acc_out_val = "X"
        try:
            wgt_data_val = int(dut.u_compute.dw_wgt_data.value)
        except:
            wgt_data_val = "X"
        try:
            wgt_rd_data_val = hex(int(dut.u_compute.wgt_rd_data.value))
        except:
            wgt_rd_data_val = "X"
        try:
            wgt_rd_en_val = int(dut.u_compute.wgt_rd_en.value)
        except:
            wgt_rd_en_val = "?"
        try:
            init_phase = int(dut.u_compute.dw_init_phase.value)
        except:
            init_phase = "?"
        try:
            dw_cnt_val = int(dut.u_compute.dw_cnt.value)
        except:
            dw_cnt_val = "?"
        try:
            dw_ri = int(dut.u_compute.dw_read_issued.value)
        except:
            dw_ri = "?"
        
        try:
            wgt_rd_addr_val = int(dut.u_compute.wgt_rd_addr.value)
        except:
            wgt_rd_addr_val = "X"
        try:
            wgt_word_addr_val = int(dut.u_compute.wgt_word_addr.value)
        except:
            wgt_word_addr_val = "X"
        
        dut._log.info(
            f"cyc={cyc:2d} st={state:2d} ph={init_phase} cnt={dw_cnt_val} ri={dw_ri} "
            f"wL={wgt_load} wV={wgt_valid} wI={wgt_idx} aC={acc_clear} "
            f"iV={in_valid} w[0]={weights_0} oV={out_valid} acc={acc_out_val} "
            f"wD={wgt_data_val} rdData={wgt_rd_data_val} rdEn={wgt_rd_en_val} "
            f"rdAddr={wgt_rd_addr_val} wAddr={wgt_word_addr_val}"
        )
        
        if done:
            break
        await Timer(1, unit="step")
    
    for _ in range(5):
        await RisingEdge(dut.clk)
    raw = dut.u_sram_act.mem[0].value
    dut._log.info(f"mem[0] = {raw}")
    # Expected: weight=3, input=7, acc=21, PPU passthrough → output byte = 21
    out_byte = int(raw) & 0xFF
    dut._log.info(f"Output byte = {out_byte}, expected 21")
    assert out_byte == 21, f"Expected 21, got {out_byte}"
