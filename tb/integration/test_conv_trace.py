import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly

def write_sram_word(dut, bank, addr, data):
    if bank == 'wgt':
        dut.u_sram_wgt.mem[addr].value = data
    elif bank == 'act':
        dut.u_sram_act.mem[addr].value = data
    elif bank == 'param':
        dut.u_sram_param.mem[addr].value = data

def pack_i8x4(b0, b1, b2, b3):
    def to_u8(v): return v & 0xFF
    return to_u8(b0) | (to_u8(b1) << 8) | (to_u8(b2) << 16) | (to_u8(b3) << 24)

STATE_NAMES = {0:'IDLE', 1:'TILE_SETUP', 2:'OC_SETUP', 3:'WGT_CMD', 4:'WGT_LOAD',
    5:'WGT_EMIT', 6:'ACT_CMD', 7:'ACT_LOAD', 8:'ACT_EMIT', 9:'ACT_FLUSH',
    10:'DRAIN_CMD', 11:'DRAIN_WAIT', 12:'PARAM_LOAD', 13:'PPU_FEED', 14:'PPU_WAIT',
    15:'WRITEBACK', 16:'OC_NEXT', 17:'TILE_NEXT', 18:'DONE'}

@cocotb.test()
async def test_conv_trace(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0; dut.start.value = 0
    dut.ppu_mode.value = 3; dut.ppu_relu_en.value = 0
    dut.ppu_bias_en.value = 0; dut.ppu_zp_en.value = 0
    for _ in range(5): await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Setup: IC=4, OC=4, 1x1 conv, all weights=1, act=[2,3,4,5]
    for oc in range(4):
        write_sram_word(dut, 'wgt', oc, pack_i8x4(1, 1, 1, 1))
    for i in range(16):
        write_sram_word(dut, 'act', i, 0)
    write_sram_word(dut, 'act', 0, pack_i8x4(2, 3, 4, 5))
    for ch in range(4):
        write_sram_word(dut, 'param', ch*4+0, 0x00000001)
        write_sram_word(dut, 'param', ch*4+1, 0)
        write_sram_word(dut, 'param', ch*4+2, 0)
        write_sram_word(dut, 'param', ch*4+3, 0)

    dut.cfg_op_type.value = 0; dut.cfg_in_c.value = 4
    dut.cfg_out_h.value = 1; dut.cfg_out_w.value = 1; dut.cfg_out_c.value = 4
    dut.cfg_kernel_h.value = 1; dut.cfg_kernel_w.value = 1
    dut.cfg_stride_h.value = 1; dut.cfg_stride_w.value = 1
    dut.cfg_pad_top.value = 0; dut.cfg_pad_left.value = 0
    dut.cfg_tile_h.value = 0; dut.cfg_tile_w.value = 0
    dut.cfg_tile_num_h.value = 1; dut.cfg_tile_num_w.value = 1
    dut.cfg_in_w.value = 1; dut.cfg_in_h.value = 1

    # Allow backdoor writes to settle
    await RisingEdge(dut.clk)

    # Verify SRAM contents before starting
    try:
        act0 = int(dut.u_sram_act.mem[0].value)
        dut._log.info(f"VERIFY act_sram[0] = 0x{act0:08X}")
    except ValueError:
        dut._log.info(f"VERIFY act_sram[0] = X (not initialized!)")

    try:
        wgt0 = int(dut.u_sram_wgt.mem[0].value)
        wgt1 = int(dut.u_sram_wgt.mem[1].value)
        dut._log.info(f"VERIFY wgt_sram[0]=0x{wgt0:08X} [1]=0x{wgt1:08X}")
    except ValueError:
        dut._log.info(f"VERIFY wgt_sram[0] or [1] = X")

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    prev_state = -1
    for cyc in range(2000):
        await RisingEdge(dut.clk)
        await ReadOnly()
        state = int(dut.u_compute.state.value)
        if state != prev_state:
            name = STATE_NAMES.get(state, f'S{state}')
            dut._log.info(f"cyc={cyc:3d} STATE → {name} ({state})")
            prev_state = state

        # Log weight loads
        if int(dut.u_compute.sa_wgt_valid.value) == 1:
            col = int(dut.u_compute.wgt_col_idx.value)
            try:
                w0 = int(dut.u_compute.sa_wgt_data[0].value)
                w1 = int(dut.u_compute.sa_wgt_data[1].value)
                w2 = int(dut.u_compute.sa_wgt_data[2].value)
                w3 = int(dut.u_compute.sa_wgt_data[3].value)
                dut._log.info(f"  WGT_EMIT col={col}: [{w0},{w1},{w2},{w3}]")
            except ValueError:
                dut._log.info(f"  WGT_EMIT col={col}: [X values]")

        # Log activation emits
        if int(dut.u_compute.sa_act_valid.value) == 1:
            try:
                a0 = int(dut.u_compute.sa_act_data[0].value)
                dut._log.info(f"  ACT_EMIT: act_data[0]={a0}")
            except ValueError:
                dut._log.info(f"  ACT_EMIT: act_data[0]=X")

        # Log act SRAM read
        try:
            act_rd_en_val = int(dut.u_compute.act_rd_en.value)
            act_wr_en_val = int(dut.u_compute.act_wr_en.value)
            if act_rd_en_val == 1 or act_wr_en_val == 1:
                try:
                    addr_r = int(dut.u_compute.act_rd_addr.value)
                    addr_r = f"{addr_r}"
                except:
                    addr_r = "X"
                try:
                    addr_w = int(dut.u_compute.act_wr_addr.value)
                    addr_w = f"{addr_w}"
                except:
                    addr_w = "X"
                try:
                    abase = int(dut.u_compute.act_base.value)
                    abase = f"{abase}"
                except:
                    abase = "X"
                try:
                    awaddr = int(dut.u_compute.act_word_addr.value)
                    awaddr = f"{awaddr}"
                except:
                    awaddr = "X"
                dut._log.info(f"  ACT_SRAM rd_en={act_rd_en_val} wr_en={act_wr_en_val} rd_addr={addr_r} wr_addr={addr_w} act_base={abase} act_word_addr={awaddr}")
        except:
            pass

        # Log act_buf and act_rd_data in ACT_LOAD/ACT_EMIT states
        if state in (7, 8):  # ACT_LOAD or ACT_EMIT
            try:
                rd_data = int(dut.u_sram_act.b_rdata.value)
                dut._log.info(f"  act_rd_data=0x{rd_data:08X}")
            except ValueError:
                dut._log.info(f"  act_rd_data=X")
            try:
                abuf = int(dut.u_compute.act_buf.value)
                dut._log.info(f"  act_buf=0x{abuf:08X}")
            except ValueError:
                dut._log.info(f"  act_buf=X")

        # Log PPU feed
        if int(dut.u_compute.ppu_in_valid.value) == 1:
            try:
                acc_val = int(dut.u_compute.ppu_acc_in.value)
                dut._log.info(f"  PPU_FEED: acc={acc_val}")
            except ValueError:
                dut._log.info(f"  PPU_FEED: acc=X")

        # Log PPU output
        try:
            ppu_ov = int(dut.u_ppu.out_valid.value)
            if ppu_ov == 1:
                try:
                    ppu_out = int(dut.u_ppu.out_data.value)
                    dut._log.info(f"  PPU_OUT: {ppu_out}")
                except ValueError:
                    dut._log.info(f"  PPU_OUT: X")
        except:
            pass

        # Log writeback
        if int(dut.u_compute.act_wr_en.value) == 1:
            try:
                wr_addr = int(dut.u_compute.act_wr_addr.value)
                wr_data = int(dut.u_compute.act_wr_data.value)
                dut._log.info(f"  WRITE: addr={wr_addr} data=0x{wr_data:08X}")
            except ValueError:
                dut._log.info(f"  WRITE: addr/data contains X")

        if int(dut.done.value) == 1:
            break
