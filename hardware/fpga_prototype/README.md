# Binary-pulse FPGA prototype

The original user RTL targets `xc7z020clg400-2` and a 50 MHz clock. The signal path is PL-side UART reception (115200 bit/s), 800 binary inputs, a fixed-weight `800→7` layer, LIF spike counting over 30 frames, argmax, and LCD display. The top-level module in `top_system.sv` is named `top__system`. Each new sample requires reset because sample-start is generated once after reset release. The design has no hardware STDP, PS7, or AXI data path.

`src/` contains the 12 user RTL modules. Comments were decoded from the original GBK files and the published copies use UTF-8; HDL logic is unchanged. `display_fonts/` contains the retained LCD bitmaps. `weights/` contains the seven original COE weight files. `constraints/` contains the original pin/timing constraints, which are board-specific.

The linear layer instantiates `blk_mem_gen_0` through `blk_mem_gen_6`. These are Vivado Block Memory Generator ROM IPs: each stores 100 words of 64 bits, with eight signed 8-bit weights per word. The corresponding COE initializes each ROM. Regenerate these IPs in Vivado for the target device and preserve the synchronous read timing before building this prototype. Vendor-generated IP output and tools are not bundled.

This original `800→7` prototype is distinct from the C1 multi-bit linear reference and C2 `800→64→7` spiking reference supplied in neighboring directories. It has no corresponding batch board-accuracy, measured power, or physical-latency dataset in this release. No new synthesis or board test is claimed.
