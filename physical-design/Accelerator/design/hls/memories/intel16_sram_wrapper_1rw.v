// This needs to match what's defined in intel16 sram verilog model
`timescale 1ps / 1ps

`define INTEL16_SRAM_MACRO_1rw_512  ip224uhdlp1p11rf_512x32m4b2c1s0_t0r0p0d0a1m1h
`define INTEL16_SRAM_MACRO_1rw_1024 ip224uhdlp1p11rf_1024x64m4b2c1s0_t0r0p0d0a1m1h
`define INTEL16_SRAM_MACRO_1rw_2048 ip224uhdlp1p11rf_2048x64m4b2c1s0_t0r0p0d0a1m1h
`define INTEL16_SRAM_MACRO_1rw_4096 ip224uhdlp1p11rf_4096x64m4b2c1s1_t0r0p0d0a1m1h
`define INTEL16_SRAM_MACRO_1rw_8192 ip224uhdlp1p11rf_8192x8m8b2c1s0_t0r0p0d0a1m1h

module intel16_sram_wrapper_1rw #(
    parameter int DEPTH = 1024,
    parameter int WIDTH = 512,
    parameter int ADDR_WIDTH = $clog2(DEPTH)
) (
    input clk,
    input ren,
    input wen,
    input [ADDR_WIDTH - 1:0] adr,
    input [WIDTH - 1:0] din,
    output [WIDTH - 1:0] q
);

  // Select the largest macro 
  localparam int MDEPTH = !(DEPTH % 8192) ? 8192 :
                          !(DEPTH % 4096) ? 4096 :
                          !(DEPTH % 2048) ? 2048 :
                          !(DEPTH % 1024) ? 1024 :
                          !(DEPTH % 512) ? 512 : 0;
  // Get its width
  localparam int MWIDTH = MDEPTH == 8192 ? 8 :
                          MDEPTH == 4096 ? 64 :
                          MDEPTH == 2048 ? 64 :
                          MDEPTH == 1024 ? 64 :
                          MDEPTH == 512 ? 32 : 0;

  localparam int MADDR_WIDTH = $clog2(MDEPTH);

  // top address bits select the macro that contained the accessed addresss
  logic [ADDR_WIDTH - MADDR_WIDTH - 1 : 0] msel;
  assign msel = adr >> MADDR_WIDTH;

  logic [WIDTH-1:0] _q[DEPTH / MDEPTH];

  logic [ADDR_WIDTH - MADDR_WIDTH - 1 : 0] msel_q;
  // cache the msel for read, because read result come back one cycle late.
  // need to use the cached msel as the MUX select to choose the dout
  always @(posedge clk) msel_q <= msel;
  assign q = _q[msel_q];

  for (genvar i = 0; i < DEPTH / MDEPTH; i = i + 1) begin : g_depth
    for (genvar j = 0; j < WIDTH / MWIDTH; j = j + 1) begin : g_width

      wire _wen, _ren;
      assign _wen = wen & (msel == i);
      assign _ren = ren & (msel == i);

      if (MDEPTH == 8192) begin : g_macro8192
        `INTEL16_SRAM_MACRO_1rw_8192 mem (
            .clk(clk),
            .ren(_ren),
            .wen(_wen),
            .adr(adr[MADDR_WIDTH-1:0]),
            .mc(3'b0),
            .mcen(1'b0),
            .clkbyp(1'b0),
            .din(din[(j+1)*MWIDTH-1:j*MWIDTH]),
            .wa(2'b0),
            .wpulse(2'b0),
            .wpulseen(1'b0),
            .fwen(1'b0),
            .q(_q[i][(j+1)*MWIDTH-1:j*MWIDTH])
        );
      end
      else if (MDEPTH == 4096) begin : g_macro4096
        `INTEL16_SRAM_MACRO_1rw_4096 mem (
            .clk(clk),
            .ren(_ren),
            .wen(_wen),
            .adr(adr[MADDR_WIDTH-1:0]),
            .mc(3'b0),
            .mcen(1'b0),
            .clkbyp(1'b0),
            .din(din[(j+1)*MWIDTH-1:j*MWIDTH]),
            .wbeb(64'b0),
            .wa(2'b0),
            .wpulse(2'b0),
            .wpulseen(1'b0),
            .fwen(1'b0),
            .q(_q[i][(j+1)*MWIDTH-1:j*MWIDTH])
        );
      end
      else if (MDEPTH == 2048) begin : g_macro2048
        `INTEL16_SRAM_MACRO_1rw_2048 mem (
            .clk(clk),
            .ren(_ren),
            .wen(_wen),
            .adr(adr[MADDR_WIDTH-1:0]),
            .mc(3'b0),
            .mcen(1'b0),
            .clkbyp(1'b0),
            .din(din[(j+1)*MWIDTH-1:j*MWIDTH]),
            .wa(2'b0),
            .wpulse(2'b0),
            .wpulseen(1'b0),
            .fwen(1'b0),
            .q(_q[i][(j+1)*MWIDTH-1:j*MWIDTH])
        );
      end
      else if (MDEPTH == 1024) begin : g_macro1024
        `INTEL16_SRAM_MACRO_1rw_1024 mem (
            .clk(clk),
            .ren(_ren),
            .wen(_wen),
            .adr(adr[MADDR_WIDTH-1:0]),
            .mc(3'b0),
            .mcen(1'b0),
            .clkbyp(1'b0),
            .din(din[(j+1)*MWIDTH-1:j*MWIDTH]),
            .wa(2'b0),
            .wpulse(2'b0),
            .wpulseen(1'b0),
            .fwen(1'b0),
            .q(_q[i][(j+1)*MWIDTH-1:j*MWIDTH])
        );
      end
      else if (MDEPTH == 512) begin : g_macro512
        `INTEL16_SRAM_MACRO_1rw_512 mem (
            .clk(clk),
            .ren(_ren),
            .wen(_wen),
            .adr(adr[MADDR_WIDTH-1:0]),
            .mc(3'b0),
            .mcen(1'b0),
            .clkbyp(1'b0),
            .din(din[(j+1)*MWIDTH-1:j*MWIDTH]),
            .wa(2'b0),
            .wpulse(2'b0),
            .wpulseen(1'b0),
            .fwen(1'b0),
            .q(_q[i][(j+1)*MWIDTH-1:j*MWIDTH])
        );
      end
    end
  end

  // -------------------
  // Assertions
  // -------------------
  // pragma translate_off
  // Parameter Checker
    function automatic void check_param;
        assert(MDEPTH != 0); // Depth must be multiple of the available macros
        assert(WIDTH % MWIDTH == 0); // Width must be multiple of the selected macro width
    endfunction

  initial check_param();
  // pragma translate_on

endmodule
