// This needs to match what's defined in intel16 sram verilog model
`timescale 1ns / 1ps

`define INTEL16_SRAM_MACRO_1r1w_128  ip224rfsbhpm1r1w128x64m1
`define INTEL16_SRAM_MACRO_1r1w_256  ip224rfsbhpm1r1w256x64m1
`define INTEL16_SRAM_MACRO_1r1w_512  ip224rfsbhpm1r1w512x64m1
`define INTEL16_SRAM_MACRO_1r1w_1024 ip224rfsbhpm1r1w1024x64m2
`define INTEL16_SRAM_MACRO_1r1w_2048 ip224rfsbhpm1r1w2048x32m4

module intel16_sram_wrapper_1r1w #(
    parameter int DEPTH = 1024,
    parameter int WIDTH = 512,
    parameter int ADDR_WIDTH = $clog2(DEPTH)
) (
    input wclk,
    input wen,
    input [ADDR_WIDTH - 1:0] waddr,
    input [WIDTH - 1:0] din,
    input rclk,
    input ren,
    input [ADDR_WIDTH - 1:0] raddr,
    output [WIDTH - 1:0] dout
);

  // Select the largest macro 
  localparam int MDEPTH = !(DEPTH % 2048) ? 2048 :
                          !(DEPTH % 1024) ? 1024 :
                          !(DEPTH % 512) ? 512 :
                          !(DEPTH % 256) ? 256 :
                          !(DEPTH % 128) ? 128 : 0;
  // Get its width
  localparam int MWIDTH = MDEPTH == 2048 ? 32 :
                          MDEPTH == 1024 ? 64 :
                          MDEPTH == 512 ? 64 :
                          MDEPTH == 256 ? 64 :
                          MDEPTH == 128 ? 64 : 0;

  localparam int MADDR_WIDTH = $clog2(MDEPTH);

  // top address bits select the macro that contained the accessed addresss
  // for two ports
  logic [ADDR_WIDTH - MADDR_WIDTH - 1 : 0] msel_r, msel_w;
  assign msel_r = raddr >> MADDR_WIDTH;
  assign msel_w = waddr >> MADDR_WIDTH;

  logic [ADDR_WIDTH - MADDR_WIDTH - 1 : 0] msel_r_q;
  // cache the msel for read, because read result come back one cycle late.
  // need to use the cached msel as the MUX select to choose the dout
  always @(posedge rclk) msel_r_q <= msel_r;
  logic [WIDTH-1:0] _dout[DEPTH / MDEPTH];
  assign dout = _dout[msel_r_q];

  for (genvar i = 0; i < DEPTH / MDEPTH; i = i + 1) begin : g_depth
    for (genvar j = 0; j < WIDTH / MWIDTH; j = j + 1) begin : g_width

      wire _wen, _ren;
      assign _wen = wen & (msel_w == i);
      assign _ren = ren & (msel_r == i);

      if (MDEPTH == 2048) begin : g_macro2048
        `INTEL16_SRAM_MACRO_1r1w_2048 mem (
          .ickwp0(wclk),
          .iwenp0(_wen),
          .iawp0(waddr[MADDR_WIDTH-1:0]),
          .idinp0(din[(j+1)*MWIDTH-1:j*MWIDTH]),
          .ickrp0(rclk),
          .irenp0(_ren),
          .iarp0(raddr[MADDR_WIDTH-1:0]),
          .iclkbyp(1'b0),
          .imce(1'b0),
          .irmce(2'b0),
          .ifuse(1'b0),
          .iwmce(4'b0),
          .odoutp0(_dout[i][(j+1)*MWIDTH-1:j*MWIDTH])
        );
      end
      else if (MDEPTH == 1024) begin : g_macro1024
        `INTEL16_SRAM_MACRO_1r1w_1024 mem (
          .ickwp0(wclk),
          .iwenp0(_wen),
          .iawp0(waddr[MADDR_WIDTH-1:0]),
          .idinp0(din[(j+1)*MWIDTH-1:j*MWIDTH]),
          .ickrp0(rclk),
          .irenp0(_ren),
          .iarp0(raddr[MADDR_WIDTH-1:0]),
          .iclkbyp(1'b0),
          .imce(1'b0),
          .irmce(2'b0),
          .ifuse(1'b0),
          .iwmce(4'b0),
          .odoutp0(_dout[i][(j+1)*MWIDTH-1:j*MWIDTH])
        );
      end
      else if (MDEPTH == 512) begin : g_macro512
        `INTEL16_SRAM_MACRO_1r1w_512 mem (
          .ickwp0(wclk),
          .iwenp0(_wen),
          .iawp0(waddr[MADDR_WIDTH-1:0]),
          .idinp0(din[(j+1)*MWIDTH-1:j*MWIDTH]),
          .ickrp0(rclk),
          .irenp0(_ren),
          .iarp0(raddr[MADDR_WIDTH-1:0]),
          .iclkbyp(1'b0),
          .imce(1'b0),
          .irmce(2'b0),
          .ifuse(1'b0),
          .iwmce(4'b0),
          .odoutp0(_dout[i][(j+1)*MWIDTH-1:j*MWIDTH])
        );
      end
      else if (MDEPTH == 256) begin : g_macro256
        `INTEL16_SRAM_MACRO_1r1w_256 mem (
          .ickwp0(wclk),
          .iwenp0(_wen),
          .iawp0(waddr[MADDR_WIDTH-1:0]),
          .idinp0(din[(j+1)*MWIDTH-1:j*MWIDTH]),
          .ickrp0(rclk),
          .irenp0(_ren),
          .iarp0(raddr[MADDR_WIDTH-1:0]),
          .iclkbyp(1'b0),
          .imce(1'b0),
          .irmce(2'b0),
          .ifuse(1'b0),
          .iwmce(4'b0),
          .odoutp0(_dout[i][(j+1)*MWIDTH-1:j*MWIDTH])
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
