flow package require MemGen
flow run /MemGen/MemoryGenerator_BuildLib {
VENDOR           Intel
RTLTOOL          DesignCompiler
TECHNOLOGY       16
LIBRARY          intel16_sram_wrapper_1r1w
MODULE           intel16_sram_wrapper_1r1w
OUTPUT_DIR       ./memories
FILES {
  { FILENAME memories/intel16_sram_wrapper_1r1w.v    FILETYPE SystemVerilog MODELTYPE generic PARSE 1 PATHTYPE relative STATICFILE 1 VHDL_LIB_MAPS work }
  { FILENAME memories/ip224rfsbhpm1r1w2048x32m4.v    FILETYPE SystemVerilog MODELTYPE generic PARSE 1 PATHTYPE relative STATICFILE 1 VHDL_LIB_MAPS work }
  { FILENAME memories/ip224rfsbhpm1r1w1024x64m2.v    FILETYPE SystemVerilog MODELTYPE generic PARSE 1 PATHTYPE relative STATICFILE 1 VHDL_LIB_MAPS work }
  { FILENAME memories/ip224rfsbhpm1r1w512x128m1.v    FILETYPE SystemVerilog MODELTYPE generic PARSE 1 PATHTYPE relative STATICFILE 1 VHDL_LIB_MAPS work }
  { FILENAME memories/ip224rfsbhpm1r1w256x128m1.v    FILETYPE SystemVerilog MODELTYPE generic PARSE 1 PATHTYPE relative STATICFILE 1 VHDL_LIB_MAPS work }
  { FILENAME memories/ip224rfsbhpm1r1w128x128m1.v    FILETYPE SystemVerilog MODELTYPE generic PARSE 1 PATHTYPE relative STATICFILE 1 VHDL_LIB_MAPS work }
}
VHDLARRAYPATH    {}
LINK_LIBRARY     {}
WRITEDELAY       0.1
INITDELAY        1
READDELAY        0.1
VERILOGARRAYPATH {}
GEN_RAM_PIPE     0
TIMEUNIT         1ns
INPUTDELAY       0.01
WIDTH            WIDTH
RAM_WRAPPER      0
AREA             0
WRITELATENCY     1
RDWRRESOLUTION   UNKNOWN
READLATENCY      1
DEPTH            DEPTH
GEN_EXT_RAM_PIPE 0
PARAMETERS {
  { PARAMETER DEPTH      TYPE hdl IGNORE 0 MIN 128 MAX {} DEFAULT 1024 }
  { PARAMETER WIDTH      TYPE hdl IGNORE 0 MIN 32  MAX {} DEFAULT 512  }
  { PARAMETER ADDR_WIDTH TYPE hdl IGNORE 0 MIN 7   MAX {} DEFAULT 10   }
}
PORTS {
  { NAME port_r MODE Read  }
  { NAME port_w MODE Write }
}
PINMAPS {
  { PHYPIN wclk  LOGPIN CLOCK        DIRECTION in  WIDTH 1.0        PHASE 1  DEFAULT {} PORTS port_w }
  { PHYPIN wen   LOGPIN WRITE_ENABLE DIRECTION in  WIDTH 1.0        PHASE 1  DEFAULT {} PORTS port_w }
  { PHYPIN waddr LOGPIN ADDRESS      DIRECTION in  WIDTH ADDR_WIDTH PHASE {} DEFAULT {} PORTS port_w }
  { PHYPIN din   LOGPIN DATA_IN      DIRECTION in  WIDTH WIDTH      PHASE {} DEFAULT {} PORTS port_w }
  { PHYPIN rclk  LOGPIN CLOCK        DIRECTION in  WIDTH 1.0        PHASE 1  DEFAULT {} PORTS port_r }
  { PHYPIN ren   LOGPIN READ_ENABLE  DIRECTION in  WIDTH 1.0        PHASE 1  DEFAULT {} PORTS port_r }
  { PHYPIN raddr LOGPIN ADDRESS      DIRECTION in  WIDTH ADDR_WIDTH PHASE {} DEFAULT {} PORTS port_r }
  { PHYPIN dout  LOGPIN DATA_OUT     DIRECTION out WIDTH WIDTH      PHASE {} DEFAULT {} PORTS port_r }
}

}
