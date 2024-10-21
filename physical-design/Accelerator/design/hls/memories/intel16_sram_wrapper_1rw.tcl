flow package require MemGen
flow run /MemGen/MemoryGenerator_BuildLib {
VENDOR           Intel
RTLTOOL          DesignCompiler
TECHNOLOGY       16
LIBRARY          intel16_sram_wrapper_1rw
MODULE           intel16_sram_wrapper_1rw
OUTPUT_DIR       ./memories
FILES {
  { FILENAME memories/intel16_sram_wrapper_1rw.v                       FILETYPE SystemVerilog MODELTYPE generic PARSE 1 PATHTYPE relative STATICFILE 1 VHDL_LIB_MAPS work }
  { FILENAME memories/ip224uhdlp1p11rf_512x32m4b2c1s0_t0r0p0d0a1m1h.v  FILETYPE SystemVerilog MODELTYPE generic PARSE 1 PATHTYPE relative STATICFILE 1 VHDL_LIB_MAPS work }
  { FILENAME memories/ip224uhdlp1p11rf_1024x64m4b2c1s1_t0r0p0d0a1m1h.v FILETYPE SystemVerilog MODELTYPE generic PARSE 1 PATHTYPE relative STATICFILE 1 VHDL_LIB_MAPS work }
  { FILENAME memories/ip224uhdlp1p11rf_4096x64m4b2c1s1_t0r0p0d0a1m1h.v FILETYPE SystemVerilog MODELTYPE generic PARSE 1 PATHTYPE relative STATICFILE 1 VHDL_LIB_MAPS work }
  { FILENAME memories/ip224uhdlp1p11rf_8192x8m8b2c1s0_t0r0p0d0a1m1h.v  FILETYPE SystemVerilog MODELTYPE generic PARSE 1 PATHTYPE relative STATICFILE 1 VHDL_LIB_MAPS work }
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
  { PARAMETER DEPTH      TYPE hdl IGNORE 0 MIN 512 MAX {} DEFAULT 1024 }
  # this is tricky, for 8192 it can go down to 8, but easier to assume it only supports 64
  { PARAMETER WIDTH      TYPE hdl IGNORE 0 MIN 64  MAX {} DEFAULT 512 }
  { PARAMETER ADDR_WIDTH TYPE hdl IGNORE 0 MIN 9   MAX {} DEFAULT 10   }
}
PORTS {
  { NAME port_0 MODE ReadWrite }
}
PINMAPS {
  { PHYPIN clk LOGPIN CLOCK        DIRECTION in  WIDTH 1.0        PHASE 1  DEFAULT {} PORTS port_0 }
  { PHYPIN ren LOGPIN READ_ENABLE  DIRECTION in  WIDTH 1.0        PHASE 1  DEFAULT {} PORTS port_0 }
  { PHYPIN wen LOGPIN WRITE_ENABLE DIRECTION in  WIDTH 1.0        PHASE 1  DEFAULT {} PORTS port_0 }
  { PHYPIN adr LOGPIN ADDRESS      DIRECTION in  WIDTH ADDR_WIDTH PHASE {} DEFAULT {} PORTS port_0 }
  { PHYPIN din LOGPIN DATA_IN      DIRECTION in  WIDTH WIDTH      PHASE {} DEFAULT {} PORTS port_0 }
  { PHYPIN q   LOGPIN DATA_OUT     DIRECTION out WIDTH WIDTH      PHASE {} DEFAULT {} PORTS port_0 }
}

}
