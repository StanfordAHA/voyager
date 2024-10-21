solution options set ComponentLibs/SearchPath {./memories /sim/kzf/birch/chip/catapult/stdcells} -append
# solution options set ComponentLibs/SearchPath {/sim/allpan/intel16_sram_wrappers /sim/kzf/birch/chip/catapult/stdcells} -append

solution library add lib224_b15_7t_108pp_tttt_0p800v_25c_tttt_ctyp_ccslnt_dc -- -rtlsyntool DesignCompiler -vendor Intel -technology 16

catch {
  solution library add intel16_sram_wrapper_1r1w
  solution library add intel16_sram_wrapper_1rw
} res

set memories(1r1w) "intel16_sram_wrapper_1r1w.intel16_sram_wrapper_1r1w"
set memories(1rw)  "intel16_sram_wrapper_1rw.intel16_sram_wrapper_1rw"
