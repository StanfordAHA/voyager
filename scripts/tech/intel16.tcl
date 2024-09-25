solution options set ComponentLibs/SearchPath {./memories /sim/kzf/birch/chip/catapult/stdcells} -append
solution library add lib224_b15_7t_108pp_tttt_0p800v_25c_tttt_ctyp_ccslnt_dc -- -rtlsyntool DesignCompiler -vendor Intel -technology 16

solution library add intel16_sram_wrapper_1024x64_1r1w
solution library add intel16_sram_wrapper_1024x64_1rw

set memories(1r1w) "intel16_sram_wrapper_1024x64_1r1w.intel16_sram_wrapper_1024x64_1r1w"
set memories(1rw)  "intel16_sram_wrapper_1024x64_1rw.intel16_sram_wrapper_1024x64_1rw"
