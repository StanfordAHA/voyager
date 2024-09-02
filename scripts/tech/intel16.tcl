solution options set ComponentLibs/SearchPath {/sim/kzf/birch/chip/catapult/memories /sim/kzf/birch/chip/catapult/stdcells} -append
solution library add lib224_b15_7t_108pp_tttt_0p800v_25c_tttt_ctyp_ccslnt_dc -- -rtlsyntool DesignCompiler -vendor Intel -technology 16

solution library add intel16_1024x256_rf_wrapper
solution library add intel16_1024x256_sram_wrapper
solution library add intel16_1024x512_rf_wrapper

#set memories(1024,128) "mem_1024x128.custom1024x128"
set memories(sp,1024,256) "intel16_1024x256_sram_wrapper"
set memories(dp,1024,256) "intel16_1024x256_rf_wrapper.intel16_1024x256_rf_wrapper"
set memories(dp,1024,512) "intel16_1024x512_rf_wrapper.intel16_1024x512_rf_wrapper"
#set memories(1024,1024) "mem_1024x1024.custom1024x1024"
