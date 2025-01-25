solution options set ComponentLibs/SearchPath {/home/shared/catapult/memories /home/shared/catapult/stdcells} -append
solution library add tcbn40ulpbwp40_c170815tt0p9v25c_dc -- -rtlsyntool DesignCompiler -vendor TSMC -technology 40nm

solution library add TS1N40LPB1024X64M4FW_wrapped
solution library add TS1N40LPB1024X128M4FWBA_wrapped
solution library add TS1N40LPB1024X256M4FWBA_wrapped
solution library add TS1N40LPB1024X512M4FWBA_wrapped
solution library add mem_1024x128
solution library add mem_1024x256
solution library add mem_1024x512
solution library add mem_1024x1024

set memories(dp,1024,128) "mem_1024x128.custom1024x128"
set memories(dp,1024,256) "mem_1024x256.custom1024x256"
set memories(dp,1024,512) "mem_1024x512.custom1024x512"
set memories(dp,1024,1024) "mem_1024x1024.custom1024x1024"
set memories(sp,1024,64) "TS1N40LPB1024X64M4FW_wrapped.TS1N40LPB1024X64M4FW_wrapped"
set memories(sp,1024,128) "TS1N40LPB1024X128M4FWBA_wrapped.TS1N40LPB1024X128M4FWBA_wrapped"
set memories(sp,1024,256) "TS1N40LPB1024X256M4FWBA_wrapped.TS1N40LPB1024X256M4FWBA_wrapped"
set memories(sp,1024,512) "TS1N40LPB1024X512M4FWBA_wrapped.TS1N40LPB1024X512M4FWBA_wrapped"

proc get_memory_name {is_sp depth width} {
    global memories
    if {$is_sp == 1} {
        return $memories(sp,${depth},${width})
    } else {
        return $memories(dp,${depth},${width})
    }
}
