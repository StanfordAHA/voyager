export PD_HOME=${PWD}
export LD_LIBRARY_PATH=/cad/mentor/2024.2/Mgc_home/lib:/cad/mentor/2024.2/Mgc_home/shared/lib:$LD_LIBRARY_PATH

eval `modulecmd bash load base catapult/2024.2 vcs/T-2022.06-SP2 verdi/T-2022.06-SP2 prime/T-2022.03 dc_shell/S-2021.06-SP5-4`

# Put this at the back since catapult-generated vcs makefile has to use this version of gcc but doesn't use absolute path
export VG_GNU_PACKAGE=/cad/synopsys/vcs_gnu_package/S-2021.09/gnu_9/linux
source /cad/synopsys/vcs_gnu_package/S-2021.09/gnu_9/linux/source_me.sh