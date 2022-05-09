#pragma once

#include <mc_connections.h>
#include <mc_scverify.h>

#include "Accelerator.h"

SC_MODULE(Accelerator_rtl) {
  typedef Pack1D<Posit<8, 1>, 16UL> PackedData;

  sc_clock CCS_INIT_S1(sysc_clk);

  // RTL interface
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);
  sc_in<sc_logic> CCS_INIT_S1(serialParamsIn_vld);
  sc_out<sc_logic> CCS_INIT_S1(serialParamsIn_rdy);
  sc_in<sc_lv<32> > CCS_INIT_S1(serialParamsIn_dat);
  sc_out<sc_logic> CCS_INIT_S1(inputAddressRequest_vld);
  sc_in<sc_logic> CCS_INIT_S1(inputAddressRequest_rdy);
  sc_out<sc_lv<64> > CCS_INIT_S1(inputAddressRequest_dat);
  sc_in<sc_logic> CCS_INIT_S1(inputDataResponse_vld);
  sc_out<sc_logic> CCS_INIT_S1(inputDataResponse_rdy);
  sc_in<sc_lv<128> > CCS_INIT_S1(inputDataResponse_dat);
  sc_out<sc_logic> CCS_INIT_S1(weightAddressRequest_vld);
  sc_in<sc_logic> CCS_INIT_S1(weightAddressRequest_rdy);
  sc_out<sc_lv<64> > CCS_INIT_S1(weightAddressRequest_dat);
  sc_in<sc_logic> CCS_INIT_S1(weightDataResponse_vld);
  sc_out<sc_logic> CCS_INIT_S1(weightDataResponse_rdy);
  sc_in<sc_lv<128> > CCS_INIT_S1(weightDataResponse_dat);
  sc_out<sc_logic> CCS_INIT_S1(vectorFetch0AddressRequest_vld);
  sc_in<sc_logic> CCS_INIT_S1(vectorFetch0AddressRequest_rdy);
  sc_out<sc_lv<64> > CCS_INIT_S1(vectorFetch0AddressRequest_dat);
  sc_in<sc_logic> CCS_INIT_S1(vectorFetch0DataResponse_vld);
  sc_out<sc_logic> CCS_INIT_S1(vectorFetch0DataResponse_rdy);
  sc_in<sc_lv<128> > CCS_INIT_S1(vectorFetch0DataResponse_dat);
  sc_out<sc_logic> CCS_INIT_S1(vectorFetch1AddressRequest_vld);
  sc_in<sc_logic> CCS_INIT_S1(vectorFetch1AddressRequest_rdy);
  sc_out<sc_lv<64> > CCS_INIT_S1(vectorFetch1AddressRequest_dat);
  sc_in<sc_logic> CCS_INIT_S1(vectorFetch1DataResponse_vld);
  sc_out<sc_logic> CCS_INIT_S1(vectorFetch1DataResponse_rdy);
  sc_in<sc_lv<128> > CCS_INIT_S1(vectorFetch1DataResponse_dat);
  sc_out<sc_logic> CCS_INIT_S1(vectorFetch2AddressRequest_vld);
  sc_in<sc_logic> CCS_INIT_S1(vectorFetch2AddressRequest_rdy);
  sc_out<sc_lv<64> > CCS_INIT_S1(vectorFetch2AddressRequest_dat);
  sc_in<sc_logic> CCS_INIT_S1(vectorFetch2DataResponse_vld);
  sc_out<sc_logic> CCS_INIT_S1(vectorFetch2DataResponse_rdy);
  sc_in<sc_lv<128> > CCS_INIT_S1(vectorFetch2DataResponse_dat);
  sc_out<sc_logic> CCS_INIT_S1(vectorOutput_vld);
  sc_in<sc_logic> CCS_INIT_S1(vectorOutput_rdy);
  sc_out<sc_lv<128> > CCS_INIT_S1(vectorOutput_dat);
  sc_out<sc_logic> CCS_INIT_S1(vectorOutputAddress_vld);
  sc_in<sc_logic> CCS_INIT_S1(vectorOutputAddress_rdy);
  sc_out<sc_lv<32> > CCS_INIT_S1(vectorOutputAddress_dat);
  sc_out<sc_logic> CCS_INIT_S1(scalarUnitOutput_vld);
  sc_in<sc_logic> CCS_INIT_S1(scalarUnitOutput_rdy);
  sc_out<sc_lv<128> > CCS_INIT_S1(scalarUnitOutput_dat);
  sc_out<sc_logic> CCS_INIT_S1(scalarOutputAddress_vld);
  sc_in<sc_logic> CCS_INIT_S1(scalarOutputAddress_rdy);
  sc_out<sc_lv<32> > CCS_INIT_S1(scalarOutputAddress_dat);
  sc_out<sc_logic> CCS_INIT_S1(startSignal_vld);
  sc_in<sc_logic> CCS_INIT_S1(startSignal_rdy);
  sc_out<sc_logic> CCS_INIT_S1(doneSignal_vld);
  sc_in<sc_logic> CCS_INIT_S1(doneSignal_rdy);

  // SysC interface
  Connections::Combinational<int, Connections::MARSHALL_PORT> CCS_INIT_S1(
      serialParamsIn);
  Connections::Combinational<MemoryRequest, Connections::MARSHALL_PORT>
      CCS_INIT_S1(inputAddressRequest);
  Connections::Combinational<PackedData, Connections::MARSHALL_PORT>
      CCS_INIT_S1(inputDataResponse);
  Connections::Combinational<MemoryRequest, Connections::MARSHALL_PORT>
      CCS_INIT_S1(weightAddressRequest);
  Connections::Combinational<PackedData, Connections::MARSHALL_PORT>
      CCS_INIT_S1(weightDataResponse);
  Connections::Combinational<MemoryRequest, Connections::MARSHALL_PORT>
      CCS_INIT_S1(vectorFetch0AddressRequest);
  Connections::Combinational<PackedData, Connections::MARSHALL_PORT>
      CCS_INIT_S1(vectorFetch0DataResponse);
  Connections::Combinational<MemoryRequest, Connections::MARSHALL_PORT>
      CCS_INIT_S1(vectorFetch1AddressRequest);
  Connections::Combinational<PackedData, Connections::MARSHALL_PORT>
      CCS_INIT_S1(vectorFetch1DataResponse);
  Connections::Combinational<MemoryRequest, Connections::MARSHALL_PORT>
      CCS_INIT_S1(vectorFetch2AddressRequest);
  Connections::Combinational<PackedData, Connections::MARSHALL_PORT>
      CCS_INIT_S1(vectorFetch2DataResponse);
  Connections::Combinational<PackedData, Connections::MARSHALL_PORT>
      CCS_INIT_S1(vectorOutput);
  Connections::Combinational<int, Connections::MARSHALL_PORT> CCS_INIT_S1(
      vectorOutputAddress);
  Connections::Combinational<PackedData, Connections::MARSHALL_PORT>
      CCS_INIT_S1(scalarUnitOutput);
  Connections::Combinational<int, Connections::MARSHALL_PORT> CCS_INIT_S1(
      scalarOutputAddress);
  Connections::SyncChannel CCS_INIT_S1(startSignal);
  Connections::SyncChannel CCS_INIT_S1(doneSignal);

  Accelerator CCS_INIT_S1(accelerator);

  void update_proc();

  SC_CTOR(Accelerator_rtl)
      : sysc_clk("sysc_clk", 5, SC_NS, 2.5, 0, SC_NS, false) {
    accelerator.clk(sysc_clk);
    accelerator.rstn(rstn);
    accelerator.serialParamsIn(serialParamsIn);

    accelerator.inputAddressRequest(inputAddressRequest);
    accelerator.inputDataResponse(inputDataResponse);
    accelerator.weightAddressRequest(weightAddressRequest);
    accelerator.weightDataResponse(weightDataResponse);
    accelerator.vectorFetch0AddressRequest(vectorFetch0AddressRequest);
    accelerator.vectorFetch0DataResponse(vectorFetch0DataResponse);
    accelerator.vectorFetch1AddressRequest(vectorFetch1AddressRequest);
    accelerator.vectorFetch1DataResponse(vectorFetch1DataResponse);
    accelerator.vectorFetch2AddressRequest(vectorFetch2AddressRequest);
    accelerator.vectorFetch2DataResponse(vectorFetch2DataResponse);
    accelerator.vectorOutput(vectorOutput);
    accelerator.vectorOutputAddress(vectorOutputAddress);
    accelerator.scalarUnitOutput(scalarUnitOutput);
    accelerator.scalarOutputAddress(scalarOutputAddress);
    accelerator.startSignal(startSignal);
    accelerator.doneSignal(doneSignal);

#define OUT_SENSIT(signal, type)                                               \
  signal.Connections::Combinational<type, Connections::MARSHALL_PORT>::out_vld \
      << signal.Connections::Combinational<                                    \
             type, Connections::MARSHALL_PORT>::out_dat                        \
      << signal##_rdy

#define IN_SENSIT(signal, type)                      \
  signal##_vld << signal##_dat                       \
               << signal.Connections::Combinational< \
                      type, Connections::MARSHALL_PORT>::in_rdy

#define SYNC_SENSIT(signal) \
  signal##_rdy << signal.Connections::conn_sync::chan::vld

    SC_THREAD(update_proc);
    sensitive << sysc_clk.posedge_event();
    async_reset_signal_is(rstn, false);
    //  << rstn << IN_SENSIT(serialParamsIn, int)
    //           << OUT_SENSIT(inputAddressRequest, MemoryRequest)
    //           << IN_SENSIT(inputDataResponse, PackedData)
    //           << OUT_SENSIT(weightAddressRequest, MemoryRequest)
    //           << IN_SENSIT(weightDataResponse, PackedData)
    //           << OUT_SENSIT(vectorFetch0AddressRequest, MemoryRequest)
    //           << IN_SENSIT(vectorFetch0DataResponse, PackedData)
    //           << OUT_SENSIT(vectorFetch1AddressRequest, MemoryRequest)
    //           << IN_SENSIT(vectorFetch1DataResponse, PackedData)
    //           << OUT_SENSIT(vectorFetch2AddressRequest, MemoryRequest)
    //           << IN_SENSIT(vectorFetch2DataResponse, PackedData)
    //           << OUT_SENSIT(vectorOutput, PackedData)
    //           << OUT_SENSIT(vectorOutputAddress, int)
    //           << OUT_SENSIT(scalarUnitOutput, PackedData)
    //           << OUT_SENSIT(scalarOutputAddress, int)
    //           << SYNC_SENSIT(startSignal) << SYNC_SENSIT(doneSignal);
  }
};
