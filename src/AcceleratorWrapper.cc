#include "AcceleratorWrapper.h"

void Accelerator_rtl::update_proc() {
  serialParamsIn.ResetWrite();
  inputAddressRequest.ResetRead();
  inputDataResponse.ResetWrite();
  weightAddressRequest.ResetRead();
  weightDataResponse.ResetWrite();
  vectorFetch0AddressRequest.ResetRead();
  vectorFetch0DataResponse.ResetWrite();
  vectorFetch1AddressRequest.ResetRead();
  vectorFetch1DataResponse.ResetWrite();
  vectorFetch2AddressRequest.ResetRead();
  vectorFetch2DataResponse.ResetWrite();
  vectorOutput.ResetRead();
  vectorOutputAddress.ResetRead();
  scalarUnitOutput.ResetRead();
  scalarOutputAddress.ResetRead();
  startSignal.ResetRead();
  doneSignal.ResetRead();

  wait();

#define IN_UPDATE(signal, type, width)                                        \
  {                                                                           \
    signal                                                                    \
        .Connections::Combinational<type, Connections::MARSHALL_PORT>::in_dat \
        .write(signal##_dat.read());                                          \
    bool vld;                                                                 \
    vector_to_type(signal##_vld.read(), 1, &vld);                             \
    signal                                                                    \
        .Connections::Combinational<type, Connections::MARSHALL_PORT>::in_vld \
        .write(vld);                                                          \
    sc_logic rdy;                                                             \
    type_to_vector(signal                                                     \
                       .Connections::Combinational<                           \
                           type, Connections::MARSHALL_PORT>::out_rdy.read(), \
                   1, rdy);                                                   \
    signal##_rdy.write(rdy);                                                  \
  }

#define OUT_UPDATE(signal, type, width)                                       \
  {                                                                           \
    sc_lv<width> dat;                                                         \
    type_to_vector(signal                                                     \
                       .Connections::Combinational<                           \
                           type, Connections::MARSHALL_PORT>::out_dat.read(), \
                   width, dat);                                               \
    signal##_dat.write(dat);                                                  \
    sc_logic vld;                                                             \
    type_to_vector(signal                                                     \
                       .Connections::Combinational<                           \
                           type, Connections::MARSHALL_PORT>::out_vld.read(), \
                   width, vld);                                               \
    signal##_vld.write(vld);                                                  \
    bool rdy;                                                                 \
    vector_to_type(signal##_rdy.read(), 1, &rdy);                             \
    signal                                                                    \
        .Connections::Combinational<type, Connections::MARSHALL_PORT>::in_rdy \
        .write(rdy);                                                          \
  }

#define SYNC_UPDATE(signal)                                                  \
  {                                                                          \
    sc_logic vld;                                                            \
    type_to_vector(signal.Connections::conn_sync::chan::vld.read(), 1, vld); \
    signal##_vld.write(vld);                                                 \
    bool rdy;                                                                \
    vector_to_type(signal##_rdy.read(), 1, &rdy);                            \
    signal.Connections::conn_sync::chan::rdy.write(rdy);                     \
  }

  while (true) {
    // Serial Params
    IN_UPDATE(serialParamsIn, int, 32);
    OUT_UPDATE(inputAddressRequest, MemoryRequest, 64);
    IN_UPDATE(inputDataResponse, PackedData, 128);
    OUT_UPDATE(weightAddressRequest, MemoryRequest, 64);
    IN_UPDATE(weightDataResponse, PackedData, 128);
    OUT_UPDATE(vectorFetch0AddressRequest, MemoryRequest, 64);
    IN_UPDATE(vectorFetch0DataResponse, PackedData, 128);
    OUT_UPDATE(vectorFetch1AddressRequest, MemoryRequest, 64);
    IN_UPDATE(vectorFetch1DataResponse, PackedData, 128);
    OUT_UPDATE(vectorFetch2AddressRequest, MemoryRequest, 64);
    IN_UPDATE(vectorFetch2DataResponse, PackedData, 128);
    OUT_UPDATE(vectorOutput, PackedData, 128);
    OUT_UPDATE(vectorOutputAddress, int, 32);
    OUT_UPDATE(scalarUnitOutput, PackedData, 128);
    OUT_UPDATE(scalarOutputAddress, int, 32);
    SYNC_UPDATE(startSignal);
    SYNC_UPDATE(doneSignal);

    wait();
  }
}
