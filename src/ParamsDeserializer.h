#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"
#include "TypeToBits.h"

SC_MODULE(ParamsDeserializer) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<int> CCS_INIT_S1(serialParamsIn);
  Connections::Out<Params> CCS_INIT_S1(paramsOut);
  Connections::Out<VectorParams> CCS_INIT_S1(vectorParamsOut);
  Connections::Out<VectorInstructionConfig> CCS_INIT_S1(vectorInstructionsOut);

  SC_CTOR(ParamsDeserializer) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  template <typename T, unsigned int interfaceWidth>
  T getSerializedParams() {
    ac_int<(T::width + interfaceWidth - 1) / interfaceWidth, false>
        serializedParamsPadded;
    for (int i = 0; i < serializedParamsPadded.width; i++) {
      ac_int<interfaceWidth, false> val = serialParamsIn.Pop();
      serializedParamsPadded.set_slc(i * interfaceWidth, val);
    }
    ac_int<T::width, false> serializedParams =
        serializedParamsPadded.template slc<interfaceWidth>(0);
    sc_lv<T::width> serializedParamsLV;
    type_to_vector(serializedParams, T::width, serializedParamsLV);
    return BitsToType<T>(serializedParamsLV);
  }

  void run() {
    serialParamsIn.Reset();
    paramsOut.Reset();
    vectorParamsOut.Reset();

    wait();
    while (true) {
      // Read params
      Params params = getSerializedParams<Params, 32>();
      paramsOut.Push(params);

      if (serialParamsIn.Pop() == 1) {  // additional parameters
        VectorParams vectorParams = getSerializedParams<VectorParams, 32>();
        vectorParamsOut.Push(vectorParams);

        VectorInstructionConfig vectorInstructionConfig =
            getSerializedParams<VectorInstructionConfig, 32>();
        vectorInstructionsOut.Push(vectorInstructionConfig);
      }
    }
  }
};
