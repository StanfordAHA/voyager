#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"

#ifdef CHECK_PE
#include "test/checker/PEChecker.h"
#endif

template <typename InputT, typename WeightT, typename OutputT>
SC_MODULE(ProcessingElement) {
 private:
  InputT weight_reg;

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<PEWeight<WeightT> > CCS_INIT_S1(weight_in);
  Connections::Out<PEWeight<WeightT> > CCS_INIT_S1(weight_out);

  Connections::In<PEInput<InputT> > CCS_INIT_S1(input_in);
  Connections::Out<PEInput<InputT> > CCS_INIT_S1(input_out);

  Connections::In<OutputT> CCS_INIT_S1(psum_in);
  Connections::Out<OutputT> CCS_INIT_S1(psum_out);

  Connections::Combinational<WeightT> CCS_INIT_S1(new_weight_fifo);
  Connections::Combinational<WeightT> CCS_INIT_S1(stored_weight);

#ifdef __SYNTHESIS__
  SC_HAS_PROCESS(ProcessingElement);
  ProcessingElement()
      : sc_module(sc_gen_unique_name("ProcessingElement"))
#else
  SC_CTOR(ProcessingElement)
#endif
  {
    SC_THREAD(store_weights);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(weight_fifo);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void weight_fifo() {
    weight_in.Reset();
    weight_out.Reset();
    new_weight_fifo.ResetWrite();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode bubble
    while (true) {
      PEWeight<WeightT> weightStruct = weight_in.Pop();

      if (weightStruct.tag == 0) {
        new_weight_fifo.Push(weightStruct.data);
      } else {
        weightStruct.tag--;
        weight_out.Push(weightStruct);
      }
    }
  }

  void store_weights() {
    stored_weight.ResetWrite();
    new_weight_fifo.ResetRead();

    wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode bubble
    while (true) {
      stored_weight.Push(new_weight_fifo.Pop());
    }
  }

  void run() {
    input_in.Reset();
    psum_in.Reset();
    input_out.Reset();
    psum_out.Reset();

    stored_weight.ResetRead();

    WeightT nextWeight0 = WeightT();
    WeightT nextWeight1 = WeightT();
    WeightT nextWeight2 = WeightT();
    bool swapWeight0 = false;
    bool swapWeight1 = false;
    bool swapWeight2 = false;

    // name is a string separated by '.'
    // get the last part of the string
    // bool debug = std::string(name()) ==
    //              "harness.accelerator.matrixUnit.matrixProcessor.systolicArray."
    //              "systolic_chunk_inst_0.systolic_row_inst_0.pe_inst_0";

    // wait();

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode bubble
    while (true) {
      PEInput<InputT> input_struct = input_in.Pop();

      if (input_struct.swapWeights) {
        weight_reg = stored_weight.Pop();
      }

      OutputT psum = psum_in.Pop();

      input_out.Push(input_struct);

      OutputT output = pe_fma(input_struct.data, weight_reg, psum);
      psum_out.Push(output);
    }
  }

  template <typename T>
  T pe_fma(T input, T weight, T psum) {
    return input * weight + psum;
  }

  OutputT pe_fma(InputT input, WeightT weight, OutputT psum) {
#ifdef CHECK_PE
    std::string inst_name = name();
    int pe_num = std::stoi(inst_name.substr(inst_name.find_last_of('_') + 1));
    pe_checker.checkReference(pe_num, input, weight, psum);
#endif
    return input.fma(weight, psum);
  }
};
