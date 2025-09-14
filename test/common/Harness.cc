#include "Harness.h"

#include <mc_connections.h>
#include <systemc.h>

#include <cassert>
#include <cstdint>  // for uint32_t
#include <string>

#include "AccelTypes.h"
#include "sysc/kernel/sc_time.h"
#include "lib/nlohmann/json.hpp"


#ifndef CFLOAT
#include "test/toolchain/MapOperation.h"

Harness::Harness(sc_module_name name, std::vector<Operation> operations,
                 char *memory)
    : sc_module(name),
      clk("clk", std::stod(std::getenv("CLOCK_PERIOD")), SC_NS, 0.5, 0, SC_NS,
          true),
      operations(operations),
      memory(memory),
      inputDataResponse_fifo("inputDataResponse_fifo", 1024),
      weightDataResponse_fifo("weightDataResponse_fifo", 1024),
      biasDataResponse_fifo("biasDataResponse_fifo", 1024),
      vectorFetch0DataResponse_fifo("vectorFetch0DataResponse_fifo", 1024),
      vectorFetch1DataResponse_fifo("vectorFetch1DataResponse_fifo", 1024),
      vectorFetch2DataResponse_fifo("vectorFetch2DataResponse_fifo", 1024) {
  accelerator.clk(clk);
  accelerator.rstn(rstn);
  accelerator.serialMatrixParamsIn(serialMatrixParamsIn);
  accelerator.serialVectorParamsIn(serialVectorParamsIn);
  accelerator.inputAddressRequest(inputAddressRequest);
  accelerator.inputDataResponse(inputDataResponse);
  accelerator.weightAddressRequest(weightAddressRequest);
  accelerator.weightDataResponse(weightDataResponse);
  accelerator.biasAddressRequest(biasAddressRequest);
  accelerator.biasDataResponse(biasDataResponse);
  accelerator.vector_fetch_0_request_out(vector_fetch_0_request_out);
  accelerator.vector_fetch_0_resp_in(vector_fetch_0_resp_in);
  accelerator.vector_fetch_1_request_out(vector_fetch_1_request_out);
  accelerator.vector_fetch_1_resp_in(vector_fetch_1_resp_in);
  accelerator.vector_fetch_2_request_out(vector_fetch_2_request_out);
  accelerator.vector_fetch_2_resp_in(vector_fetch_2_resp_in);
  accelerator.vector_fetch_3_request_out(vector_fetch_3_request_out);
  accelerator.vector_fetch_3_resp_in(vector_fetch_3_resp_in);
  accelerator.vector_output(vector_output);
  accelerator.vector_output_address(vector_output_address);
  accelerator.scalar_output(scalar_output);
  accelerator.scalar_output_address(scalar_output_address);

  accelerator.matrixUnitStartSignal(matrixUnitStartSignal);
  accelerator.matrixUnitDoneSignal(matrixUnitDoneSignal);
  accelerator.vectorUnitStartSignal(vectorUnitStartSignal);
  accelerator.vectorUnitDoneSignal(vectorUnitDoneSignal);

#if SUPPORT_MX
  accelerator.inputScaleAddressRequest(inputScaleAddressRequest);
  accelerator.inputScaleDataResponse(inputScaleDataResponse);
  accelerator.weightScaleAddressRequest(weightScaleAddressRequest);
  accelerator.weightScaleDataResponse(weightScaleDataResponse);
#endif

  SC_CTHREAD(reset, clk);

  SC_THREAD(readRequestInputs);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseInputs);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

#if SUPPORT_MX
  SC_THREAD(readRequestInputScale);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseInputScale);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);
#endif

  SC_THREAD(readRequestWeights);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseWeights);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

#if SUPPORT_MX
  SC_THREAD(readRequestWeightScale);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseWeightScale);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);
#endif

  SC_THREAD(readRequestVector0);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseVector0);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(readRequestVector1);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseVector1);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(readRequestVector2);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseVector2);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(readRequestVector3);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseVector3);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(readRequestBias);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendResponseBias);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(storeVectorOutputs);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(storeScalarOutputs);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendParams);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  accessCounter = new AccessCounter();
// do not set access counters for an RTL simulation
#ifndef CCS_DUT_RTL
  accelerator.matrixUnit.inputBuffer.accessCounter = accessCounter;
  accelerator.matrixUnit.weightBuffer.accessCounter = accessCounter;
#endif
}

template <int Width>
void Harness::readMemoryRequest(
    Connections::Combinational<MemoryRequest> *request_out,
    sc_fifo<ac_int<Width, false>> *data_fifo, int channel) {
  request_out->ResetRead();

  constexpr int num_bytes = Width / 8;

  wait();

  while (true) {
    MemoryRequest request = request_out->Pop();

    uint64_t base_address = request.address;
    int total_bytes = request.burst_size;
    int num_words = total_bytes / num_bytes;

    accessCounter->increment(std::string(name()), total_bytes);

    ac_int<Width, false> bits;

    for (int i = 0; i < num_words; i++) {
      for (int j = 0; j < num_bytes; j++) {
        uint64_t address = base_address + i * num_bytes + j;
        DLOG("read addr: " << address << " data: " << memory[address]
                           << std::endl);
                           unsigned char result_byte = memory[address];
        uint32_t result = result_byte;
        // CCS_LOG("read addr: " <<std::hex <<  address << " byte num: " << j << " channel: " << channel << " data: " << std::hex << std::setw(2) << std::setfill('0') << result << std::endl);
        //CCS_LOG("read addr: " <<  address << " byte num: " << j << " channel: " << channel << " data: " << std::hex  << result << std::endl);

        if (channel == 0) {
          input_data_file << std::hex << std::setw(2) << result << std::setfill('0') << std::endl;
        } else if (channel == 3) {
          inputScale_data_file << std::hex << std::setw(2) << result << std::setfill('0') << std::endl;
        } else if (channel == 1) {
          weight_data_file << std::hex << std::setw(2) << result << std::setfill('0') << std::endl;
        } else if (channel == 4) {
          weightScale_data_file << std::hex << std::setw(2) << result << std::setfill('0') << std::endl;
        } else if (channel == 2) {
          bias_data_file << std::hex << std::setw(2) << result << std::setfill('0') << std::endl;
        } else if (channel == 6){
          vectorFetch1_data_file << std::hex << std::setw(2) << result << std::setfill('0') << std::endl;
        }

        bits.set_slc(j * 8, static_cast<ac_int<8, false>>(memory[address]));
      }

      data_fifo->write(bits);
    }
  }
}

template <int Width>
void Harness::sendMemoryResponse(
    sc_fifo<ac_int<Width, false>> *data_fifo,
    Connections::Combinational<ac_int<Width, false>> *response) {
  response->ResetWrite();

  wait();

  while (true) {
    response->Push(data_fifo->read());
  }
}

template <int Width>
void Harness::storeMemoryResponse(
    Connections::Combinational<ac_int<Width, false>> *data_out,
    Connections::Combinational<ac_int<ADDRESS_WIDTH, false>> *address_out) {
  data_out->ResetRead();
  address_out->ResetRead();

  constexpr int num_bytes = Width / 8;

  wait();

  while (true) {
    uint64_t address = address_out->Pop();
    auto data = data_out->Pop();
    DLOG("write address: " << address << " data: " << data);

    accessCounter->increment(std::string(name()) + "_" + "outputs", num_bytes);

    for (int i = 0; i < num_bytes; i++) {
      memory[address + i] = data.template slc<8>(i * 8);
    }
  }
}

void Harness::readRequestInputs() {
  readMemoryRequest(&inputAddressRequest, &inputDataResponse_fifo, 0);
}
void Harness::sendResponseInputs() {
  sendMemoryResponse(&inputDataResponse_fifo, &inputDataResponse);
}

void Harness::readRequestInputScale() {
#if SUPPORT_MX
  readMemoryRequest(&inputScaleAddressRequest, &inputScaleDataResponse_fifo, 3);
#endif
}
void Harness::sendResponseInputScale() {
#if SUPPORT_MX
  sendMemoryResponse(&inputScaleDataResponse_fifo, &inputScaleDataResponse);
#endif
}

void Harness::readRequestWeights() {
  readMemoryRequest(&weightAddressRequest, &weightDataResponse_fifo, 1);
}
void Harness::sendResponseWeights() {
  sendMemoryResponse(&weightDataResponse_fifo, &weightDataResponse);
}

void Harness::readRequestWeightScale() {
#if SUPPORT_MX
  readMemoryRequest(&weightScaleAddressRequest, &weightScaleDataResponse_fifo, 4);
#endif
}
void Harness::sendResponseWeightScale() {
#if SUPPORT_MX
  sendMemoryResponse(&weightScaleDataResponse_fifo, &weightScaleDataResponse);
#endif
}

void Harness::readRequestVector0() {
  readMemoryRequest(&vector_fetch_0_request_out,
                    &vectorFetch0DataResponse_fifo, 5);
}
void Harness::sendResponseVector0() {
  sendMemoryResponse(&vectorFetch0DataResponse_fifo, &vector_fetch_0_resp_in);
}

void Harness::readRequestVector1() {
  readMemoryRequest(&vector_fetch_1_request_out,
                    &vectorFetch1DataResponse_fifo, 6);
}
void Harness::sendResponseVector1() {
  sendMemoryResponse(&vectorFetch1DataResponse_fifo, &vector_fetch_1_resp_in);
}

void Harness::readRequestVector2() {
  readMemoryRequest(&vector_fetch_2_request_out,
                    &vectorFetch2DataResponse_fifo, 7);
}
void Harness::sendResponseVector2() {
  sendMemoryResponse(&vectorFetch2DataResponse_fifo, &vector_fetch_2_resp_in);
}

void Harness::readRequestVector3() {
  readMemoryRequest(&vector_fetch_3_request_out,
                    &vectorFetch3DataResponse_fifo, 8);
}
void Harness::sendResponseVector3() {
  sendMemoryResponse(&vectorFetch3DataResponse_fifo, &vector_fetch_3_resp_in);
}

void Harness::readRequestBias() {
  readMemoryRequest(&biasAddressRequest, &biasDataResponse_fifo, 2);
}
void Harness::sendResponseBias() {
  sendMemoryResponse(&biasDataResponse_fifo, &biasDataResponse);
}

void Harness::storeVectorOutputs() {
  storeMemoryResponse(&vector_output, &vector_output_address);
}

void Harness::storeScalarOutputs() {
  storeMemoryResponse(&scalar_output, &scalar_output_address);
}

void Harness::reset() {
  rstn.write(0);
  wait(5);
  rstn.write(1);
  wait();
}

template <typename T, unsigned int interfaceWidth>
void sendSerializedParams(
    T params,
    Connections::Combinational<ac_int<interfaceWidth, false>> *serialParamsIn) {
  ac_int<T::width, false> serializedParam;
  vector_to_type(TypeToBits<T>(params), false, &serializedParam);

  // round up to the nearest multiple of interfaceWidth
  ac_int<((T::width + interfaceWidth - 1) / interfaceWidth) * interfaceWidth,
         false>
      serializedParamsPadded = serializedParam;

  for (int i = 0; i < serializedParamsPadded.width / interfaceWidth; i++) {
    serialParamsIn->Push(serializedParamsPadded.template slc<interfaceWidth>(
        i * interfaceWidth));
  }
}

// Dumping for AHA flow
template <typename T, unsigned int interfaceWidth>
void dumpSerializedParams(T params) {
    ac_int<T::width, false> serializedParam;
    vector_to_type(TypeToBits<T>(params), false, &serializedParam);

    // round up to the nearest multiple of interfaceWidth
    ac_int<((T::width + interfaceWidth - 1) / interfaceWidth) * interfaceWidth,
          false>
        serializedParamsPadded = serializedParam;

    std::ofstream params_file;
    // Delete the file if it exists
    std::remove("serialized_matrix_params.txt");
    params_file.open("serialized_matrix_params.txt", std::ios::app);
    if (!params_file.is_open()) {
      spdlog::error("Failed to open serialized_matrix_params.txt for writing.");
      return;
    }

    int hex_width = interfaceWidth / 4;
    int num_params = serializedParamsPadded.width / interfaceWidth;
    params_file << "SIZE: " << num_params << std::endl;

    for (int i = 0; i < serializedParamsPadded.width / interfaceWidth; i++) {
      uint64_t param_slice = (serializedParamsPadded.template slc<interfaceWidth>(
        i * interfaceWidth));
      params_file << std::hex << std::setw(hex_width) << std::setfill('0') << param_slice;
      if (i < (serializedParamsPadded.width / interfaceWidth) - 1) {
        params_file << std::endl;
      }
    }
    params_file.close();
}


void Harness::sendParams() {
  matrixUnitStartSignal.ResetRead();
  matrixUnitDoneSignal.ResetRead();
  vectorUnitStartSignal.ResetRead();
  vectorUnitDoneSignal.ResetRead();

  serialMatrixParamsIn.ResetWrite();
  serialVectorParamsIn.ResetWrite();

  wait();

  // Iterate through all params, ie all layers
  for (int i = 0; i < operations.size(); i++) {
    // Dumping for AHA flow
    // Delete the file if it exists
    std::remove("input_data_systemC.txt");
    (this->input_data_file).open("input_data_systemC.txt", std::ios::app);
    if (!(this->input_data_file).is_open()) {
      spdlog::error("Failed to open input_data_systemC.txt for writing.");
      return;
    }

    // Delete the file if it exists
    std::remove("inputScale_data_systemC.txt");
    (this->inputScale_data_file).open("inputScale_data_systemC.txt", std::ios::app);
    if (!(this->inputScale_data_file).is_open()) {
      spdlog::error("Failed to open inputScale_data_systemC.txt for writing.");
      return;
    }

    // Delete the file if it exists
    std::remove("weight_data_systemC.txt");
    (this->weight_data_file).open("weight_data_systemC.txt", std::ios::app);
    if (!(this->weight_data_file).is_open()) {
      spdlog::error("Failed to open weight_data_systemC.txt for writing.");
      return;
    }

    // Delete the file if it exists
    std::remove("weightScale_data_systemC.txt");
    (this->weightScale_data_file).open("weightScale_data_systemC.txt", std::ios::app);
    if (!(this->weightScale_data_file).is_open()) {
      spdlog::error("Failed to open weightScale_data_systemC.txt for writing.");
      return;
    }

    // Delete the file if it exists
    std::remove("bias_data_systemC.txt");
    (this->bias_data_file).open("bias_data_systemC.txt", std::ios::app);
    if (!(this->bias_data_file).is_open()) {
      spdlog::error("Failed to open bias_data_systemC.txt for writing.");
      return;
    }

    // Delete the file if it exists
    std::remove("vectorFetch1_data_systemC.txt");
    (this->vectorFetch1_data_file).open("vectorFetch1_data_systemC.txt", std::ios::app);
    if (!(this->vectorFetch1_data_file).is_open()) {
      spdlog::error("Failed to open bias_data_systemC.txt for writing.");
      return;
    }



    currentOperation = operations.at(i);

    std::deque<AcceleratorMemoryMap> accelerator_memory_maps;
    std::deque<BaseParams *> accelerator_params;
    MapOperation(currentOperation, accelerator_params, accelerator_memory_maps, false, false);

    std::deque<AcceleratorMemoryMap> dump_accelerator_memory_maps;
    std::deque<BaseParams *> dump_accelerator_params;


    const char* zircon_fx_fy_stride_workaround_env = std::getenv("ZIRCON_FX_FY_STRIDE_WORKAROUND");
    const char* zircon_cgra_psum_workaround_env = std::getenv("ZIRCON_CGRA_PSUM_WORKAROUND");
    const char* k_dim_host_tiling_env = std::getenv("K_DIM_HOST_TILING");
    const char* zircon_input_act_padding_workaround_env = std::getenv("ZIRCON_INPUT_ACT_PADDING_WORKAROUND");
    const char* zircon_inner_loop_reduction_workaround_env = std::getenv("ZIRCON_INNER_LOOP_REDUCTION_WORKAROUND");
    const char* zircon_gemm_x_dim_host_tiling_env = std::getenv("ZIRCON_GEMM_X_DIM_HOST_TILING");
    bool dump_tiling = true;
    bool zircon_fx_fy_stride_workaround = zircon_fx_fy_stride_workaround_env && std::stoi(zircon_fx_fy_stride_workaround_env) == 1;
    bool zircon_cgra_psum_workaround = zircon_cgra_psum_workaround_env && std::stoi(zircon_cgra_psum_workaround_env) == 1;
    bool k_dim_host_tiling = k_dim_host_tiling_env && std::stoi(k_dim_host_tiling_env) == 1;
    bool zircon_input_act_padding_workaround = zircon_input_act_padding_workaround_env && std::stoi(zircon_input_act_padding_workaround_env) == 1;
    bool zircon_inner_loop_reduction_workaround = zircon_inner_loop_reduction_workaround_env && std::stoi(zircon_inner_loop_reduction_workaround_env) == 1;
    bool zircon_gemm_x_dim_host_tiling = zircon_gemm_x_dim_host_tiling_env && std::stoi(zircon_gemm_x_dim_host_tiling_env) == 1;
    bool hack_tiling = zircon_fx_fy_stride_workaround || zircon_cgra_psum_workaround || k_dim_host_tiling || zircon_input_act_padding_workaround || zircon_inner_loop_reduction_workaround || zircon_gemm_x_dim_host_tiling;
    // Last two args are dump tiling and hack tiling for the AHA flow
    MapOperation(currentOperation, dump_accelerator_params, dump_accelerator_memory_maps, dump_tiling, hack_tiling);

    int runtime_scale_factor = 1;
    std::cout << "Operation: " << currentOperation.name << std::endl;
    if (currentOperation.has_shrunk_tiling) {
      runtime_scale_factor = currentOperation.shrink_factor;
      std::cout << "Scaling operation by " << runtime_scale_factor << std::endl;
    }

    while (accelerator_params.size() > 0) {
      bool matrixParamsValid, vectorParamsValid;

      BaseParams *baseParam = accelerator_params.front();
      BaseParams *dumpBaseParam = dump_accelerator_params.front();

      MatrixParams *matrixParams = dynamic_cast<MatrixParams *>(baseParam);
      MatrixParams *dumpMatrixParams = dynamic_cast<MatrixParams *>(dumpBaseParam);
      matrixParamsValid = matrixParams != NULL;

      if (matrixParamsValid) {
        accelerator_params.pop_front();
        baseParam = accelerator_params.front();

        dump_accelerator_params.pop_front();
        dumpBaseParam = dump_accelerator_params.front();
      }

      VectorParams *vectorParams = dynamic_cast<VectorParams *>(baseParam);
      VectorInstructionConfig *vectorInstructionConfig;
      vectorParamsValid = vectorParams != NULL;

      if (vectorParamsValid) {
        accelerator_params.pop_front();
        baseParam = accelerator_params.front();

        vectorInstructionConfig =
            dynamic_cast<VectorInstructionConfig *>(baseParam);
        accelerator_params.pop_front();
      }

      if (matrixParamsValid) {
        sendSerializedParams<MatrixParams, 32>(*matrixParams,
                                               &serialMatrixParamsIn);

        // LAYER PARAMS
        std::ifstream tensor_metadata_file("tensor_metadata.json");
        if (!tensor_metadata_file.is_open()) {
          spdlog::error("Failed to open tensor_metadata.json for reading.");
          return;
        }

        nlohmann::json tensor_metadata;
        tensor_metadata_file >> tensor_metadata;

        uint64_t glb_base_addr = tensor_metadata["mu_glb_base_address"].get<uint64_t>();
        printf("\nMU-GLB base address: %d\n", glb_base_addr);
        uint64_t input_offset;
        uint64_t input_scale_offset;
        uint64_t weight_offset;
        uint64_t weight_scale_offset;
        uint64_t bias_offset;

        int input_size = 1;
        int input_scale_size = 1;
        int weight_size = 1;
        int weight_scale_size = 1;

        if (tensor_metadata["has_input"].get<bool>()) {
          input_offset = tensor_metadata["ops"][0]["kwargs"]["input"]["tensor"]["glb_base_address"];
          auto& input_shape = tensor_metadata["ops"][0]["kwargs"]["input"]["tensor"]["shape"];
          for (const auto& dim : input_shape) {
            input_size *= dim.get<int>();
          }
        }

        if (tensor_metadata["has_input_scale"].get<bool>()) {
          input_scale_offset = tensor_metadata["ops"][0]["kwargs"]["input_scale"]["tensor"]["glb_base_address"];
          auto& input_scale_shape = tensor_metadata["ops"][0]["kwargs"]["input_scale"]["tensor"]["shape"];
          for (const auto& dim : input_scale_shape) {
              input_scale_size *= dim.get<int>();
          }
        }

        if (tensor_metadata["has_weight"].get<bool>()) {
          weight_offset = tensor_metadata["ops"][0]["kwargs"]["weight"]["tensor"]["glb_base_address"];
          auto& weight_shape = tensor_metadata["ops"][0]["kwargs"]["weight"]["tensor"]["shape"];
          for (const auto& dim : weight_shape) {
            weight_size *= dim.get<int>();
          }
        }

        if (tensor_metadata["has_weight_scale"].get<bool>()) {
          weight_scale_offset = tensor_metadata["ops"][0]["kwargs"]["weight_scale"]["tensor"]["glb_base_address"];
          auto& weight_scale_shape = tensor_metadata["ops"][0]["kwargs"]["weight_scale"]["tensor"]["shape"];
          for (const auto& dim : weight_scale_shape) {
            weight_scale_size *= dim.get<int>();
          }
        }

        if (tensor_metadata["has_bias"].get<bool>()) {
          bias_offset = tensor_metadata["ops"][0]["kwargs"]["bias"]["tensor"]["glb_base_address"];
        }

        // Adjust the offsets if using zircon CGRA PSUM workaround: account for tiling along reduction dimension
        if (zircon_cgra_psum_workaround) {
          const char* num_psums_env = std::getenv("NUM_PSUMS");
          const char* psum_idx_env = std::getenv("PSUM_IDX");

          printf("NUM_PSUMS: %s\n", num_psums_env);
          printf("PSUM_IDX: %s\n", psum_idx_env);

          int num_psums;
          int psum_idx;

          if (num_psums_env && psum_idx_env) {
            num_psums = std::stoi(num_psums_env);
            psum_idx = std::stoi(psum_idx_env);
          } else {
            throw std::runtime_error(
                "Zircon CGRA PSUM workaround requires NUM_PSUMS and PSUM_IDX "
                "environment variables to be set.");
          }

          input_offset += psum_idx * (input_size / num_psums);;
          input_scale_offset += psum_idx * (input_scale_size/num_psums);
          weight_offset += psum_idx * (weight_size/num_psums);
          weight_scale_offset += psum_idx * (weight_scale_size/num_psums);
        }

        // Print all the offsets and add them to the dumpMatrixParams
        if (tensor_metadata["has_input"].get<bool>()) {
          std::cout << "Input offset: " << input_offset << std::endl;
          dumpMatrixParams->INPUT_OFFSET = input_offset;
        }

        if (tensor_metadata["has_input_scale"].get<bool>()) {
          std::cout << "Input scale offset: " << input_scale_offset << std::endl;
          dumpMatrixParams->INPUT_SCALE_OFFSET = input_scale_offset;
        }

        if (tensor_metadata["has_weight"].get<bool>()) {
          std::cout << "Weight offset: " << weight_offset << std::endl;
          dumpMatrixParams->WEIGHT_OFFSET = weight_offset;
        }

        if (tensor_metadata["has_weight_scale"].get<bool>()) {
          std::cout << "Weight scale offset: " << weight_scale_offset << std::endl;
          dumpMatrixParams->WEIGHT_SCALE_OFFSET = weight_scale_offset;
        }

        if (tensor_metadata["has_bias"].get<bool>()) {
          std::cout << "Bias offset: " << bias_offset << std::endl;
          dumpMatrixParams->BIAS_OFFSET = bias_offset;
        }

        const char* conv1_bias_hack_env = std::getenv("CONV1_BIAS_HACK");
        bool conv1_bias_hack = conv1_bias_hack_env && std::stoi(conv1_bias_hack_env) == 1;
        if (conv1_bias_hack){
          dumpMatrixParams->has_bias = true;
        }

        dumpSerializedParams<MatrixParams, 32>(*dumpMatrixParams);
        matrixUnitStartSignal.SyncPop();
      }

      sc_time start = sc_time_stamp();
      CCS_LOG("----- Accelerator Layer '" << currentOperation.name
                                          << "' Started. -----");

      if (vectorParamsValid) {
        sendSerializedParams<VectorParams, 32>(*vectorParams,
                                               &serialVectorParamsIn);
        sendSerializedParams<VectorInstructionConfig, 32>(
            *vectorInstructionConfig, &serialVectorParamsIn);
        vectorUnitStartSignal.SyncPop();
      }

      CCS_LOG("----- Accelerator Layer '" << currentOperation.name
                                          << "' Started. -----");

      if (matrixParamsValid) {
        matrixUnitDoneSignal.SyncPop();
      }
      if (vectorParamsValid) {
        vectorUnitDoneSignal.SyncPop();
      }
      CCS_LOG("----- Accelerator Layer '" << currentOperation.name
                                          << "' Finished. -----");
      sc_time end = sc_time_stamp();

      std::cout << "Default time unit: " << sc_get_default_time_unit()
                << std::endl;

      int runtime = runtime_scale_factor * int(end.to_default_time_units() -
                                               start.to_default_time_units());
      std::cout << "Runtime: " << runtime << " ns" << std::endl;

      accessCounter->print_summary(currentOperation.tiling,
                                   currentOperation.has_valid_tiling);

      accelerator_memory_maps.pop_front();
    }
  }

  sc_stop();
}

#endif

void run_accelerator(std::vector<Operation> operations, char *memory) {
#ifdef CFLOAT
  spdlog::error(
      "The SystemC model does not support the CFloat datatype. Only the gold "
      "model should be used for CFloat.\n");
  std::abort();
#else
  Harness harness("harness", operations, memory);
  sc_start();
#endif
}
