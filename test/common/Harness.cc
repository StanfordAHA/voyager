#include "Harness.h"

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"

Harness::Harness(sc_module_name name, std::vector<SimplifiedParams> params_list,
                 INPUT_DATATYPE *sram, INPUT_DATATYPE *rram,
                 MemoryMap memoryMap)
    : sc_module(name),
      clk("clk", 1, SC_NS, 0.5, 0, SC_NS, true),
      params_list(params_list),
      sramMemory(sram),
      rramMemory(rram),
      memoryMap(memoryMap) {
  accelerator.clk(clk);
  accelerator.rstn(rstn);
  accelerator.serialParamsIn(serialParamsIn);

  accelerator.inputAddressRequest(inputAddressRequest);
  accelerator.inputDataResponse(inputDataResponse);
  accelerator.weightAddressRequest(weightAddressRequest);
  accelerator.weightDataResponse(weightDataResponse);
  // FIXME!
  // accelerator.vectorFetchAddressRequest(vectorAddressRequest);
  // accelerator.vectorFetchDataResponse(vectorDataResponse);
  // accelerator.scalarAddressRequest(scalarAddressRequest);
  // accelerator.scalarDataResponse(scalarDataResponse);
  // accelerator.varianceAddressRequest(varianceAddressRequest);
  // accelerator.varianceDataResponse(varianceDataResponse);
  // accelerator.biasAddressRequest(biasAddressRequest);
  // accelerator.biasDataResponse(biasDataResponse);
  // accelerator.residualAddressRequest(residualAddressRequest);
  // accelerator.residualDataResponse(residualDataResponse);
  // accelerator.vectorUnitOutput(vectorOutput);
  // accelerator.outputAddress(vectorOutputAddress);
  accelerator.startSignal(start);
  accelerator.doneSignal(done);

  SC_CTHREAD(reset, clk);

  SC_THREAD(memAccessInputs);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(memAccessWeights);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(memAccessVector);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(memAccessScalar);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(memAccessVariance);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(memAccessBias);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(memAccessResidual);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(storeOutputs);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(sendParams);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(waitForStart);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);

  SC_THREAD(waitForDone);
  sensitive << clk.posedge_event();
  async_reset_signal_is(rstn, false);
}

void Harness::reset() {
  rstn.write(0);
  wait(5);
  rstn.write(1);
  wait();
}

void Harness::memAccessBurst(
    Connections::Combinational<MemoryRequest> *addressRequest,
    Connections::Combinational<Pack1D<INPUT_DATATYPE, DIMENSION> >
        *dataResponse,
    MemorySource memSource) {
  INPUT_DATATYPE *memory = memSource == SRAM ? sramMemory : rramMemory;

  addressRequest->ResetRead();
  dataResponse->ResetWrite();

  wait();

  while (true) {
    MemoryRequest memRequest = addressRequest->Pop();

    for (int b = 0; b < memRequest.burstSize / DIMENSION; b++) {
      Pack1D<INPUT_DATATYPE, DIMENSION> data;
      for (int i = 0; i < DIMENSION; i++) {
        data[i] = memory[memRequest.address + b * DIMENSION + i];
      }
      dataResponse->Push(data);
    }
  }
}

void Harness::memAccessPack(
    Connections::Combinational<int> *addressRequest,
    Connections::Combinational<Pack1D<INPUT_DATATYPE, DIMENSION> >
        *dataResponse,
    MemorySource memSource) {
  INPUT_DATATYPE *memory = memSource == SRAM ? sramMemory : rramMemory;

  addressRequest->ResetRead();
  dataResponse->ResetWrite();

  wait();

  while (true) {
    Pack1D<INPUT_DATATYPE, DIMENSION> data;
    int count = 0;
    while (count < DIMENSION) {
      int address = addressRequest->Pop();
      data[count] = memory[address];
      count++;
    }
    dataResponse->Push(data);
  }
}

void Harness::memAccess(
    Connections::Combinational<int> *addressRequest,
    Connections::Combinational<INPUT_DATATYPE> *dataResponse,
    MemorySource memSource) {
  INPUT_DATATYPE *memory = memSource == SRAM ? sramMemory : rramMemory;

  addressRequest->ResetRead();
  dataResponse->ResetWrite();

  wait();

  while (true) {
    int address = addressRequest->Pop();
    dataResponse->Push(memory[address]);
  }
}

void Harness::memAccessInputs() {
  memAccessBurst(&inputAddressRequest, &inputDataResponse, memoryMap.inputs);
}

void Harness::memAccessWeights() {
  memAccessBurst(&weightAddressRequest, &weightDataResponse, memoryMap.weights);
}

void Harness::memAccessVector() {
  memAccessPack(&vectorAddressRequest, &vectorDataResponse, memoryMap.inputs);
}

void Harness::memAccessScalar() {
  memAccess(&scalarAddressRequest, &scalarDataResponse, memoryMap.inputs);
}

void Harness::memAccessVariance() {
  memAccess(&varianceAddressRequest, &varianceDataResponse, memoryMap.inputs);
}

void Harness::memAccessBias() {
  memAccessBurst(&biasAddressRequest, &biasDataResponse, memoryMap.bias);
}

void Harness::memAccessResidual() {
  memAccessBurst(&residualAddressRequest, &residualDataResponse,
                 memoryMap.residual);
}

template <typename T, unsigned int interfaceWidth>
void sendSerializedParams(T params,
                          Connections::Combinational<int> *serialParamsIn) {
  ac_int<T::width, false> serializedParam;
  vector_to_type(TypeToBits<T>(params), false, &serializedParam);

  ac_int<(T::width + interfaceWidth - 1) / interfaceWidth, false>
      serializedParamsPadded = serializedParam;
  for (int i = 0; i < serializedParamsPadded.width; i++) {
    serialParamsIn->Push(serializedParamsPadded.template slc<interfaceWidth>(
        i * interfaceWidth));
  }
}

void Harness::sendParams() {
  serialParamsIn.ResetWrite();

  wait();

  for (SimplifiedParams params : params_list){
    // create Matrix and Vector Params from SimplifiedParams
    if (params.SOFTMAX) {
      // TODO
    } else if (params.FC) {
      // TODO
    } else if (params.NO_NORM) {
      // TODO
    } else {
      // matrix params
      serialParamsIn.Push(1);

      MatrixParams matrixParams;
      matrixParams.INPUT_OFFSET = params.INPUT_OFFSET;
      matrixParams.WEIGHT_OFFSET = params.WEIGHT_OFFSET;
      matrixParams.OUTPUT_OFFSET = params.OUTPUT_OFFSET;
      matrixParams.SOFTMAX = params.SOFTMAX;
      matrixParams.SCALE = 0;  // unused
      matrixParams.TRANSPOSE = params.TRANSPOSE;
      matrixParams.VECTOR_OFFSET = 0;     // unused
      matrixParams.VEC_OP = 0;            // unused
      matrixParams.VEC_SUB = 0;           // unused
      matrixParams.VEC_SQUARE = 0;        // unused
      matrixParams.VEC_REDUCE = 0;        // unused
      matrixParams.CONST_SCALE = 0;       // unused
      matrixParams.VEC_SCALE_OFFSET = 0;  // unused
      matrixParams.VEC_SUB_OFFSET = 0;    // unused
      matrixParams.RELU = params.RELU;
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
          matrixParams.loops[i][j] = params.loops[i][j];
        }
        matrixParams.inputXLoopIndex[i] = params.inputXLoopIndex[i];
        matrixParams.inputYLoopIndex[i] = params.inputYLoopIndex[i];
        matrixParams.reductionLoopIndex[i] = params.reductionLoopIndex[i];
        matrixParams.weightLoopIndex[i] = params.weightLoopIndex[i];
        matrixParams.weightReuseIndex[i] = params.weightReuseIndex[i];
      }
      matrixParams.fxIndex = params.fxIndex;
      matrixParams.fyIndex = params.fyIndex;
      matrixParams.matMul = false;  // unused
      matrixParams.STRIDE = params.STRIDE;
      matrixParams.REPLICATION = params.REPLICATION;
      matrixParams.MAXPOOL = params.MAXPOOL;
      matrixParams.BIAS = params.BIAS;
      matrixParams.BIAS_OFFSET = params.BIAS_OFFSET;
      matrixParams.RESIDUAL = params.RESIDUAL;
      matrixParams.RESIDUAL_OFFSET = params.RESIDUAL_OFFSET;
      matrixParams.AVGPOOL = params.AVGPOOL;
      sendSerializedParams<MatrixParams, 32>(matrixParams, &serialParamsIn);

      serialParamsIn.Push(0);

      serialParamsIn.Push(1);
      VectorParams vectorParams;

      vectorParams.VECTOR_OFFSET = params.INPUT_OFFSET;
      vectorParams.addressGen0Enable = false;
      vectorParams.ADDRESS_GEN1_OFFSET = params.RESIDUAL_OFFSET;
      vectorParams.addressGen1Mode = params.RESIDUAL;
      vectorParams.addressGen1Loops;  // TODO

      vectorParams.VECTOR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;
      vectorParams.SCALAR_OUTPUT_OFFSET = params.OUTPUT_OFFSET;
      vectorParams.scalarOutputCount = 0;
      vectorParams.MAXPOOL = params.MAXPOOL;
      vectorParams.AVGPOOL = params.AVGPOOL;
      vectorParams.outputLoops;

      sendSerializedParams<VectorParams, 32>(vectorParams, &serialParamsIn);
    }

    serialParamsIn.Push(0);  // FIXME to send vector parameters

    wait();
  }
}

void Harness::storeOutputs() {
  vectorOutput.ResetRead();
  vectorOutputAddress.ResetRead();

  wait();

  while (true) {
    Pack1D<OUTPUT_DATATYPE, DIMENSION> data = vectorOutput.Pop();
    int address = vectorOutputAddress.Pop();
    for (int i = 0; i < DIMENSION; i++) {
      sramMemory[address + i] = data[i];
    }
  }
}

void Harness::waitForStart() {
  start.ResetRead();

  wait();

  int i = 0;
  for (SimplifiedParams params : params_list){
    start.SyncPop();
    CCS_LOG("Accelerator Layer "+ std::to_string(i) +" Started.");
    i++;
  }
}
void Harness::waitForDone() {
  done.ResetRead();
  wait();

  for (SimplifiedParams params : params_list){
    done.SyncPop();
    CCS_LOG("Accelerator Layer Finished.");
  }

  CCS_LOG("Accelerator Finished.");
  sc_stop();
}

void run_op(std::vector<SimplifiedParams> params_list, INPUT_DATATYPE *sramMemory,
            INPUT_DATATYPE *rramMemory, MemoryMap memoryMap) {
  Harness harness("harness", params_list, sramMemory, rramMemory, memoryMap);
  sc_start();
}
