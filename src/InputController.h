#pragma once

#include <mc_connections.h>
#include <systemc.h>

#include "AccelTypes.h"

template <typename DTYPE, int NROWS>
SC_MODULE(InputController) {
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Params> paramsIn;

  Connections::Out<MemoryRequest> CCS_INIT_S1(addressRequest);
  Connections::In<Pack1D<DTYPE, NROWS> > CCS_INIT_S1(dataResponse);

  Connections::Out<int> writeAddress[2];
  Connections::Out<Pack1D<DTYPE, NROWS> > writeData[2];
  Connections::Out<int> writeControl[2];
  Connections::Out<int> readAddress[2];
  Connections::Out<int> readControl[2];

  Connections::Combinational<Params> CCS_INIT_S1(fetcherParams);
  Connections::Combinational<Params> CCS_INIT_S1(writerParams);
  Connections::Combinational<Params> CCS_INIT_S1(readerParams);

  SC_CTOR(InputController) {
    SC_THREAD(read_params);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(fetcher);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(reader);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

    SC_THREAD(writer);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
  }

  void fetcher() {
    addressRequest.Reset();
    fetcherParams.ResetRead();

    wait();

    while (true) {
      Params params = fetcherParams.Pop();

      int FX = params.loops[1][params.fxIndex];
      int FY = params.loops[1][params.fyIndex];

      int loop_counters[2][6];
      int loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }

      // set irrelevant loop bounds to 1
      loop_bounds[1][params.weightLoopIndex[1]] = 1;
      loop_bounds[1][params.fxIndex] = 1;
      loop_bounds[1][params.fyIndex] = 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0; loop_counters[0][0] < loop_bounds[0][0];
           loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0; loop_counters[0][1] < loop_bounds[0][1];
             loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0; loop_counters[0][2] < loop_bounds[0][2];
               loop_counters[0][2]++) {
            // fetching border pixels is a little tricky
            // for the outer tiles, we don't fetch borders (they are known to be
            // 0)

            // reset loop bounds
            loop_bounds[1][params.inputXLoopIndex[1]] =
                params.loops[1][params.inputXLoopIndex[1]] * params.STRIDE;
            loop_bounds[1][params.inputYLoopIndex[1]] =
                params.loops[1][params.inputYLoopIndex[1]] * params.STRIDE;

            int x_min_offset = 0;
            int x_max_offset = 0;
            int y_min_offset = 0;
            int y_max_offset = 0;

            if (loop_counters[0][params.inputXLoopIndex[0]] != 0) {
              x_min_offset = (FX - 1) / 2;
              loop_bounds[1][params.inputXLoopIndex[1]] += (FX - 1) / 2;
            }

            if (loop_counters[0][params.inputXLoopIndex[0]] !=
                loop_bounds[0][params.inputXLoopIndex[0]] - 1) {
              x_max_offset = (FX - 1) / 2;
              loop_bounds[1][params.inputXLoopIndex[1]] += (FX - 1) / 2;
            }

            if (loop_counters[0][params.inputYLoopIndex[0]] != 0) {
              y_min_offset = (FY - 1) / 2;
              loop_bounds[1][params.inputYLoopIndex[1]] += (FY - 1) / 2;
            }

            if (loop_counters[0][params.inputYLoopIndex[0]] !=
                loop_bounds[0][params.inputYLoopIndex[0]] - 1) {
              y_max_offset = (FY - 1) / 2;
              loop_bounds[1][params.inputYLoopIndex[1]] += (FY - 1) / 2;
            }

            // inner memory
            for (loop_counters[1][0] = 0;
                 loop_counters[1][0] < loop_bounds[1][0];
                 loop_counters[1][0]++) {
              for (loop_counters[1][1] = 0;
                   loop_counters[1][1] < loop_bounds[1][1];
                   loop_counters[1][1]++) {
                for (loop_counters[1][2] = 0;
                     loop_counters[1][2] < loop_bounds[1][2];
                     loop_counters[1][2]++) {
                  for (loop_counters[1][3] = 0;
                       loop_counters[1][3] < loop_bounds[1][3];
                       loop_counters[1][3]++) {
                    for (loop_counters[1][4] = 0;
                         loop_counters[1][4] < loop_bounds[1][4];
                         loop_counters[1][4]++) {
                      for (loop_counters[1][5] = 0;
                           loop_counters[1][5] < loop_bounds[1][5];
                           loop_counters[1][5]++) {
                        MemoryRequest memRequest;
                        if (params.matMul) {
                          int m0 = loop_counters[1][params.inputXLoopIndex[1]];
                          int m1 = loop_counters[0][params.inputXLoopIndex[0]];
                          int M0 = params.loops[1][params.inputXLoopIndex[1]];
                          int N1 =
                              params.loops[1][params.reductionLoopIndex[1]];
                          int n1 =
                              loop_counters[1][params.reductionLoopIndex[1]];

                          // change addressing
                          int baseAddress =
                              (m0 + m1 * M0) * (N1 * NROWS) + n1 * NROWS;
                          int burstSize = NROWS;
                          memRequest = {params.INPUT_OFFSET + baseAddress,
                                        burstSize};
                        } else {
                          int x0 = loop_counters[1][params.inputXLoopIndex[1]];
                          int x1 = loop_counters[0][params.inputXLoopIndex[0]];
                          int X0 = params.STRIDE *
                                   params.loops[1][params.inputXLoopIndex[1]];
                          int X1 = params.loops[0][params.inputXLoopIndex[0]];
                          int y0 = loop_counters[1][params.inputYLoopIndex[1]];
                          int y1 = loop_counters[0][params.inputYLoopIndex[0]];
                          int Y0 = params.STRIDE *
                                   params.loops[1][params.inputYLoopIndex[1]];
                          int Y1 = params.loops[0][params.inputYLoopIndex[0]];
                          int c1 =
                              loop_counters[1][params.reductionLoopIndex[1]];
                          int C1 =
                              params.loops[1][params.reductionLoopIndex[1]];

                          int c = c1 * NROWS;
                          int C = C1 * NROWS;

                          int x = (x0 - x_min_offset) + x1 * X0;
                          int X = X0 * X1;

                          int y = (y0 - y_min_offset) + y1 * Y0;
                          int Y = Y0 * Y1;

                          int baseAddress = y * X * C + x * C + c;
                          int burstSize = NROWS;
                          memRequest = {params.INPUT_OFFSET + baseAddress,
                                        burstSize};
                        }

                        // CCS_LOG("memory request: " << memRequest);
                        addressRequest.Push(memRequest);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  void writer() {
    writerParams.ResetRead();
    dataResponse.Reset();

    writeControl[0].Reset();
    writeControl[1].Reset();
    writeAddress[0].Reset();
    writeAddress[1].Reset();
    writeData[0].Reset();
    writeData[1].Reset();

    wait();

    while (true) {
      Params params = writerParams.Pop();

      bool bankSel = 0;

      int FX = params.loops[1][params.fxIndex];
      int FY = params.loops[1][params.fyIndex];
      int fx_bound = (FX - 1) / 2;
      int fy_bound = (FY - 1) / 2;

      int loop_counters[2][6];
      int loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }

      // set irrelevant loop bounds to 1
      loop_bounds[1][params.weightLoopIndex[1]] = 1;
      loop_bounds[1][params.fxIndex] = 1;
      loop_bounds[1][params.fyIndex] = 1;

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0; loop_counters[0][0] < loop_bounds[0][0];
           loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0; loop_counters[0][1] < loop_bounds[0][1];
             loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0; loop_counters[0][2] < loop_bounds[0][2];
               loop_counters[0][2]++) {
            // reset loop bounds
            loop_bounds[1][params.inputXLoopIndex[1]] =
                params.loops[1][params.inputXLoopIndex[1]] * params.STRIDE;
            loop_bounds[1][params.inputYLoopIndex[1]] =
                params.loops[1][params.inputYLoopIndex[1]] * params.STRIDE;

            int x_min_offset = fx_bound;
            int y_min_offset = fy_bound;
            loop_bounds[1][params.inputXLoopIndex[1]] += FX - 1;
            loop_bounds[1][params.inputYLoopIndex[1]] += FY - 1;

            // inner memory
            for (loop_counters[1][0] = 0;
                 loop_counters[1][0] < loop_bounds[1][0];
                 loop_counters[1][0]++) {
              // TODO: make this dynamic
              writeControl[bankSel].Push(loop_bounds[1][1] * loop_bounds[1][2] *
                                         loop_bounds[1][3] * loop_bounds[1][4] *
                                         loop_bounds[1][5]);

              for (loop_counters[1][1] = 0;
                   loop_counters[1][1] < loop_bounds[1][1];
                   loop_counters[1][1]++) {
                for (loop_counters[1][2] = 0;
                     loop_counters[1][2] < loop_bounds[1][2];
                     loop_counters[1][2]++) {
                  for (loop_counters[1][3] = 0;
                       loop_counters[1][3] < loop_bounds[1][3];
                       loop_counters[1][3]++) {
                    for (loop_counters[1][4] = 0;
                         loop_counters[1][4] < loop_bounds[1][4];
                         loop_counters[1][4]++) {
                      for (loop_counters[1][5] = 0;
                           loop_counters[1][5] < loop_bounds[1][5];
                           loop_counters[1][5]++) {
                        Pack1D<DTYPE, NROWS> data;

                        int x0 = loop_counters[1][params.inputXLoopIndex[1]];
                        int x1 = loop_counters[0][params.inputXLoopIndex[0]];
                        int X0 = params.loops[1][params.inputXLoopIndex[1]];
                        int X1 = params.loops[0][params.inputXLoopIndex[0]];

                        int y0 = loop_counters[1][params.inputYLoopIndex[1]];
                        int y1 = loop_counters[0][params.inputYLoopIndex[0]];
                        int Y0 = params.loops[1][params.inputYLoopIndex[1]];
                        int Y1 = params.loops[0][params.inputYLoopIndex[0]];

                        int full_x =
                            (x0 - x_min_offset) + x1 * params.STRIDE * X0;
                        int full_y =
                            (y0 - y_min_offset) + y1 * params.STRIDE * Y0;

                        if ((full_x < 0) || (full_y < 0) ||
                            (full_x >= params.STRIDE * X0 * X1) ||
                            (full_y >= params.STRIDE * Y0 * Y1)) {
                          data = 0;
                          CCS_LOG(full_x << ", " << full_y);
                        } else {
                          data = dataResponse.Pop();
                          CCS_LOG(data);
                        }

                        int address =
                            (y0) * (params.STRIDE * X0 + FX - 1) + (x0);

                        writeAddress[bankSel].Push(address);
                        writeData[bankSel].Push(data);

                        // CCS_LOG("address: " << address << " data: " << data);
                      }
                    }
                  }
                }
              }
              bankSel = !bankSel;
            }
          }
        }
      }
    }
  }

  void reader() {
    readerParams.ResetRead();

    readControl[0].Reset();
    readControl[1].Reset();
    readAddress[0].Reset();
    readAddress[1].Reset();

    wait();

    while (true) {
      Params params = readerParams.Pop();

      int FX = params.loops[1][params.fxIndex];
      int FY = params.loops[1][params.fyIndex];

      bool bankSel = 0;

      int loop_counters[2][6];
      int loop_bounds[2][6];

#pragma hls_unroll yes
      for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
          loop_bounds[i][j] = params.loops[i][j];
        }
      }

#pragma hls_pipeline_init_interval 1
#pragma hls_pipeline_stall_mode flush
      for (loop_counters[0][0] = 0; loop_counters[0][0] < loop_bounds[0][0];
           loop_counters[0][0]++) {
        for (loop_counters[0][1] = 0; loop_counters[0][1] < loop_bounds[0][1];
             loop_counters[0][1]++) {
          for (loop_counters[0][2] = 0; loop_counters[0][2] < loop_bounds[0][2];
               loop_counters[0][2]++) {
            // inner memory
            for (loop_counters[1][0] = 0;
                 loop_counters[1][0] < loop_bounds[1][0];
                 loop_counters[1][0]++) {
              readControl[bankSel].Push(loop_bounds[1][1] * loop_bounds[1][2] *
                                        loop_bounds[1][3] * loop_bounds[1][4] *
                                        loop_bounds[1][5]);

              for (loop_counters[1][1] = 0;
                   loop_counters[1][1] < loop_bounds[1][1];
                   loop_counters[1][1]++) {
                for (loop_counters[1][2] = 0;
                     loop_counters[1][2] < loop_bounds[1][2];
                     loop_counters[1][2]++) {
                  for (loop_counters[1][3] = 0;
                       loop_counters[1][3] < loop_bounds[1][3];
                       loop_counters[1][3]++) {
                    for (loop_counters[1][4] = 0;
                         loop_counters[1][4] < loop_bounds[1][4];
                         loop_counters[1][4]++) {
                      for (loop_counters[1][5] = 0;
                           loop_counters[1][5] < loop_bounds[1][5];
                           loop_counters[1][5]++) {
                        int x0 = loop_counters[1][params.inputXLoopIndex[1]];
                        int X0 = params.loops[1][params.inputXLoopIndex[1]];
                        int y0 = loop_counters[1][params.inputYLoopIndex[1]];
                        int Y0 = params.loops[1][params.inputYLoopIndex[1]];
                        int fx = loop_counters[1][params.fxIndex];
                        int fy = loop_counters[1][params.fyIndex];

                        int x = params.STRIDE * x0 + fx;
                        int y = params.STRIDE * y0 + fy;

                        int address = y * (params.STRIDE * X0 + FX - 1) + x;
                        readAddress[bankSel].Push(address);

                        // CCS_LOG("x: " << x << ", y: " << y << ", fx: " << fx
                        //               << ", fy: " << fy
                        //               << ", address: " << address);
                      }
                    }
                  }
                }
              }
              bankSel = !bankSel;
            }
          }
        }
      }
    }
  }

  void read_params() {
    paramsIn.Reset();
    fetcherParams.ResetWrite();
    writerParams.ResetWrite();
    readerParams.ResetWrite();

    wait();

    while (true) {
      Params params = paramsIn.Pop();

      fetcherParams.Push(params);
      writerParams.Push(params);
      readerParams.Push(params);
    }
  }
};
