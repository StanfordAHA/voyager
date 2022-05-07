#pragma once

#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/stringize.hpp>

#include "AccelTypes.h"
#include "ArchitectureParams.h"

#define REPEAT(x) BOOST_PP_REPEAT(DIMENSION, x, 0)

template <typename T, int NUM_REGS>
class Fifo {
 public:
  Fifo() {}

#pragma hls_design interface ccore
  void run(T input, T &output) {
  SHIFT:
    for (int i = NUM_REGS - 1; i >= 0; i--) {
      if (i == 0) {
        regs[i] = input;
      } else {
        regs[i] = regs[i - 1];
      }

      output = regs[NUM_REGS - 1];
    }
  }

 private:
  T regs[NUM_REGS];
};

/*
 * Takes an input of Pack1D<DTYPE, SIZE> and skews it to produce
 * n=SIZE outputs of DTYPE
 */
template <typename DTYPE, int SIZE>
SC_MODULE(SerializedSkewer) {
 private:
  sc_fifo<DTYPE> fifo[SIZE];

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<Pack1D<DTYPE, SIZE> > CCS_INIT_S1(din);
  Connections::Out<DTYPE> dout[SIZE];

  SC_CTOR(SerializedSkewer) {
    SC_THREAD(writeFifos);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

#define SC_THREAD_EXP(x) BOOST_PP_CAT(x, i)
#define SC_THREAD_2(x) SC_THREAD(x)

#define DECL_THREADS(z, i, unused)                                          \
  declare_thread_process(BOOST_PP_CAT(BOOST_PP_CAT(readFifos, i), _handle), \
                         BOOST_PP_STRINGIZE(BOOST_PP_CAT(readFifos, i)),    \
                                            SC_CURRENT_USER_MODULE,         \
                                            BOOST_PP_CAT(readFifos, i));    \
  sensitive << clk.pos();                                                   \
  async_reset_signal_is(rstn, false);

    REPEAT(DECL_THREADS)
#undef DECL_THREADS
  }

  void writeFifos() {
    din.Reset();

    wait();

    while (true) {
      Pack1D<DTYPE, SIZE> input = din.Pop();

#pragma hls_unroll yes
      for (int i = 0; i < SIZE; i++) {
        fifo[i].write(input[i]);
      }
    }
  }

#define DECL_FUNCS(z, i, unused)      \
  void BOOST_PP_CAT(readFifos, i)() { \
    dout[i].Reset();                  \
    wait();                           \
    while (true) {                    \
      dout[i].Push(fifo[i].read());   \
    }                                 \
  }

  REPEAT(DECL_FUNCS)
#undef DECL_FUNCS
};

/*
 * Takes an input of n=SIZE outputs of DTYPE and skews it to produce
 * an output Pack1D<DTYPE, SIZE>
 */
template <typename DTYPE, int SIZE>
SC_MODULE(DeserializedSkewer) {
 private:
  sc_fifo<DTYPE> fifo[SIZE];

 public:
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);

  Connections::In<DTYPE> din[SIZE];
  Connections::Out<Pack1D<DTYPE, SIZE> > CCS_INIT_S1(dout);

  SC_CTOR(DeserializedSkewer) {
    SC_THREAD(readFifos);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);

#define DECL_THREADS(z, i, unused)                                           \
  declare_thread_process(BOOST_PP_CAT(BOOST_PP_CAT(writeFifos, i), _handle), \
                         BOOST_PP_STRINGIZE(BOOST_PP_CAT(writeFifos, i)),    \
                                            SC_CURRENT_USER_MODULE,          \
                                            BOOST_PP_CAT(writeFifos, i));    \
  sensitive << clk.pos();                                                    \
  async_reset_signal_is(rstn, false);

    REPEAT(DECL_THREADS)
#undef DECL_THREADS
  }

  void readFifos() {
    dout.Reset();

    wait();

    while (true) {
      Pack1D<DTYPE, SIZE> output;

#pragma hls_unroll yes
      for (int i = 0; i < SIZE; i++) {
        output[i] = fifo[i].read();
      }

      dout.Push(output);
    }
  }

#define DECL_FUNCS(z, i, unused)       \
  void BOOST_PP_CAT(writeFifos, i)() { \
    din[i].Reset();                    \
    wait();                            \
    while (true) {                     \
      fifo[i].write(din[i].Pop());     \
    }                                  \
  }

  REPEAT(DECL_FUNCS)
#undef DECL_FUNCS
};
