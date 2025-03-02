#pragma once

#include <mc_connections.h>
#include <systemc.h>

template <class T>
SC_MODULE(Tieoff) {
 public:
  Connections::In<T> CCS_INIT_S1(in);

#ifdef CONNECTIONS_FAST_SIM
  sc_in<bool> CCS_INIT_S1(clk);
  sc_in<bool> CCS_INIT_S1(rstn);
#endif

#ifdef __SYNTHESIS__
  SC_HAS_PROCESS(Tieoff);
  Tieoff()
      : sc_module(sc_gen_unique_name("Tieoff"))
#else
  SC_CTOR(Tieoff)
#endif
  {
#ifdef CONNECTIONS_FAST_SIM
    SC_THREAD(drive_rdy);
    sensitive << clk.pos();
    async_reset_signal_is(rstn, false);
#else
    SC_METHOD(drive_rdy);
    sensitive << in.rdy;

#ifdef CONNECTIONS_SIM_ONLY
    in.disable_spawn();
#endif

#endif
  }

  void drive_rdy() {
#ifdef CONNECTIONS_FAST_SIM
    in.Reset();

    wait();

    while (true) {
      in.Pop();
    }
#else
    in.rdy = 1;
#endif
  }
};
