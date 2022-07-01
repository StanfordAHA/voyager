#pragma once

#include <mc_connections.h>
#include <systemc.h>

template <class T>
SC_MODULE(Tieoff) {
 public:
  Connections::In<T> CCS_INIT_S1(in);

  SC_CTOR(Tieoff) {
    SC_METHOD(drive_rdy);
    sensitive << in.rdy;

#ifdef CONNECTIONS_SIM_ONLY
    in.disable_spawn();
#endif
  }

  void drive_rdy() { in.rdy = 1; }
};