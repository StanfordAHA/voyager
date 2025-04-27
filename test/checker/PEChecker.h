#pragma once
#define NO_SYSC

#include <deque>
#include <tuple>

#include "src/ArchitectureParams.h"

template <typename T>
class Checker {
 public:
  Checker() { reference = new std::deque<T>(); }

  ~Checker() { delete reference; }

  void add_reference(T value) { reference->push_back(value); }

  void check_reference(T value) {
    if (reference->size() == 0) {
      // if weight is 0, then we don't need to check (happens during
      // replication)
      if (static_cast<float>(std::get<1>(value)) == 0) {
        return;
      }
      throw std::runtime_error("No reference values left to check!");
    }

    T ref = reference->front();

    if (ref != value) {
      if (static_cast<float>(std::get<1>(value)) == 0) {
        return;
      }
      std::cout << "Reference: " << std::get<0>(ref) << " " << std::get<1>(ref)
                << " " << std::get<2>(ref) << std::endl;
      std::cout << "Value: " << std::get<0>(value) << " " << std::get<1>(value)
                << " " << std::get<2>(value) << std::endl;
      throw std::runtime_error("Value mismatch");
    }
    reference->pop_front();
  }

 private:
  std::deque<T> *reference;
};

class PEChecker {
 public:
  PEChecker() {
    for (int i = 0; i < IC_DIMENSION * OC_DIMENSION; i++) {
      checkers[i] = new Checker<
          std::tuple<INPUT_DATATYPE, INPUT_DATATYPE, ACCUM_DATATYPE> >();
    }
  }

  ~PEChecker() {
    for (int i = 0; i < IC_DIMENSION * OC_DIMENSION; i++) {
      delete checkers[i];
    }
  }

  void add_reference(int pe_num, INPUT_DATATYPE input, INPUT_DATATYPE weight,
                     ACCUM_DATATYPE accum) {
    checkers[pe_num]->add_reference(std::make_tuple(input, weight, accum));
  }

  void check_reference(int pe_num, INPUT_DATATYPE input, INPUT_DATATYPE weight,
                       ACCUM_DATATYPE accum) {
    checkers[pe_num]->check_reference(std::make_tuple(input, weight, accum));
  }

 private:
  Checker<std::tuple<INPUT_DATATYPE, INPUT_DATATYPE, ACCUM_DATATYPE> >
      *checkers[IC_DIMENSION * OC_DIMENSION];
};

extern PEChecker pe_checker;
