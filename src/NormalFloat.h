#pragma once

// clang-format off
#include <ac_std_float.h>
// clang-format on

class NormalFloat4 {
 public:
  typedef ac_int<4, false> ac_int_rep;
  static constexpr unsigned int width = 4;

  typedef Int<6, true> Decoded;

  static inline const ac_int<6, true> code[16] = {
      -31, -22, -16, -12, -9, -6, -3, 0, 2, 5, 8, 10, 14, 17, 22, 31};

  ac_int_rep index;

  NormalFloat4() {}
  NormalFloat4(const ac_int_rep &other) { index = other; }

#ifndef __SYNTHESIS__
  NormalFloat4(const float other) { index = other; }
#endif

  template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
            ac_q_mode Q>
  NormalFloat4(
      const StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q> &other);

  ac_int<4, false> bits_rep() { return index; }

  static Decoded max() {
    ac_int<6, true> r(31);
    return Decoded(r);
  }

  void set_bits(int i) { index = i; }

  void set_zero() { index = 7; }

  template <int W, bool S>
  operator Int<W, S>() const {
    return code[index];
  }

  operator float() const { return code[index]; }

  template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
            ac_q_mode Q>
  operator StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>() const {
    using FloatType = StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q>;
    FloatType f;
    f.float_val = typename FloatType::ac_float_rep(code[index]);
    return f;
  }

#ifndef NO_SYSC
  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    m & index;
  }
#endif
};

template <int mantissa, int exp, bool useDWImpl, bool ieee_compliance,
          ac_q_mode Q>
NormalFloat4::NormalFloat4(
    const StdFloat<mantissa, exp, useDWImpl, ieee_compliance, Q> &other) {
  using FloatType = typename StdFloat<mantissa, exp, useDWImpl, ieee_compliance,
                                      Q>::ac_float_rep;

  const FloatType float_val = other.float_val;

  if (float_val <= FloatType(1.0)) {
    if (float_val <= FloatType(-10.5)) {
      if (float_val <= FloatType(-19.0)) {
        index = float_val <= FloatType(-26.5) ? 0 : 1;
      } else {
        index = float_val <= FloatType(-14.0) ? 2 : 3;
      }
    } else {
      if (float_val <= FloatType(-4.5)) {
        index = float_val <= FloatType(-7.5) ? 4 : 5;
      } else {
        index = float_val <= FloatType(-1.5) ? 6 : 7;
      }
    }
  } else {
    if (float_val <= FloatType(12.0)) {
      if (float_val <= FloatType(6.5)) {
        index = float_val <= FloatType(3.5) ? 8 : 9;
      } else {
        index = float_val <= FloatType(9.0) ? 10 : 11;
      }
    } else {
      if (float_val <= FloatType(19.5)) {
        index = float_val <= FloatType(15.5) ? 12 : 13;
      } else {
        index = float_val <= FloatType(26.5) ? 14 : 15;
      }
    }
  }
}
