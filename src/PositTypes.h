#pragma once

#include <ac_int.h>

inline int max(int a, int b) { return a > b ? a : b; }

class PositFP;

class Posit {
 public:
  Posit() {}
  Posit(const PositFP &input);
  Posit(const float &f);
  Posit(int i) { bits = i; }

  // overridden operators
  Posit &operator+=(const Posit &rhs);
  Posit &operator-=(const Posit &rhs);
  Posit &operator*=(const Posit &rhs);
  Posit operator*(const Posit &rhs);
  bool operator<(const Posit &rhs) const;
  operator float() const;

  ac_int<8, false> bits;
  static const unsigned int width = 8;

  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    m &bits;
  }

  bool isNegative() const { return bits.slc<1>(width - 1) == 1; }
  void setZero() { bits = 0; }

  inline friend void sc_trace(sc_trace_file *tf, const Posit &posit,
                              const std::string &name) {
    // TODO
  }

  inline friend std::ostream &operator<<(ostream &os, const Posit &posit) {
    os << float(posit);
    return os;
  }

 private:
  static const int nbits = 8;
  static const int es = 1;
};

class PositFP {
 public:
  PositFP() {}
  PositFP(const Posit &input);
  PositFP(int i) { setZero(); }

  bool isZero() const { return get_scale() == -127; }
  void setZero() {
    set_fraction(0);
    set_scale(-127);
  }

  PositFP &operator+=(const PositFP &rhs);
  PositFP &operator+=(const Posit &rhs);

  PositFP operator*(const PositFP &op) {
    PositFP result;
    result.set_sign(get_sign() ^ op.get_sign());
    result.set_scale(get_scale() + op.get_scale());

    result.set_fraction((get_fraction() >> 8) * (op.get_fraction() >> 8));

    if (result.get_fraction()[15]) {
      result.set_scale(result.get_scale() + 1);
    } else {
      result.set_fraction(result.get_fraction() << 1);
    }

    return result;
  }

  PositFP operator+(const PositFP &op) {
    PositFP result;
    // align the fraction
    int result_scale = max(get_scale(), op.get_scale());
    ac_int<16, false> r1 = get_fraction() >> (result_scale - get_scale() + 1);
    ac_int<16, false> r2 =
        op.get_fraction() >> (result_scale - op.get_scale() + 1);
    ac_int<16, false> sum;

    int shift = 0;
    if (get_sign() == op.get_sign()) {
      result.set_sign(get_sign());
      sum = r1 + r2;
      if (sum[15]) shift = -1;
    } else {
      if (r1 > r2) {
        sum = r1 - r2;
        result.set_sign(get_sign());
      } else if (r1 < r2) {
        sum = r2 - r1;
        result.set_sign(op.get_sign());
      } else {
        result.setZero();
        return result;
      }
      shift = 14 - fsb16(sum);
    }

    if (shift > 16) {
      result.setZero();
      return result;
    }

    result.set_scale(result_scale - shift);
    result.set_fraction(sum << (shift + 1));

    return result;
  }

  PositFP operator-(const PositFP &op) {
    PositFP negOp = op;
    negOp.set_sign(!negOp.get_sign());

    return *this + negOp;
  }

  PositFP fma(const PositFP &op2, const PositFP &op3) {
    if (isZero() || op2.isZero()) {
      return op3;
    } else {
      PositFP product = *this * op2;

      if (op3.isZero()) {
        return product;
      } else {
        PositFP sum = product + op3;
        return sum;
      }
    }
  }

  bool operator<(const PositFP &rhs) const {
    if (get_sign() ^ rhs.get_sign()) return get_sign();
    if (get_scale() != rhs.get_scale()) return get_scale() < rhs.get_scale();
    return get_fraction() < rhs.get_fraction();
  }

  ac_int<1, false> get_sign() const { return bits.slc<1>(0); }
  ac_int<8, true> get_scale() const { return bits.slc<8>(1); }
  ac_int<16, false> get_fraction() const { return bits.slc<16>(9); }

  void set_sign(ac_int<1, false> sign) { bits.set_slc(0, sign); }
  void set_scale(ac_int<8, true> scale) { bits.set_slc(1, scale); }
  void set_fraction(ac_int<16, false> fraction) { bits.set_slc(9, fraction); }

  ac_int<1 + 8 + 16, false> bits;

  static const unsigned int width = 1 + 8 + 16;

  inline friend void sc_trace(sc_trace_file *tf, const PositFP &positFP,
                              const std::string &name) {
    // TODO
  }

  inline friend std::ostream &operator<<(ostream &os, const PositFP &positFP) {
    Posit p = positFP;
    os << p;
    return os;
  }

 private:
  static const int nbits = 8;
  static const int es = 1;

  inline int fsb(ac_int<8, false> v) {
    int c;
    c = 1;
    if (!(v & 0xf0)) {
      v <<= 4;
      c += 4;
    }
    if (!(v & 0xc0)) {
      v <<= 2;
      c += 2;
    }
    c -= v & 0x80 ? 1 : 0;
    return 7 - c;
  }

  inline int fsb16(ac_int<16, false> v) {
    int c;
    c = 1;
    if (!(v & 0xff00)) {
      v <<= 8;
      c += 8;
    }
    if (!(v & 0xf000)) {
      v <<= 4;
      c += 4;
    }
    if (!(v & 0xc000)) {
      v <<= 2;
      c += 2;
    }
    c -= v & 0x8000 ? 1 : 0;
    return 15 - c;
  }
};

// Encoder
inline Posit::Posit(const PositFP &input) {
  if (input.isZero()) {
    bits = 0;
  } else {
    // TODO: handle infinity
    if ((input.get_scale() >> es) > 6 or (input.get_scale() >> es) < -6) {
      bits = input.get_scale() > 0 ? 0xEF : 0x1;
    } else {
      int pt_len = nbits + 3 + es;
      ac_int<12, false> pt_bits, regime, exponent, fraction, sticky_bit;

      bool s = input.get_sign();
      int e = input.get_scale();
      bool r = (e >= 0);

      int run = r ? (1 + (e >> es)) : -(e >> es);
      regime = r ? (1 << (run + 1)) - 1 : 0;
      regime[0] = !r;

      exponent = (e % 2 + 2) % 2;
      int nf = max(0, nbits + 1 - (2 + run + es));
      fraction = (input.get_fraction() << 1) >> (16 - nf);

      pt_bits = regime << es + nf + 1;
      pt_bits |= exponent << nf + 1;
      pt_bits |= fraction << 1;
      pt_bits |=
          input.get_fraction() << nf ? 0x1 : 0x0;  // any after fbits - 1 - nf

      int len = 1 + max(nbits + 1, 2 + run + es);
      bool blast = pt_bits[len - nbits];
      bool bafter = pt_bits[len - nbits - 1];
      bool bsticky = pt_bits << (pt_len - (len - nbits - 1));
      bool rb = (blast & bafter) | (bafter & bsticky);

      bits = pt_bits >> (len - nbits);
      if (rb) bits++;
    }
    if (input.get_sign()) bits = bits.bit_complement() + 1;
  }
}

// decoder
inline PositFP::PositFP(const Posit &input) {
  ac_int<8, false> posit_bits = input.bits;
  if (posit_bits == 0) {
    set_fraction(0);
    set_scale(-127);
  } else {
    set_sign(posit_bits.slc<1>(nbits - 1));
    if (get_sign())
      posit_bits =
          (posit_bits - 1).bit_complement();  // convert to positive posit

    posit_bits <<= 1;  // remove sign bit

    ac_int<1, false> leadingBit = posit_bits.slc<1>(nbits - 1);

    int index = leadingBit ? fsb(posit_bits.bit_complement()) : fsb(posit_bits);

    set_scale((leadingBit == 1) ? 6 - index : index - 7);
    set_scale(get_scale() * 2);

    if (index > 0)
      set_scale(get_scale() + posit_bits.slc<1>(index - 1));  // exponent bit

    set_fraction(posit_bits << (nbits - index + es));
    set_fraction((get_fraction() << 7) | 0x8000);
  }
}

inline Posit &Posit::operator+=(const Posit &rhs) {
  PositFP op1 = *this;
  PositFP op2 = rhs;

  *this = op1 + op2;
  return *this;
}

inline Posit &Posit::operator-=(const Posit &rhs) {
  PositFP op1 = *this;
  PositFP op2 = rhs;

  *this = op1 - op2;
  return *this;
}

inline Posit &Posit::operator*=(const Posit &rhs) {
  *this = *this * rhs;
  return *this;
}

inline Posit Posit::operator*(const Posit &rhs) {
  PositFP op1 = *this;
  PositFP op2 = rhs;

  return op1 * op2;
}

inline bool Posit::operator<(const Posit &rhs) const {
  PositFP op1 = *this;
  PositFP op2 = rhs;

  return op1 < op2;
}

inline PositFP &PositFP::operator+=(const PositFP &rhs) {
  *this = *this + rhs;

  return *this;
}

inline PositFP &PositFP::operator+=(const Posit &rhs) {
  PositFP op = rhs;
  *this = *this + op;

  return *this;
}

inline Posit::operator float() const {
#ifndef __SYNTHESIS__
  union ufloat {
    float f;
    uint32_t u;
  };
  union ufloat uf;

  PositFP p = *this;
  uf.u = p.get_sign() ? 1 << 31 : 0;
  uf.u += (p.get_scale() + 127) << 23;
  uf.u += (p.get_fraction() & ((1 << 15) - 1)) << 8;
  return uf.f;
#else
  return 0.0;
#endif
}

inline Posit::Posit(const float &f) {
#ifndef __SYNTHESIS__
  union ufloat {
    float f;
    uint32_t u;
  };
  union ufloat uf;

  uf.f = f;
  PositFP p;
  p.set_sign(uf.f < 0);
  p.set_scale(((uf.u >> 23) & 0xFF) - 127);
  p.set_fraction(((uf.u & ((1 << 23) - 1)) >> 8) | (1 << 15));

  *this = p;
#else
  set_zero();
#endif
}

inline bool operator==(const PositFP &lhs, const PositFP &rhs) {
  return (lhs.get_sign() == rhs.get_sign()) &&
         (lhs.get_scale() == rhs.get_scale()) &&
         (lhs.get_fraction() == rhs.get_fraction());
}

inline bool operator==(const Posit &lhs, const Posit &rhs) {
  return lhs.bits == rhs.bits;
}
