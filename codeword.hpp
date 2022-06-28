// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <iostream>
#include <vector>

#include "score.hpp"
#include "utils.hpp"

#if defined(__CUDA_ARCH__)
#define DEVICE_ANNOTATION __device__ __host__
#else
#define DEVICE_ANNOTATION
#endif

// Class to hold a codeword for the Mastermind game.
//
// This is represented as a packed group of 4-bit digits, up to 8 digits. Colors are pre-computed and packed into 64 or
// 128 bits for different versions of the scoring functions.
template <uint8_t PIN_COUNT, uint8_t COLOR_COUNT>
class Codeword {
 public:
  constexpr Codeword() noexcept : codeword(0xFFFFFFFF), colorCounts8(0) {}

  constexpr Codeword(uint32_t codeword) noexcept : codeword(codeword), colorCounts8(computeColorCounts8(codeword)) {}

  bool isInvalid() const { return codeword == -0xFFFFFFFF; }

  bool operator==(const Codeword other) const { return codeword == other.codeword; }

  DEVICE_ANNOTATION uint32_t packedCodeword() const { return codeword; }
  DEVICE_ANNOTATION unsigned __int128 packedColors8() const { return colorCounts8; }

  constexpr static uint64_t TOTAL_CODEWORDS = constPow<uint64_t>(COLOR_COUNT, PIN_COUNT);
  constexpr static Score WINNING_SCORE = Score(PIN_COUNT, 0);  // "0x40" for a 4-pin game.
  constexpr static uint32_t ONE_PINS = 0x11111111u & ((1lu << PIN_COUNT * 4u) - 1);

  Score score(const Codeword &guess) const;

  static std::vector<Codeword> &getAllCodewords();

  std::ostream &dump(std::ostream &stream) const;

 private:
  unsigned __int128 colorCounts8;  // Room for 16 8-bit counters
  uint32_t codeword;

  // All codewords for the given pin and color counts.
  static inline std::vector<Codeword> allCodewords;

  Score scoreSimpleLoops(const Codeword &guess) const;
  Score scoreCountingScalar(const Codeword &guess) const;
  Score scoreCountingAutoVec(const Codeword &guess) const;
  Score scoreCountingHandVec(const Codeword &guess) const;

  // Pre-compute color counts for all Codewords. Building this two ways right now for experimentation. The packed 4-bit
  // counters are good for the scalar versions and overall memory usage, while the 8-bit counters are needed for SSE/AVX
  // vectorization, both auto and by-hand. https://godbolt.org/z/bfM86K
  constexpr static unsigned __int128 computeColorCounts8(uint32_t word) {
    unsigned __int128 cc8 = 0;
    for (int i = 0; i < PIN_COUNT; i++) {
      cc8 += ((unsigned __int128)1) << ((word & 0xFu) * 8);
      word >>= 4u;
    }
    return cc8;
  }

  constexpr static uint64_t computeColorCounts4(uint32_t word) {
    uint64_t cc4 = 0;
    for (int i = 0; i < PIN_COUNT; i++) {
      cc4 += 1lu << ((word & 0xFu) * 4);
      word >>= 4u;
    }
    return cc4;
  }

  // Legacy, but here for reference.
  constexpr static uint32_t computeOrdinal(uint32_t word) {
    uint32_t o = 0;
    uint32_t mult = 1;
    for (int i = 0; i < PIN_COUNT; i++) {
      o += ((word & 0xFu) - 1) * mult;
      word >>= 4u;
      mult *= COLOR_COUNT;
    }
    return o;
  }
};

template <uint8_t PIN_COUNT, uint8_t COLOR_COUNT>
std::ostream &operator<<(std::ostream &stream, const Codeword<PIN_COUNT, COLOR_COUNT> &codeword);

#include "codeword.cpp"
