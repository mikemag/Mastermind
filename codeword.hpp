// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <iostream>
#include <vector>

#include "score.hpp"
#include "utils.hpp"

// Class to hold a codeword for the Mastermind game.
//
// This is represented as a packed group of 4-bit digits, up to 8 digits. Colors are pre-computed and packed into 64 or
// 128 bits for different versions of the scoring functions.
template <uint8_t PIN_COUNT, uint8_t COLOR_COUNT>
class Codeword {
#if defined(__CUDACC__)
  using CT = typename std::conditional<(COLOR_COUNT <= sizeof(uint64_t)), uint64_t, unsigned __int128>::type;
#else
  // TODO: improve the CPU version to handle smaller color counts
  using CT = unsigned __int128;
#endif

 public:
  CUDA_HOST_AND_DEVICE constexpr Codeword() noexcept : codeword(0xFFFFFFFF), colorCounts(0) {}

  constexpr Codeword(uint32_t codeword) noexcept : codeword(codeword), colorCounts(computeColorCounts(codeword)) {}

  CUDA_HOST_AND_DEVICE bool isInvalid() const { return codeword == 0xFFFFFFFF; }

  CUDA_HOST_AND_DEVICE bool operator==(const Codeword other) const { return codeword == other.codeword; }

  CUDA_HOST_AND_DEVICE uint32_t packedCodeword() const { return codeword; }
  CT packedColors() const { return colorCounts; }
#if defined(__CUDACC__)
  CUDA_HOST_AND_DEVICE uint4 packedColorsCUDA() const {
    if constexpr (isSize2()) {
      return {0, 0, ccOverlay.z, ccOverlay.w};
    } else {
      static_assert(isSize4());
      return {ccOverlay.x, ccOverlay.y, ccOverlay.z, ccOverlay.w};
    }
  }
#endif

  constexpr static uint64_t TOTAL_CODEWORDS = constPow<uint64_t>(COLOR_COUNT, PIN_COUNT);
  constexpr static Score WINNING_SCORE = Score(PIN_COUNT, 0);  // "0x40" for a 4-pin game.
  constexpr static uint32_t ONE_PINS = 0x11111111u & ((1lu << PIN_COUNT * 4u) - 1);

  Score score(const Codeword &guess) const;

  static std::vector<Codeword> &getAllCodewords();

  std::ostream &dump(std::ostream &stream) const;

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

  CUDA_HOST_AND_DEVICE constexpr static bool isSize2() { return sizeof(CT) == 8; }
  CUDA_HOST_AND_DEVICE constexpr static bool isSize4() { return sizeof(CT) == 16; }

 private:
  struct UInt128Overlay {
    unsigned int x, y, z, w;
  };

  struct UInt64Overlay {
    unsigned int z, w;
  };

  using OT = typename std::conditional<std::is_same_v<CT, uint64_t>, UInt64Overlay, UInt128Overlay>::type;

  union {
    CT colorCounts;  // Room for 8 or 16 8-bit counters, depending on COLOR_COUNT
    OT ccOverlay;
  };
  uint32_t codeword;

  // All codewords for the given pin and color counts.
  static inline std::vector<Codeword> allCodewords;

  Score scoreSimpleLoops(const Codeword &guess) const;
  Score scoreCountingScalar(const Codeword &guess) const;
  Score scoreCountingAutoVec(const Codeword &guess) const;
  Score scoreCountingHandVec(const Codeword &guess) const;

  // Pre-compute color counts for all Codewords. Building this two ways right now for experimentation. The packed
  // 4-bit counters are good for the scalar versions and overall memory usage, while the 8-bit counters are needed for
  // SSE/AVX vectorization, both auto and by-hand. https://godbolt.org/z/bfM86K
  constexpr static CT computeColorCounts(uint32_t word) {
    CT cc8 = 0;
    for (int i = 0; i < PIN_COUNT; i++) {
      cc8 += ((CT)1) << (((word & 0xFu) - 1) * 8);
      word >>= 4u;
    }
    return cc8;
  }
};

#if defined(__CUDACC__)
static_assert(sizeof(Codeword<4, 8>) == 16);
static_assert(__alignof__(Codeword<4, 8>) == 8);
#else
static_assert(sizeof(Codeword<4, 8>) == 32);
static_assert(__alignof__(Codeword<4, 8>) == 16);
#endif
static_assert(sizeof(Codeword<4, 9>) == 32);
static_assert(__alignof__(Codeword<4, 9>) == 16);

template <uint8_t PIN_COUNT, uint8_t COLOR_COUNT>
std::ostream &operator<<(std::ostream &stream, const Codeword<PIN_COUNT, COLOR_COUNT> &codeword);

#include "codeword.inl"
