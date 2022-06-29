// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>  // For SSE/AVX support

#include <bit>
#include <iomanip>

#include "score.hpp"
#include "utils.hpp"

using namespace std;

template <uint8_t PIN_COUNT, uint8_t COLOR_COUNT>
std::ostream &operator<<(std::ostream &stream, const Codeword<PIN_COUNT, COLOR_COUNT> &codeword) {
  return codeword.dump(stream);
}

// This is a relatively simple O(2p) scoring method. First we count black hits and unused colors in O(p) time,
// then we consume colors in O(p) time and count white hits. This is quite efficient for a rather simple scoring
// method, with the only real complexity being the packing of pins and colors to reduce space used.
// https://godbolt.org/z/sEEjcY
//
// Elapsed time 4.4948s, average search 3.4682ms
template <uint8_t PIN_COUNT, uint8_t COLOR_COUNT>
Score Codeword<PIN_COUNT, COLOR_COUNT>::scoreSimpleLoops(const Codeword &guess) const {
  uint8_t b = 0;
  uint8_t w = 0;
  uint64_t unusedColorCounts = 0;  // Room for 16 4-bit counters

  uint32_t s = this->codeword;
  uint32_t g = guess.codeword;
  for (int i = 0; i < PIN_COUNT; i++) {
    if ((g & 0xFu) == (s & 0xFu)) {
      b++;
    } else {
      unusedColorCounts += 1lu << ((s & 0xFu) * 4);
    }
    s >>= 4u;
    g >>= 4u;
  }

  s = this->codeword;
  g = guess.codeword;
  for (int i = 0; i < PIN_COUNT; i++) {
    if ((g & 0xFu) != (s & 0xFu) && (unusedColorCounts & (0xFlu << ((g & 0xFu) * 4))) > 0) {
      w++;
      unusedColorCounts -= 1lu << ((g & 0xFu) * 4);
    }
    s >>= 4u;
    g >>= 4u;
  }
  return Score(b, w);
}

// This uses the full counting method from Knuth, plus some fun bit twiddling hacks and SWAR action. This is O(c),
// with constant time to get black hits, and often quite a bit less than O(c) time to get the total hits (and thus the
// white hits.)
//
// Find black hits with xor, which leaves zero nibbles on matches, then count the zeros in the result. This is a
// variation on determining if a word has a zero byte from https://graphics.stanford.edu/~seander/bithacks.html. This
// part ends with using std::popcount() to count the zero nibbles, and when compiled with C++ 20 and -march=native we
// get a single popcountl instruction generated. Codegen example: https://godbolt.org/z/MofY33
//
// Next, Codewords now carry their color counts with them, and we can run over them and add up total hits per Knuth by
// aggregating min color counts between the secret and guess.
//
// Overall this method is much faster than the previous version by ~40% on the 4p6c game with no score cache. It's a
// big improvement for larger games and surprisingly efficient overall.
//
// Elapsed time 3.1218s, average search 2.4088ms
#if 0
template <uint8_t PIN_COUNT, uint8_t COLOR_COUNT>
Score Codeword<PIN_COUNT, COLOR_COUNT>::scoreCountingScalar(const Codeword &guess) const {
  constexpr static uint32_t unusedPinsMask = (uint32_t)(0xFFFFFFFFlu << (PIN_COUNT * 4u));
  uint32_t v = this->codeword ^ guess.codeword;  // Matched pins are now 0.
  v |= unusedPinsMask;                           // Ensure that any unused pin positions are non-zero.
  uint32_t r = ~((((v & 0x77777777u) + 0x77777777u) | v) | 0x77777777u);  // Yields 1 bit per matched pin
  uint8_t b = _mm_popcnt_u32(r);

  int allHits = 0;
  uint64_t scc = this->colorCounts4;
  uint64_t gcc = guess.colorCounts4;
  do {                                              // colorCounts are never 0, so a do-while is solid win
    allHits += std::min(scc & 0xFlu, gcc & 0xFlu);  // cmp/cmovb, no branching
    scc >>= 4u;
    gcc >>= 4u;
  } while (scc != 0 && gcc != 0);  // Early out for many combinations

  return Score(b, allHits - b);
}
#endif

// This uses the full counting method from Knuth, but is organized to allow auto-vectorization of the second part.
// When properly vectorized by the compiler, this method is O(1) time and space.
//
// See scoreCountingScalar() for an explanation of how hits are computed.
//
// Clang's auto-vectorizer will pick up on the modified loop and use vpminub to compute all minimums in a single
// vector op, then use a fixed sequence of shuffles and adds to sum the minimums. Overall perf is very sensitive to
// alignment of the colorCounts. Unaligned fields will make this slower than the scalar version. The auto-vectorizer
// is also pretty sensitive to how the code is structured, and the code it generates for adding up the minimums is
// pretty large and sub-optimal. https://godbolt.org/z/arcE5e
//
// Elapsed time 2.2948s, average search 1.7707ms
template <uint8_t PIN_COUNT, uint8_t COLOR_COUNT>
Score Codeword<PIN_COUNT, COLOR_COUNT>::scoreCountingAutoVec(const Codeword &guess) const {
  constexpr static uint32_t unusedPinsMask = (uint32_t)(0xFFFFFFFFlu << (PIN_COUNT * 4u));
  uint32_t v = this->codeword ^ guess.codeword;  // Matched pins are now 0.
  v |= unusedPinsMask;                           // Ensure that any unused pin positions are non-zero.
  uint32_t r = ~((((v & 0x77777777u) + 0x77777777u) | v) | 0x77777777u);  // Yields 1 bit per matched pin
  uint8_t b = _mm_popcnt_u32(r);

  int allHits = 0;
  auto *scc = (uint8_t *)&(this->colorCounts8);
  auto *gcc = (uint8_t *)&(guess.colorCounts8);
  for (int i = 0; i < 16; i++) {
    allHits += std::min(scc[i], gcc[i]);
  }

  return Score(b, allHits - b);
}

// This uses the full counting method from Knuth, but computing the sum of all hits is vectorized by-hand. This is
// O(1) for both parts, guaranteed no matter what the compiler decides to do. https://godbolt.org/z/KvPf1Y
//
// See scoreCountingScalar() for an explanation of how hits are computed.
//
// Elapsed time 0.9237s, average search 0.7127ms
template <uint8_t PIN_COUNT, uint8_t COLOR_COUNT>
Score Codeword<PIN_COUNT, COLOR_COUNT>::scoreCountingHandVec(const Codeword &guess) const {
  constexpr static uint32_t unusedPinsMask = (uint32_t)(0xFFFFFFFFlu << (PIN_COUNT * 4u));
  uint32_t v = this->codeword ^ guess.codeword;  // Matched pins are now 0.
  v |= unusedPinsMask;                           // Ensure that any unused pin positions are non-zero.
  uint32_t r = ~((((v & 0x77777777u) + 0x77777777u) | v) | 0x77777777u);  // Yields 1 bit per matched pin
  uint8_t b = _mm_popcnt_u32(r);

  // Load the 128-bit color counts into vector registers. Each one is 16 8-bit counters.
  __m128i_u secretColorsVec = _mm_loadu_si128((__m128i_u *)&this->colorCounts8);
  __m128i_u guessColorsVec = _mm_loadu_si128((__m128i_u *)&guess.colorCounts8);

  // Find the minimum of each pair of 8-bit counters in one instruction.
  __m128i_u minColorsVec = _mm_min_epu8(secretColorsVec, guessColorsVec);

  // Add up all of the 8-bit counters into two 16-bit sums in one instruction.
  __m128i vsum = _mm_sad_epu8(minColorsVec, _mm_setzero_si128());

  // Pull out the two 16-bit sums and add them together normally to get our final answer. 3 instructions.
  int allHits = _mm_extract_epi16(vsum, 0) + _mm_extract_epi16(vsum, 4);

  return Score(b, allHits - b);
}

// Wrapper for the real scoring function to allow for easy experimentation.
// Note: I used to employ a score cache here, see docs/Score_Cache.md for details.
template <uint8_t PIN_COUNT, uint8_t COLOR_COUNT>
Score Codeword<PIN_COUNT, COLOR_COUNT>::score(const Codeword &guess) const {
  return scoreCountingHandVec(guess);
}

// Make a list of all codewords for a given number of "colors". Colors are represented by the digits 1 thru n. This
// figures out how many codewords there are, which is COLOR_COUNT ^ PIN_COUNT, then converts the base-10 number of each
// codeword to it's base-COLOR_COUNT representation.
template <uint8_t PIN_COUNT, uint8_t COLOR_COUNT>
vector<Codeword<PIN_COUNT, COLOR_COUNT>> &Codeword<PIN_COUNT, COLOR_COUNT>::getAllCodewords() {
  if (allCodewords.empty()) {
    allCodewords.reserve(TOTAL_CODEWORDS);

    for (uint i = 0; i < TOTAL_CODEWORDS; i++) {
      uint w = i;
      uint32_t cw = 0;
      int di = 0;
      do {
        cw |= (w % COLOR_COUNT) << (4u * di++);
        w /= COLOR_COUNT;
      } while (w > 0);

      cw += ONE_PINS;  // Colors start at 1, not 0.
      allCodewords.emplace_back(cw);
    }
  }
  return allCodewords;
}

template <uint8_t PIN_COUNT, uint8_t COLOR_COUNT>
std::ostream &Codeword<PIN_COUNT, COLOR_COUNT>::dump(std::ostream &stream) const {
  std::ios state(nullptr);
  state.copyfmt(stream);
  stream << std::hex << std::setw(PIN_COUNT) << std::setfill('0') << codeword;
  // stream << " o:" << ordinal << " cc4:" << std::setw(16) << colorCounts4 << " cc8:" << std::setw(32) << (unsigned
  // long long)colorCounts8;
  stream.copyfmt(state);
  return stream;
}
