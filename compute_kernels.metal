// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <metal_stdlib>

#include "compute_kernel_constants.h"

using namespace metal;

// The core of Knuth's Mastermind algorithm, and others, as a Metal compute kernels.
//
// Scores here are not the classic combination of black hits and white hits. A score's ordinal is (b(p + 1) - ((b -
// 1)b) / 2) + w. See docs/Score_Ordinals.md for details. By using the score's ordinal we can have densely packed set
// of counters to form the subset counts as we go. These scores never escape the GPU, so it doesn't matter that they
// don't match any other forms of scores in the rest of the program.

// Mastermind scoring function
//
// This mirrors the scalar version very closely. It's the full counting method from Knuth, plus some fun bit twiddling
// hacks and SWAR action. This is O(colors), with constant time to get black hits, and often quite a bit less than
// O(colors) time to get the total hits (and thus the white hits.)
//
// Find black hits with xor, which leaves zero nibbles on matches, then count the zeros in the result. This is a
// variation on determining if a word has a zero byte from https://graphics.stanford.edu/~seander/bithacks.html. This
// part ends with using the GPU's SIMD popcount() to count the zero nibbles.
//
// Next, color counts come from the parallel buffer, and we can run over them and add up total hits, per Knuth[1], by
// aggregating min color counts between the secret and guess.
//
// Templated w/ pinCount and explicitly specalized below. We lookup the correct version to use by name during startup.
template <uint32_t pinCount>
uint8_t scoreCodewords(device const uint32_t &secret, device const uint4 &secretColors, constant uint32_t &guess,
                       constant uint4 &guessColors) {
  constexpr uint32_t unusedPinsMask = 0xFFFFFFFFu & ~((1lu << pinCount * 4u) - 1);
  uint32_t v = secret ^ guess;  // Matched pins are now 0.
  v |= unusedPinsMask;          // Ensure that any unused pin positions are non-zero.
  uint32_t r = ~((((v & 0x77777777u) + 0x77777777u) | v) | 0x77777777u);  // Yields 1 bit per matched pin
  uint8_t b = popcount(r);

  device const uchar4 *scc = (device const uchar4 *)&secretColors;
  constant uchar4 *gcc = (constant uchar4 *)&guessColors;
  uchar4 mins1 = min(scc[0], gcc[0]);
  uchar4 mins2 = min(scc[1], gcc[1]);
  uchar4 mins3 = min(scc[2], gcc[2]);
  uchar4 mins4 = min(scc[3], gcc[3]);
  uchar4 totals = (mins1 + mins2) + (mins3 + mins4);
  int allHits = (totals.x + totals.y) + (totals.z + totals.w);

  // Given w = ah - b, simplify to i = bp - ((b - 1)b) / 2) + ah. I wonder if the compiler noticed that.
  // https://godbolt.org/z/ab5vTn -- gcc 10.2 notices and simplifies, clang 11.0.0 misses it.
  return b * pinCount - ((b - 1) * b) / 2 + allHits;
}

// The common portion of the kernels which scores all possible solutions against a given guess and computes subset
// sizes and whether or not the guess is a possible solution.
template <uint32_t pinCount>
bool computeSubsetSizes(int totalScores, threadgroup uint32_t *scoreCounts, device const uint32_t &secret,
                        device const uint4 &secretColors, constant uint32_t &possibleSolutionsCount,
                        constant uint32_t *possibleSolutions, constant uint4 *possibleSolutionsColors) {
  bool isPossibleSolution = false;
  for (int i = 0; i < totalScores; i++) scoreCounts[i] = 0;

  for (uint32_t i = 0; i < possibleSolutionsCount; i++) {
    uint8_t score = scoreCodewords<pinCount>(secret, secretColors, possibleSolutions[i], possibleSolutionsColors[i]);
    scoreCounts[score]++;
    if (score == totalScores - 1) {  // The last score is the winning score
      isPossibleSolution = true;
    }
  }
  return isPossibleSolution;
}

// This takes two sets of codewords: the "all codewords" set, which is every possible codeword, and the "possible
// solutions" set. They're separated into two buffers each: one for codewords, which are packed into 32 bits, and
// one for pre-computed color counts, packed into 128 bits as 16 8-bit counters.
//
// The all codewords set is placed into GPU memory once at program start and remains constant.
//
// The possible solutions set changes each time, both content and length, but reuses the same buffers.
//
// Output is two arrays with elements for each of the all codewords set: one with the score, and one bool
// for whether or not the codeword is a possible solution.
//
// Finally, there's threadgroup memory for each thread with enough room for all of the intermediate subset sizes.
// This is important: the first version held this in threadlocal memory and occupancy was terrible.
template <uint32_t pinCount, Algo algo, bool supportFamilyMac2>
kernel void subsettingAlgosKernel(
    device const uint32_t *allCodewords [[buffer(BufferIndexAllCodewords)]],
    device const uint4 *allCodewordsColors [[buffer(BufferIndexAllCodewordsColors)]],
    constant uint32_t &possibleSolutionsCount [[buffer(BufferIndexPossibleSolutionsCount)]],
    constant uint32_t *possibleSolutions [[buffer(BufferIndexPossibleSolutions)]],
    constant uint4 *possibleSolutionsColors [[buffer(BufferIndexPossibleSolutionsColors)]],
    device uint32_t *scores [[buffer(BufferIndexScores)]],
    device bool *remainingIsPossibleSolution [[buffer(BufferIndexRemainingIsPossibleSolution)]],
    device uint32_t *fullyDiscriminatingCodewords [[buffer(BufferIndexFullyDiscriminatingCodewords)]],
    threadgroup uint32_t *tgScoreCounts [[threadgroup(0)]], const uint tidGrid [[thread_position_in_grid]],
    const uint tidGroup [[thread_position_in_threadgroup]], const uint threadsPerSIMDGroup [[threads_per_simdgroup]]) {
  // Total scores = (p * (p + 3)) / 2, but +1 for imperfect packing.
  constexpr int totalScores = ((pinCount * (pinCount + 3)) / 2) + 1;
  threadgroup uint32_t *scoreCounts = &tgScoreCounts[tidGroup * totalScores];

  bool isPossibleSolution =
      computeSubsetSizes<pinCount>(totalScores, scoreCounts, allCodewords[tidGrid], allCodewordsColors[tidGrid],
                                   possibleSolutionsCount, possibleSolutions, possibleSolutionsColors);
  remainingIsPossibleSolution[tidGrid] = isPossibleSolution;

  uint32_t largestSubsetSize = 0;
  uint32_t totalUsedSubsets = 0;
  float entropySum = 0.0;
  float expectedSize = 0.0;
  for (int i = 0; i < totalScores; i++) {
    if (scoreCounts[i] > 0) {
      totalUsedSubsets++;
      switch (algo) {
        case Knuth:
          largestSubsetSize = max(largestSubsetSize, scoreCounts[i]);
          break;
        case MostParts:
          // Already done
          break;
        case ExpectedSize:
          expectedSize += ((float)scoreCounts[i] * (float)scoreCounts[i]) / possibleSolutionsCount;
          break;
        case Entropy:
          float pi = (float)scoreCounts[i] / possibleSolutionsCount;
          entropySum -= pi * log(pi);
          break;
      }
    }
  }
  switch (algo) {
    case Knuth:
      scores[tidGrid] = possibleSolutionsCount - largestSubsetSize;
      break;
    case MostParts:
      scores[tidGrid] = totalUsedSubsets;
      break;
    case ExpectedSize:
      scores[tidGrid] = (uint32_t)round(expectedSize * 1'000'000.0) * -1;  // 9 digits of precision
      break;
    case Entropy:
      scores[tidGrid] = round(entropySum * 1'000'000'000.0);  // 9 digits of precision
      break;
  }

  // If we find some guesses which are fully discriminating, we want to pick the first one lexically to play. tidGrid is
  // the same as the ordinal for each member of allCodewords, so we can simply take the min tidGrid. We'll have every
  // member of each SIMD group that finds such a solution vote on the minimum, and have the first of them write the
  // result. I could do a further reduction to a value per threadgroup, or a final single value, but for now I'll just
  // let the CPU take the first non-zero result and run with it.
  if (supportFamilyMac2) {
    if (totalUsedSubsets == possibleSolutionsCount) {
      uint d = simd_min(tidGrid);
      if (simd_is_first()) {
        fullyDiscriminatingCodewords[tidGrid / threadsPerSIMDGroup] = d;
      }
    }
  }
}

// -----------------------------------------------------------------------------------
// Explicit specializations with MTLGPUFamilyMac2 support

// Explicit specializations Knuth
template [[host_name("findKnuthGuessKernel_2")]] kernel void subsettingAlgosKernel<2, Knuth, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findKnuthGuessKernel_3")]] kernel void subsettingAlgosKernel<3, Knuth, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findKnuthGuessKernel_4")]] kernel void subsettingAlgosKernel<4, Knuth, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findKnuthGuessKernel_5")]] kernel void subsettingAlgosKernel<5, Knuth, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findKnuthGuessKernel_6")]] kernel void subsettingAlgosKernel<6, Knuth, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findKnuthGuessKernel_7")]] kernel void subsettingAlgosKernel<7, Knuth, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findKnuthGuessKernel_8")]] kernel void subsettingAlgosKernel<8, Knuth, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

// Explicit specializations Most Parts
template [[host_name("findMostPartsGuessKernel_2")]] kernel void subsettingAlgosKernel<2, MostParts, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findMostPartsGuessKernel_3")]] kernel void subsettingAlgosKernel<3, MostParts, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findMostPartsGuessKernel_4")]] kernel void subsettingAlgosKernel<4, MostParts, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findMostPartsGuessKernel_5")]] kernel void subsettingAlgosKernel<5, MostParts, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findMostPartsGuessKernel_6")]] kernel void subsettingAlgosKernel<6, MostParts, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findMostPartsGuessKernel_7")]] kernel void subsettingAlgosKernel<7, MostParts, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findMostPartsGuessKernel_8")]] kernel void subsettingAlgosKernel<8, MostParts, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

// Explicit specializations Entropy
template [[host_name("findEntropyGuessKernel_2")]] kernel void subsettingAlgosKernel<2, Entropy, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findEntropyGuessKernel_3")]] kernel void subsettingAlgosKernel<3, Entropy, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findEntropyGuessKernel_4")]] kernel void subsettingAlgosKernel<4, Entropy, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findEntropyGuessKernel_5")]] kernel void subsettingAlgosKernel<5, Entropy, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findEntropyGuessKernel_6")]] kernel void subsettingAlgosKernel<6, Entropy, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findEntropyGuessKernel_7")]] kernel void subsettingAlgosKernel<7, Entropy, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findEntropyGuessKernel_8")]] kernel void subsettingAlgosKernel<8, Entropy, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

// Explicit specializations Expected Size
template [[host_name("findExpectedSizeGuessKernel_2")]] kernel void subsettingAlgosKernel<2, ExpectedSize, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findExpectedSizeGuessKernel_3")]] kernel void subsettingAlgosKernel<3, ExpectedSize, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findExpectedSizeGuessKernel_4")]] kernel void subsettingAlgosKernel<4, ExpectedSize, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findExpectedSizeGuessKernel_5")]] kernel void subsettingAlgosKernel<5, ExpectedSize, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findExpectedSizeGuessKernel_6")]] kernel void subsettingAlgosKernel<6, ExpectedSize, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findExpectedSizeGuessKernel_7")]] kernel void subsettingAlgosKernel<7, ExpectedSize, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findExpectedSizeGuessKernel_8")]] kernel void subsettingAlgosKernel<8, ExpectedSize, true>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

// -----------------------------------------------------------------------------------
// Explicit specializations *without* MTLGPUFamilyMac2 support

// Explicit specializations Knuth
template [[host_name("findKnuthGuessKernel_no2_2")]] kernel void subsettingAlgosKernel<2, Knuth, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findKnuthGuessKernel_no2_3")]] kernel void subsettingAlgosKernel<3, Knuth, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findKnuthGuessKernel_no2_4")]] kernel void subsettingAlgosKernel<4, Knuth, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findKnuthGuessKernel_no2_5")]] kernel void subsettingAlgosKernel<5, Knuth, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findKnuthGuessKernel_no2_6")]] kernel void subsettingAlgosKernel<6, Knuth, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findKnuthGuessKernel_no2_7")]] kernel void subsettingAlgosKernel<7, Knuth, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findKnuthGuessKernel_no2_8")]] kernel void subsettingAlgosKernel<8, Knuth, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

// Explicit specializations Most Parts
template [[host_name("findMostPartsGuessKernel_no2_2")]] kernel void subsettingAlgosKernel<2, MostParts, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findMostPartsGuessKernel_no2_3")]] kernel void subsettingAlgosKernel<3, MostParts, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findMostPartsGuessKernel_no2_4")]] kernel void subsettingAlgosKernel<4, MostParts, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findMostPartsGuessKernel_no2_5")]] kernel void subsettingAlgosKernel<5, MostParts, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findMostPartsGuessKernel_no2_6")]] kernel void subsettingAlgosKernel<6, MostParts, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findMostPartsGuessKernel_no2_7")]] kernel void subsettingAlgosKernel<7, MostParts, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findMostPartsGuessKernel_no2_8")]] kernel void subsettingAlgosKernel<8, MostParts, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

// Explicit specializations Entropy
template [[host_name("findEntropyGuessKernel_no2_2")]] kernel void subsettingAlgosKernel<2, Entropy, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findEntropyGuessKernel_no2_3")]] kernel void subsettingAlgosKernel<3, Entropy, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findEntropyGuessKernel_no2_4")]] kernel void subsettingAlgosKernel<4, Entropy, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findEntropyGuessKernel_no2_5")]] kernel void subsettingAlgosKernel<5, Entropy, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findEntropyGuessKernel_no2_6")]] kernel void subsettingAlgosKernel<6, Entropy, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findEntropyGuessKernel_no2_7")]] kernel void subsettingAlgosKernel<7, Entropy, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findEntropyGuessKernel_no2_8")]] kernel void subsettingAlgosKernel<8, Entropy, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

// Explicit specializations Expected Size
template [[host_name("findExpectedSizeGuessKernel_no2_2")]] kernel void subsettingAlgosKernel<2, ExpectedSize, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findExpectedSizeGuessKernel_no2_3")]] kernel void subsettingAlgosKernel<3, ExpectedSize, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findExpectedSizeGuessKernel_no2_4")]] kernel void subsettingAlgosKernel<4, ExpectedSize, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findExpectedSizeGuessKernel_no2_5")]] kernel void subsettingAlgosKernel<5, ExpectedSize, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findExpectedSizeGuessKernel_no2_6")]] kernel void subsettingAlgosKernel<6, ExpectedSize, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findExpectedSizeGuessKernel_no2_7")]] kernel void subsettingAlgosKernel<7, ExpectedSize, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);

template [[host_name("findExpectedSizeGuessKernel_no2_8")]] kernel void subsettingAlgosKernel<8, ExpectedSize, false>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, device uint32_t *, threadgroup uint32_t *, const uint, const uint, const uint);
