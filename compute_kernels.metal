// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <metal_stdlib>

using namespace metal;

// The core of Knuth's Mastermind algorithm, and others, as a Metal compute kernels.
//
// Scores here are not the classic combindation of black hits and white hits. A score's ordinal is (b(p + 1) - ((b -
// 1)b) / 2) + w. See README.md for details.  By using the score's ordinal we can have densly packed set of counters to
// form the subset counts as we go. These scores never escape the GPU, so it doesn't matter that they don't match any
// other forms of scores in the rest of the program.

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
// Next, color counts come from the parallel buffer, and we can run over them and add up total hits per Knuth by
// aggregating min color counts between the secret and guess. While I made a SIMD vesion of this for the CPU version,
// I haven't tried to see if that's possible here.
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

  int allHits = 0;
  device const uint8_t *scc = (device const uint8_t *)&secretColors;
  constant uint8_t *gcc = (constant uint8_t *)&guessColors;
  for (int i = 0; i < 16; i++) {
    allHits += min(scc[i], gcc[i]);
  }

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
template <uint32_t pinCount>
kernel void findKnuthGuessKernel(device const uint32_t *allCodewords, device const uint4 *allCodewordsColors,
                                 constant uint32_t &possibleSolutionsCount, constant uint32_t *possibleSolutions,
                                 constant uint4 *possibleSolutionsColors, device uint32_t *scores,
                                 device bool *remainingIsPossibleSolution,
                                 threadgroup uint32_t *tgScoreCounts [[threadgroup(0)]],
                                 const uint tidGrid [[thread_position_in_grid]],
                                 const uint tidGroup [[thread_position_in_threadgroup]]) {
  // Total scores = (p * (p + 3)) / 2, but +1 for imperfect packing.
  constexpr int totalScores = ((pinCount * (pinCount + 3)) / 2) + 1;
  threadgroup uint32_t *scoreCounts = &tgScoreCounts[tidGroup * totalScores];

  bool isPossibleSolution =
      computeSubsetSizes<pinCount>(totalScores, scoreCounts, allCodewords[tidGrid], allCodewordsColors[tidGrid],
                                   possibleSolutionsCount, possibleSolutions, possibleSolutionsColors);

  uint32_t largestSubsetSize = 0;
  for (int i = 0; i < totalScores; i++) {
    largestSubsetSize = max(largestSubsetSize, scoreCounts[i]);
  }
  scores[tidGrid] = possibleSolutionsCount - largestSubsetSize;
  remainingIsPossibleSolution[tidGrid] = isPossibleSolution;
}

// Compute kernel for the Most Parts strategy
template <uint32_t pinCount>
kernel void findMostPartsGuessKernel(device const uint32_t *allCodewords, device const uint4 *allCodewordsColors,
                                     constant uint32_t &possibleSolutionsCount, constant uint32_t *possibleSolutions,
                                     constant uint4 *possibleSolutionsColors, device uint32_t *scores,
                                     device bool *remainingIsPossibleSolution,
                                     threadgroup uint32_t *tgScoreCounts [[threadgroup(0)]],
                                     const uint tidGrid [[thread_position_in_grid]],
                                     const uint tidGroup [[thread_position_in_threadgroup]]) {
  // Total scores = (p * (p + 3)) / 2, but +1 for imperfect packing.
  constexpr int totalScores = ((pinCount * (pinCount + 3)) / 2) + 1;
  threadgroup uint32_t *scoreCounts = &tgScoreCounts[tidGroup * totalScores];

  bool isPossibleSolution =
      computeSubsetSizes<pinCount>(totalScores, scoreCounts, allCodewords[tidGrid], allCodewordsColors[tidGrid],
                                   possibleSolutionsCount, possibleSolutions, possibleSolutionsColors);

  uint32_t totalUsedSubsets = 0;
  for (int i = 0; i < totalScores; i++) {
    if (scoreCounts[i] > 0) {
      totalUsedSubsets++;
    }
  }
  scores[tidGrid] = totalUsedSubsets;
  remainingIsPossibleSolution[tidGrid] = isPossibleSolution;
}

// Compute kernel for the Entropy strategy
template <uint32_t pinCount>
kernel void findEntropyGuessKernel(device const uint32_t *allCodewords, device const uint4 *allCodewordsColors,
                                   constant uint32_t &possibleSolutionsCount, constant uint32_t *possibleSolutions,
                                   constant uint4 *possibleSolutionsColors, device uint32_t *scores,
                                   device bool *remainingIsPossibleSolution,
                                   threadgroup uint32_t *tgScoreCounts [[threadgroup(0)]],
                                   const uint tidGrid [[thread_position_in_grid]],
                                   const uint tidGroup [[thread_position_in_threadgroup]]) {
  // Total scores = (p * (p + 3)) / 2, but +1 for imperfect packing.
  constexpr int totalScores = ((pinCount * (pinCount + 3)) / 2) + 1;
  threadgroup uint32_t *scoreCounts = &tgScoreCounts[tidGroup * totalScores];

  bool isPossibleSolution =
      computeSubsetSizes<pinCount>(totalScores, scoreCounts, allCodewords[tidGrid], allCodewordsColors[tidGrid],
                                   possibleSolutionsCount, possibleSolutions, possibleSolutionsColors);

  float entropySum = 0.0;
  for (int i = 0; i < totalScores; i++) {
    if (scoreCounts[i] > 0) {
      float pi = (float)scoreCounts[i] / possibleSolutionsCount;
      entropySum -= pi * log(pi);
    }
  }
  scores[tidGrid] = round(entropySum * 1'000'000'000.0);  // 9 digits of precision
  remainingIsPossibleSolution[tidGrid] = isPossibleSolution;
}

// Compute kernel for the Expected Size strategy
template <uint32_t pinCount>
kernel void findExpectedSizeGuessKernel(device const uint32_t *allCodewords, device const uint4 *allCodewordsColors,
                                        constant uint32_t &possibleSolutionsCount, constant uint32_t *possibleSolutions,
                                        constant uint4 *possibleSolutionsColors, device uint32_t *scores,
                                        device bool *remainingIsPossibleSolution,
                                        threadgroup uint32_t *tgScoreCounts [[threadgroup(0)]],
                                        const uint tidGrid [[thread_position_in_grid]],
                                        const uint tidGroup [[thread_position_in_threadgroup]]) {
  // Total scores = (p * (p + 3)) / 2, but +1 for imperfect packing.
  constexpr int totalScores = ((pinCount * (pinCount + 3)) / 2) + 1;
  threadgroup uint32_t *scoreCounts = &tgScoreCounts[tidGroup * totalScores];

  bool isPossibleSolution =
      computeSubsetSizes<pinCount>(totalScores, scoreCounts, allCodewords[tidGrid], allCodewordsColors[tidGrid],
                                   possibleSolutionsCount, possibleSolutions, possibleSolutionsColors);

  float expectedSize = 0.0;
  for (int i = 0; i < totalScores; i++) {
    if (scoreCounts[i] > 0) {
      expectedSize += ((float)scoreCounts[i] * (float)scoreCounts[i]) / possibleSolutionsCount;
    }
  }

  scores[tidGrid] = (uint32_t)round(expectedSize * 1'000'000.0) * -1;  // 9 digits of precision
  remainingIsPossibleSolution[tidGrid] = isPossibleSolution;
}

// Explicit specializations Knuth
template [[host_name("findKnuthGuessKernel_2")]] kernel void findKnuthGuessKernel<2>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

template [[host_name("findKnuthGuessKernel_3")]] kernel void findKnuthGuessKernel<3>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

template [[host_name("findKnuthGuessKernel_4")]] kernel void findKnuthGuessKernel<4>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

template [[host_name("findKnuthGuessKernel_5")]] kernel void findKnuthGuessKernel<5>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

template [[host_name("findKnuthGuessKernel_6")]] kernel void findKnuthGuessKernel<6>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

template [[host_name("findKnuthGuessKernel_7")]] kernel void findKnuthGuessKernel<7>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

template [[host_name("findKnuthGuessKernel_8")]] kernel void findKnuthGuessKernel<8>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

// Explicit specializations Most Parts
template [[host_name("findMostPartsGuessKernel_2")]] kernel void findMostPartsGuessKernel<2>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

template [[host_name("findMostPartsGuessKernel_3")]] kernel void findMostPartsGuessKernel<3>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

template [[host_name("findMostPartsGuessKernel_4")]] kernel void findMostPartsGuessKernel<4>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

template [[host_name("findMostPartsGuessKernel_5")]] kernel void findMostPartsGuessKernel<5>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

template [[host_name("findMostPartsGuessKernel_6")]] kernel void findMostPartsGuessKernel<6>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

template [[host_name("findMostPartsGuessKernel_7")]] kernel void findMostPartsGuessKernel<7>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

template [[host_name("findMostPartsGuessKernel_8")]] kernel void findMostPartsGuessKernel<8>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

// Explicit specializations Entropy
template [[host_name("findEntropyGuessKernel_2")]] kernel void findEntropyGuessKernel<2>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

template [[host_name("findEntropyGuessKernel_3")]] kernel void findEntropyGuessKernel<3>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

template [[host_name("findEntropyGuessKernel_4")]] kernel void findEntropyGuessKernel<4>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

template [[host_name("findEntropyGuessKernel_5")]] kernel void findEntropyGuessKernel<5>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

template [[host_name("findEntropyGuessKernel_6")]] kernel void findEntropyGuessKernel<6>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

template [[host_name("findEntropyGuessKernel_7")]] kernel void findEntropyGuessKernel<7>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

template [[host_name("findEntropyGuessKernel_8")]] kernel void findEntropyGuessKernel<8>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

// Explicit specializations Expected Size
template [[host_name("findExpectedSizeGuessKernel_2")]] kernel void findExpectedSizeGuessKernel<2>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

template [[host_name("findExpectedSizeGuessKernel_3")]] kernel void findExpectedSizeGuessKernel<3>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

template [[host_name("findExpectedSizeGuessKernel_4")]] kernel void findExpectedSizeGuessKernel<4>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

template [[host_name("findExpectedSizeGuessKernel_5")]] kernel void findExpectedSizeGuessKernel<5>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

template [[host_name("findExpectedSizeGuessKernel_6")]] kernel void findExpectedSizeGuessKernel<6>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

template [[host_name("findExpectedSizeGuessKernel_7")]] kernel void findExpectedSizeGuessKernel<7>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);

template [[host_name("findExpectedSizeGuessKernel_8")]] kernel void findExpectedSizeGuessKernel<8>(
    device const uint32_t *, device const uint4 *, constant uint32_t &, constant uint32_t *, constant uint4 *,
    device uint32_t *, device bool *, threadgroup uint32_t *, const uint, const uint);
