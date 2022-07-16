// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cmath>

#include "preset_initial_guesses.h"

// Multiple algorithms for solving Mastermind
//
// There are many strategies for selecting the next "best" guess when playing Mastermind, and the types here capture
// some common ones. Most of them rely on splitting the remaining possible guesses into groups or subsets based on their
// scores vs each other.
//
// References:
// [1] D.E. Knuth. The computer as Master Mind. Journal of Recreational Mathematics, 9(1):1–6, 1976.
// https://www.cs.uni.edu/~wallingf/teaching/cs3530/resources/knuth-mastermind.pdf
//
// [2] Geoffroy Ville, An Optimal Mastermind (4,7) Strategy and More Results in the Expected Case, March
// 2013, arXiv:1305.1010 [cs.GT].
//
// [3] Barteld Kooi, Yet another mastermind Strategy. International Computer Games Association Journal,
// 28(1):13–20, 2005. https://www.researchgate.net/publication/30485793_Yet_another_Mastermind_strategy

namespace Algos {

struct Knuth {
  using MaxSubsetSizeT = int32_t;
  constexpr static const char* name = "Knuth";

  template <typename SubsetSizeT>
  CUDA_HOST_AND_DEVICE static void accumulateSubsetSize(SubsetSizeT& s) {
    s++;
  }

  using RankingAccumulatorType = uint32_t;

  template <typename SubsetSizeT>
  CUDA_HOST_AND_DEVICE static void accumulateRanking(RankingAccumulatorType& rankingAccumulator, SubsetSizeT& s,
                                                     uint32_t possibleSolutionsCount) {
    if (s > rankingAccumulator) rankingAccumulator = s;
  }

  CUDA_HOST_AND_DEVICE static uint32_t computeRank(const RankingAccumulatorType rankingAccumulator,
                                                   const uint32_t possibleSolutionsCount) {
    return possibleSolutionsCount - rankingAccumulator;
  }

  template <uint8_t PIN_COUNT, uint8_t COLOR_COUNT>
  constexpr static uint32_t presetInitialGuess() {
    return presetInitialGuessKnuth<PIN_COUNT, COLOR_COUNT>();
  }
};

struct MostParts {
  using MaxSubsetSizeT = int8_t;
  constexpr static const char* name = "Most Parts";

  template <typename SubsetSizeT>
  CUDA_HOST_AND_DEVICE static void accumulateSubsetSize(SubsetSizeT& s) {
    s = 1;
  }

  using RankingAccumulatorType = uint32_t;

  template <typename SubsetSizeT>
  CUDA_HOST_AND_DEVICE static void accumulateRanking(RankingAccumulatorType& rankingAccumulator, SubsetSizeT& s,
                                                     uint32_t possibleSolutionsCount) {
    rankingAccumulator++;
  }

  CUDA_HOST_AND_DEVICE static uint32_t computeRank(RankingAccumulatorType rankingAccumulator,
                                                   uint32_t possibleSolutionsCount) {
    return rankingAccumulator;
  }

  template <uint8_t PIN_COUNT, uint8_t COLOR_COUNT>
  constexpr static uint32_t presetInitialGuess() {
    return presetInitialGuessMostParts<PIN_COUNT, COLOR_COUNT>();
  }
};

struct ExpectedSize {
  using MaxSubsetSizeT = int32_t;
  constexpr static const char* name = "Expected Size";

  template <typename SubsetSizeT>
  CUDA_HOST_AND_DEVICE static void accumulateSubsetSize(SubsetSizeT& s) {
    s++;
  }

  using RankingAccumulatorType = float;

  template <typename SubsetSizeT>
  CUDA_HOST_AND_DEVICE static void accumulateRanking(RankingAccumulatorType& rankingAccumulator, SubsetSizeT& s,
                                                     uint32_t possibleSolutionsCount) {
    rankingAccumulator += ((float)s * (float)s) / possibleSolutionsCount;
  }

  CUDA_HOST_AND_DEVICE static uint32_t computeRank(RankingAccumulatorType rankingAccumulator,
                                                   uint32_t possibleSolutionsCount) {
#pragma nv_diagnostic push
#pragma nv_diag_suppress 68
    // This is a bit broken, and needs to be to match the semantics in the paper.
    return (uint32_t)round(rankingAccumulator * 1'000'000.0) * -1;  // 9 digits of precision
#pragma nv_diagnostic pop
  }

  template <uint8_t PIN_COUNT, uint8_t COLOR_COUNT>
  constexpr static uint32_t presetInitialGuess() {
    return presetInitialGuessExpectedSize<PIN_COUNT, COLOR_COUNT>();
  }
};

struct Entropy {
  using MaxSubsetSizeT = int32_t;
  constexpr static const char* name = "Entropy";

  template <typename SubsetSizeT>
  CUDA_HOST_AND_DEVICE static void accumulateSubsetSize(SubsetSizeT& s) {
    s++;
  }

  using RankingAccumulatorType = float;

  template <typename SubsetSizeT>
  CUDA_HOST_AND_DEVICE static void accumulateRanking(RankingAccumulatorType& rankingAccumulator, SubsetSizeT& s,
                                                     uint32_t possibleSolutionsCount) {
    float pi = (float)s / possibleSolutionsCount;
    rankingAccumulator -= pi * log(pi);
  }

  CUDA_HOST_AND_DEVICE static uint32_t computeRank(RankingAccumulatorType rankingAccumulator,
                                                   uint32_t possibleSolutionsCount) {
    return round(rankingAccumulator * 1'000'000'000.0);  // 9 digits of precision
  }

  template <uint8_t PIN_COUNT, uint8_t COLOR_COUNT>
  constexpr static uint32_t presetInitialGuess() {
    return presetInitialGuessEntropy<PIN_COUNT, COLOR_COUNT>();
  }
};

}  // namespace Algos
