// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "algos.hpp"
#include "codeword.hpp"

template <uint8_t PIN_COUNT_, uint8_t COLOR_COUNT_, bool LOG_, typename ALGO_>
struct SolverConfig {
  constexpr static uint8_t PIN_COUNT = PIN_COUNT_;
  constexpr static uint8_t COLOR_COUNT = COLOR_COUNT_;
  constexpr static bool LOG = LOG_;
  using ALGO = ALGO_;

  using CodewordT = Codeword<PIN_COUNT, COLOR_COUNT>;
  using SubsetSizeT = typename ALGO::MaxSubsetSizeT;

  constexpr static int TOTAL_SCORES = (PIN_COUNT * (PIN_COUNT + 3)) / 2;
  constexpr static uint32_t MAX_SCORE_SLOTS = (PIN_COUNT << 4u) + 1;
  constexpr static CodewordT INITIAL_GUESS = ALGO::template presetInitialGuess<PIN_COUNT, COLOR_COUNT>();

  // Packed scores have room for the extra impossible score, so +1 for imperfect packing
  constexpr static int TOTAL_PACKED_SCORES = TOTAL_SCORES + 1;
};

template <typename SolverConfig>
class Solver {
  using CodewordT = typename SolverConfig::CodewordT;

 public:
  virtual void playAllGames() = 0;

  size_t getMaxDepth() const { return maxDepth; }
  size_t getTotalTurns() const { return totalTurns; };
  size_t getTotalGames() const { return SolverConfig::CodewordT::TOTAL_CODEWORDS; }

  virtual std::string getName() const { return "tmp name"; }

 protected:
  size_t maxDepth = 0;
  size_t totalTurns = 0;

  // - printstats, recordstats, dump
  // - flags and opt config
};

#include "solver.inl"