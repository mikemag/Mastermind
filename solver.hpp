// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <chrono>

#include "algos.hpp"
#include "codeword.hpp"
#include "counters.hpp"
#include "region.hpp"

struct SolverConfigBase {};

template <uint8_t PIN_COUNT_, uint8_t COLOR_COUNT_, bool LOG_, typename ALGO_>
struct SolverConfig : public SolverConfigBase {
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

class Solver {
 public:
  virtual std::chrono::nanoseconds playAllGames(uint32_t packedInitialGuess) = 0;

  uint getMaxDepth() const { return maxDepth; }
  unsigned long long int getTotalTurns() const { return totalTurns; }
  virtual bool usesGPU() const { return false; }

  // Output the strategy for visualization with GraphViz. Copy-and-paste the output file to sites
  // like https://dreampuf.github.io/GraphvizOnline or http://www.webgraphviz.com/. Or install
  // GraphViz locally and run with the following command:
  //
  //   twopi -Tjpg mastermind_strategy_4p6c.gv > mastermind_strategy_4p6c.jpg
  //
  // Parameters for the graph are currently set to convey the point while being reasonably readable
  // in a large JPG.
  virtual void dump() = 0;
  virtual vector<uint32_t> getGuessesForGame(uint32_t packedCodeword) = 0;

  virtual void printStats() = 0;
  virtual void recordStats(StatsRecorder &sr) = 0;

 protected:
  uint maxDepth = 0;
  unsigned long long int totalTurns = 0;

  template <typename SolverConfig, typename CodewordT, typename RegionID>
  void dump(vector<RegionID> &regionIDs);

  template <typename Solver, typename SolverConfig, typename CodewordT, typename RegionID>
  vector<uint32_t> getGuessesForGame(uint32_t packedCodeword, vector<RegionID> &regionIDs);

  virtual uint32_t getPackedCodewordForRegion(int level, uint32_t regionIndex) const = 0;
  virtual uint8_t getStandardScore(uint8_t score) = 0;
};

#include "solver.inl"