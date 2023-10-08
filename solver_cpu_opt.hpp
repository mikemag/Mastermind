// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cassert>
#include <vector>

#include "solver.hpp"

// CPU Implementation w/ Some Optimizations
//
// This is a version that runs on the CPU and is structured very close to the CUDA version. It's much faster than the
// reference impl, and includes the most common gameplay shortcuts.
//
// For instance, on one machine for 5p8c Knuth the ref impl takes 63.5160s, and this one takes 8.6398s.

template <typename SolverConfig_>
class SolverCPUFaster : public Solver {
  using CodewordT = typename SolverConfig_::CodewordT;
  using RegionIDT = RegionID<unsigned __int128, SolverConfig_::CodewordT::WINNING_SCORE.result>;

 public:
  using SolverConfig = SolverConfig_;
  constexpr static const char* name = "CPU Faster";

  SolverCPUFaster() : counters(counterDescs.descs.size()) {}
  std::chrono::nanoseconds playAllGames(uint32_t packedInitialGuess) override;
  void dump() override;
  vector<uint32_t> getGuessesForGame(uint32_t packedCodeword) override;

  void printStats() override {
    for (auto& c : counterDescs.descs) {
      cout << c.desc << ": " << commaString(counters[c.index]) << endl;
    }
  }

  void recordStats(StatsRecorder& sr) override {
    for (auto& c : counterDescs.descs) {
      sr.add(c.name, counters[c.index]);
    }
  }

  constexpr static CounterDescriptors<1> counterDescs{{
      {"Scores", "Codeword comparisons"},
  }};

 private:
  struct ACrCacheKey {
    uint32_t isZero;
    uint32_t isFree;

    bool operator==(const ACrCacheKey& other) const { return isZero == other.isZero && isFree == other.isFree; }
  };

  struct ACrCacheKeyHash {
    size_t operator()(const ACrCacheKey& key) const {
      return std::hash<uint32_t>{}(key.isZero) ^ std::hash<uint32_t>{}(key.isFree);
    }
  };

  std::unordered_map<ACrCacheKey, vector<CodewordT>, ACrCacheKeyHash> symACCache;

  vector<vector<CodewordT>> nextMovesList;
  vector<RegionIDT> regionIDs;
  vector<unsigned long long int> counters;

  CodewordT nextGuess(const vector<CodewordT>& allCodewords, const vector<CodewordT>& possibleSolutions,
                      const vector<CodewordT>& usedCodewords);
  bool shortcutSmallSets(const vector<CodewordT>& possibleSolutions, CodewordT& nextGuess);
  vector<typename SolverConfig::CodewordT> getReducedAC(const vector<CodewordT>& allCodewords,
                                                        const vector<CodewordT>& possibleSolutions,
                                                        const vector<CodewordT>& usedCodewords, uint depth);

  uint32_t getPackedCodewordForRegion(int level, uint32_t regionIndex) const override {
    return nextMovesList[level][regionIndex].packedCodeword();
  }
  uint8_t getStandardScore(uint8_t score) override { return score; }
};

#include "solver_cpu_opt.inl"
