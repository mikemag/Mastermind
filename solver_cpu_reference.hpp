// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cassert>
#include <vector>

#include "solver.hpp"

// CPU Reference Implementation
//
// This is a simple impl to serve as a reference for all the others. It's not optimized for speed. Hopefully it's clear.
// More details w/ the impl.

template <typename SolverConfig_>
class SolverReferenceImpl : public Solver {
  using CodewordT = typename SolverConfig_::CodewordT;
  using RegionID = RegionID<unsigned __int128, SolverConfig_::CodewordT::WINNING_SCORE.result>;

 public:
  using SolverConfig = SolverConfig_;
  constexpr static const char* name = "CPU Reference Impl";

  std::chrono::nanoseconds playAllGames(uint32_t packedInitialGuess) override;

  void dump() override;

 private:
  CodewordT nextGuess(const vector<CodewordT>& possibleSolutions, const vector<CodewordT>& usedCodewords);

  uint32_t getPackedCodewordForRegion(int level, uint32_t regionIndex) const override {
    return nextMovesList[level][regionIndex].packedCodeword();
  }
  uint8_t getStandardScore(uint8_t score) override { return score; }

  vector<vector<CodewordT>> nextMovesList;
  vector<RegionID> regionIDs;
};

#include "solver_cpu_reference.inl"
