// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <vector>

#include "solver.hpp"

// CUDA implementation for playing all games at once
//
// TODO: this needs a lot of notes and docs consolidated

template <typename SolverConfig_>
class SolverCUDA : public Solver {
  using CodewordT = typename SolverConfig_::CodewordT;
  using RegionID = RegionID<unsigned __int128, SolverConfig_::TOTAL_PACKED_SCORES - 1>;

 public:
  using SolverConfig = SolverConfig_;
  constexpr static const char* name = "CUDA";

  std::chrono::nanoseconds playAllGames(uint32_t packedInitialGuess) override;

  void dump() override;

 private:
  uint32_t getPackedCodewordForRegion(int level, uint32_t regionIndex) const override {
    return CodewordT::getAllCodewords()[nextMovesList[level][regionIndex]].packedCodeword();
  }

  // Convert a packed GPU score back to a "standard" black & white pins score
  uint8_t getStandardScore(uint8_t score) override {
    if (packedToStandardScore.empty()) {
      for (int b = 0; b <= SolverConfig::PIN_COUNT; b++) {
        for (int w = 0; w <= SolverConfig::PIN_COUNT - b; w++) {
          packedToStandardScore.push_back(static_cast<uint8_t>((b << 4) | w));
        }
      }
    }
    return packedToStandardScore[score];
  }

  vector<RegionID> regionIDs;
  vector<vector<uint32_t>> nextMovesList;
  vector<uint8_t> packedToStandardScore;
};

#include "solver_cuda.inl"
