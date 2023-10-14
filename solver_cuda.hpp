// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <vector>

#include "solver.hpp"

// CUDA implementation for playing all games at once
//
// This plays all games at once, playing a turn on every game and forming a set of next guesses for the next turn.
// Scoring all games subdivides those games with the same score into disjoint regions. We form an id for each region
// made up of a growing list of scores, and sorting the list of games by region id groups the regions together. Then we
// can find a single best guess for each region and play that for all remaining games in the region.
//
// Pseudocode for the algorithm:
//   start: all games get the same initial guess
//
//   while any games have guesses to play:
//     score all games against their next guess, if any, which was given per-region
//       append their score to their region id
//       if no games have guesses, then we're done
//     stable sort all games by region id
//     get start and run length for each region by id
//     for reach region:
//       games with a win at the end of their region id get no new guess
//       otherwise, find next guess using the region itself as the possible solutions set PS
//
// This algorithm applies nicely to the GPU. We can score in parallel, use parallel partitions, reductions, sorts, and
// transformations to form the regions and get them ready for the core kernels. And the core n^2 kernels can run in
// parallel as GPU resources allow. All gameplay state resides in GPU memory, with little or no state returning to the
// CPU.
//
// See the impl of playAllGames() for more details.

template <typename SolverConfig_>
class SolverCUDA : public Solver {
  using CodewordT = typename SolverConfig_::CodewordT;
  using RegionID = RegionID<unsigned __int128, SolverConfig_::TOTAL_PACKED_SCORES - 1>;

 public:
  using SolverConfig = SolverConfig_;
  constexpr static const char* name = "CUDA";

  SolverCUDA() : counters(counterDescs.descs.size()) {}

  std::chrono::nanoseconds playAllGames(uint32_t packedInitialGuess) override;

  bool usesGPU() const override { return true; }

  void dump() override;
  vector<uint32_t> getGuessesForGame(uint32_t packedCodeword) override;

  void printStats() override {
    for (auto& c : counterDescs.descs) {
      cout << c.desc << ": " << commaString(counters[c.index]) << endl;
    }
  }
  void recordStats(StatsRecorder& sr) override {
    sr.add("Use Sym Opt", applySymOpt);
    for (auto& c : counterDescs.descs) {
      sr.add(c.name, counters[c.index]);
    }
  }

  constexpr static CounterDescriptors<8> counterDescs{{
      {"Scores", "Codeword comparisons"},
      {"Tiny Regions", "Total tiny regions"},
      {"Tiny Games", "Total tiny games"},
      {"FDOpt Regions", "Total FD Opt regions"},
      {"FDOpt Games", "Total FD Opt games"},
      {"Big Regions", "Total big regions"},
      {"ACr Count", "Total ACr generated"},
      {"ACr Size", "Total ACr size"},
  }};

 private:
  vector<RegionID> regionIDs;
  vector<vector<uint32_t>> nextMovesList;
  vector<uint8_t> packedToStandardScore;
  vector<unsigned long long int> counters;

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

  // Tuned on a RTX 4090. A little brute force, would be interesting to see if there's something more nuanced some day.
  constexpr static bool applySymOpt =
      SolverConfig::SYMOPT && (constPow((int)SolverConfig::COLOR_COUNT, (int)SolverConfig::PIN_COUNT)) > 400000;

 public:
  // The only reason these are public is due to limitations w/ CUDA __device__ functions
  struct ZFColors {
    uint32_t zero;
    uint32_t free;

    CUDA_HOST_AND_DEVICE bool operator<(const ZFColors& other) const {
      if (zero == other.zero) {
        return free < other.free;
      }
      return zero < other.zero;
    }

    CUDA_HOST_AND_DEVICE bool operator==(const ZFColors& other) const {
      return zero == other.zero && free == other.free;
    }
  };

  void buildZerosAndFrees(const CodewordT* pdAllCodewords, thrust::device_vector<RegionID>& dRegionIDs,
                          thrust::device_vector<RegionID>::iterator& dRegionIDsEnd, uint32_t regionCount,
                          thrust::device_vector<uint32_t>& dRegionStarts, uint32_t** pdNextMovesVecs,
                          uint32_t nextMovesVecsSize, thrust::device_vector<ZFColors>& dZFColors);

  uint32_t buildSomeACr(uint32_t start, thrust::host_vector<ZFColors>& dZFColors,
                        thrust::device_vector<CodewordT>& dAllCodewords, uint32_t regionCount,
                        thrust::device_vector<uint32_t>& dACrBuffer, thrust::device_vector<uint32_t>& dACrStarts,
                        thrust::device_vector<uint32_t>& dACrLengths);
};

#include "solver_cuda.inl"
