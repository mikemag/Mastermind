// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <string>

#include "compute_kernel_constants.h"
#include "gpu_interface.hpp"

template <uint8_t p, uint8_t c, Algo a, typename SubsetSize, bool l>
class CUDAGPUInterface : public GPUInterface {
 public:
  CUDAGPUInterface();
  ~CUDAGPUInterface() override;

  bool gpuAvailable() const override { return true; }

  uint32_t* getAllCodewordsBuffer() override { return dAllCodewords; }
  unsigned __int128* getAllCodewordsColorsBuffer() override { return dAllCodewordsColors; }

  void setAllCodewordsCount(uint32_t count) override {
    // TODO: this is redundant for this impl, and likely for the Metal impl too. Need to fix this up.
  }
  void syncAllCodewords(uint32_t count) override {
    // TODO: let it page fault for now, come back and add movement hints if necessary.
  }

  uint32_t* getPossibleSolutionsBuffer() override { return dPossibleSolutionsHost; }
  unsigned __int128* getPossibleSolutionsColorsBuffer() override { return dPossibleSolutionsColorsHost; }
  void setPossibleSolutionsCount(uint32_t count) override { possibleSolutionsCount = count; }

  uint32_t* getUsedCodewordsBuffer() override { return dUsedCodewords; }
  void setUsedCodewordsCount(uint32_t count) override { usedCodewordsCount = count; }

  void sendComputeCommand() override;

  uint32_t* getScores() override { return nullptr; }
  bool* getRemainingIsPossibleSolution() override { return nullptr; }

  uint32_t* getFullyDiscriminatingCodewords(uint32_t& count) override {
    count = 0;
    return nullptr;
  }
  uint32_t getFDGuess() override { return *dFdGuess; }
  IndexAndScore getBestGuess(uint32_t allCodewordsCount, std::vector<uint32_t>& usedCodewords,
                             uint32_t (*codewordGetter)(uint32_t)) override {
    return dPerBlockSolutions[0];
  }

  std::string getGPUName() override { return deviceName; }

 private:
  void dumpDeviceInfo();

  // Total scores = (p * (p + 3)) / 2, but +1 for imperfect packing.
  static constexpr int totalScores = ((p * (p + 3)) / 2) + 1;

  uint32_t* dAllCodewords;
  unsigned __int128* dAllCodewordsColors;
  uint32_t* dPossibleSolutions;
  unsigned __int128* dPossibleSolutionsColors;
  uint32_t possibleSolutionsCount;
  uint32_t* dPossibleSolutionsHost;
  unsigned __int128* dPossibleSolutionsColorsHost;

  uint32_t* dUsedCodewords;
  uint32_t usedCodewordsCount;

  uint32_t* dFdGuess;
  IndexAndScore* dPerBlockSolutions;

  uint32_t threadsPerBlock;
  uint32_t numBlocks;
  size_t sharedMemSize = 0;
  uint32_t roundedTotalCodewords;

  string deviceName;
};

//#include "cuda_gpu_interface.cu"
