// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <string>

#include "compute_kernel_constants.h"
#include "gpu_interface.hpp"

template <uint8_t p, uint8_t c, Algo a, bool l>
class CUDAGPUInterface : public GPUInterface {
 public:
  CUDAGPUInterface();
  ~CUDAGPUInterface() override;

  bool gpuAvailable() const override;

  uint32_t* getAllCodewordsBuffer() override;
  unsigned __int128* getAllCodewordsColorsBuffer() override;
  void setAllCodewordsCount(uint32_t count) override;
  void syncAllCodewords(uint32_t count) override;

  uint32_t* getPossibleSolutionsBuffer() override;
  unsigned __int128* getPossibleSolutionsColorsBuffer() override;
  void setPossibleSolutionsCount(uint32_t count) override;

  uint32_t* getUsedCodewordsBuffer() override;
  void setUsedCodewordsCount(uint32_t count) override;

  void sendComputeCommand() override;

  uint32_t* getScores() override;
  bool* getRemainingIsPossibleSolution() override;

  uint32_t* getFullyDiscriminatingCodewords(uint32_t& count) override;
  uint32_t getFDGuess() override;
  IndexAndScore getBestGuess(uint32_t allCodewordsCount, std::vector<uint32_t>& usedCodewords,
                             uint32_t (*codewordGetter)(uint32_t)) override;

  std::string getGPUName() override;

 private:
  void dumpDeviceInfo();

  // Total scores = (p * (p + 3)) / 2, but +1 for imperfect packing.
  static constexpr int totalScores = ((p * (p + 3)) / 2) + 1;

  uint32_t* dAllCodewords;
  unsigned __int128* dAllCodewordsColors;
  uint32_t* dPossibleSolutions;
  unsigned __int128* dPossibleSolutionsColors;
  uint32_t possibleSolutionsCount;

  uint32_t* dUsedCodewords;
  uint32_t usedCodewordsCount;

  uint32_t* dFdGuess;
  IndexAndScore* dPerBlockSolutions;

  uint32_t threadsPerBlock;
  uint32_t numBlocks;
  size_t sharedMemSize = 0;

  string deviceName;
};

//#include "cuda_gpu_interface.cu"
