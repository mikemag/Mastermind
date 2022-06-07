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

  void sendComputeCommand() override;

  uint32_t* getScores() override;
  bool* getRemainingIsPossibleSolution() override;

  uint32_t* getFullyDiscriminatingCodewords(uint32_t& count) override;

  std::string getGPUName() override;

 private:
  uint32_t* dAllCodewords;
  unsigned __int128* dAllCodewordsColors;
  uint32_t* dPossibleSolutions;
  unsigned __int128* dPossibleSolutionsColors;
  uint32_t possibleSolutionsCount;
  uint32_t* dScores;
  bool* dRemainingIsPossibleSolution;

  uint32_t fdCount;
  uint32_t* dFullyDiscriminatingCodewords;

  uint32_t threadsPerBlock;
  uint32_t numBlocks;
  size_t sharedMemSize;
};

//#include "cuda_gpu_interface.cu"
