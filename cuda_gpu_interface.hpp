// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <string>

#include "compute_kernel_constants.h"
#include "gpu_interface.hpp"
#include "strategy_config.hpp"

template <typename SubsettingStrategyConfig>
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

  uint32_t* getPossibleSolutionsBuffer() override { return possibleSolutionsHost; }
  unsigned __int128* getPossibleSolutionsColorsBuffer() override { return possibleSolutionsColorsHost; }
  void setPossibleSolutionsCount(uint32_t count) override { possibleSolutionsCount = count; }

  uint32_t* getUsedCodewordsBuffer() override { return littleStuff.usedCodewords; }
  void setUsedCodewordsCount(uint32_t count) override { littleStuff.usedCodewordsCount = count; }

  void sendComputeCommand() override;

  uint32_t* getScores() override { return nullptr; }
  bool* getRemainingIsPossibleSolution() override { return nullptr; }

  uint32_t* getFullyDiscriminatingCodewords(uint32_t& count) override {
    count = 0;
    return nullptr;
  }
  uint32_t getFDGuess() override { return littleStuff.fdGuess; }
  IndexAndScore getBestGuess(uint32_t allCodewordsCount, std::vector<uint32_t>& usedCodewords,
                             uint32_t (*codewordGetter)(uint32_t)) override {
    return littleStuff.bestGuess;
  }

  std::unordered_map<std::string, std::string>& getGPUInfo() override {
    gpuInfo["GPU Subsetting Kernels Executed"] = to_string(totalSubsettingKernels);
    gpuInfo["GPU PS Size in 32bits"] = to_string(psSizesIn32Bits);
    gpuInfo["GPU PS Size in 16bits"] = to_string(psSizesIn16Bits);
    gpuInfo["GPU PS Size in 8bits"] = to_string(psSizesIn8Bits);
    return gpuInfo;
  }

  // A grab bag of small things we need to send back and forth to our kernels.
  struct LittleStuff {
    GPUInterface::IndexAndScore bestGuess;
    uint32_t fdGuess;
    uint32_t usedCodewordsCount;
    uint32_t usedCodewords[100];
  };

 private:
  void dumpDeviceInfo();

  template <typename SubsettingAlgosKernelConfig>
  void launchSubsettingKernel();

  uint32_t* dAllCodewords{};
  unsigned __int128* dAllCodewordsColors{};
  uint32_t* dPossibleSolutions{};
  unsigned __int128* dPossibleSolutionsColors{};
  uint32_t possibleSolutionsCount{};
  uint32_t* possibleSolutionsHost{};
  unsigned __int128* possibleSolutionsColorsHost{};

  IndexAndScore* dPerBlockSolutions{};

  LittleStuff* dLittleStuff{};
  LittleStuff littleStuff;

  unordered_map<string, string> gpuInfo;

  uint64_t totalSubsettingKernels = 0;
  uint64_t psSizesIn32Bits = 0;
  uint64_t psSizesIn16Bits = 0;
  uint64_t psSizesIn8Bits = 0;
};
