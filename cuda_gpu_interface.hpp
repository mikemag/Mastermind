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
class CUDAGPUInterface : public GPUInterface<typename SubsettingStrategyConfig::CodewordT> {

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

  void sendComputeCommand(const std::vector<typename SubsettingStrategyConfig::CodewordT>& possibleSolutions,
                          const std::vector<uint32_t>& usedCodewords) override;

  uint32_t getFullyDiscriminatingGuess() override { return littleStuff.fdGuess; }
  IndexAndScore getBestGuess() override {
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
    IndexAndScore bestGuess;
    uint32_t fdGuess;
    uint32_t usedCodewordsCount;
    uint32_t usedCodewords[100];
  };

 private:
  void dumpDeviceInfo();

  template <typename SubsettingAlgosKernelConfig>
  void launchSubsettingKernel(uint32_t possibleSolutionsCount);

  uint32_t* dAllCodewords{};
  unsigned __int128* dAllCodewordsColors{};
  Codeword<SubsettingStrategyConfig::PIN_COUNT, SubsettingStrategyConfig::COLOR_COUNT>* dPossibleSolutions{};
  IndexAndScore* dPerBlockSolutions{};
  LittleStuff* dLittleStuff{};

  LittleStuff littleStuff;

  unordered_map<string, string> gpuInfo;

  uint64_t totalSubsettingKernels = 0;
  uint64_t psSizesIn32Bits = 0;
  uint64_t psSizesIn16Bits = 0;
  uint64_t psSizesIn8Bits = 0;
};
