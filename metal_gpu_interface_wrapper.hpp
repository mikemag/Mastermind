// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "gpu_interface.hpp"

#ifdef __OBJC__
#define OBJC_CLASS(name) @class name
#else
#define OBJC_CLASS(name) typedef struct objc_object name
#endif

OBJC_CLASS(MetalGPUInterface);

// TODO: the Metal impl is completely broken now. I didn't want to keep hacking up the current work advancing w/ CUDA
// just to keep this working. I need to get back to this at some point and fix it all up, implement the new things in
// the Metal kernels, etc.

class MetalGPUInterfaceWrapper : public GPUInterface {
  MetalGPUInterface* wrapped;
  std::unordered_map<std::string, std::string> gpuInfo;

 public:
  MetalGPUInterfaceWrapper(unsigned int pinCount, unsigned int totalCodewords, const char* kernelName);
  ~MetalGPUInterfaceWrapper() override;

  bool gpuAvailable() const override { return wrapped != nullptr; }

  uint32_t* getAllCodewordsBuffer() override;
  unsigned __int128* getAllCodewordsColorsBuffer() override;
  void setAllCodewordsCount(uint32_t count) override;
  void syncAllCodewords(uint32_t count) override;

  uint32_t* getPossibleSolutionsBuffer() override;
  unsigned __int128* getPossibleSolutionsColorsBuffer() override;
  void setPossibleSolutionsCount(uint32_t count) override;

  uint32_t* getUsedCodewordsBuffer() override { return nullptr; }
  void setUsedCodewordsCount(uint32_t count) override {}

  void sendComputeCommand() override;

  uint32_t* getScores() override;
  bool* getRemainingIsPossibleSolution() override;

  uint32_t* getFullyDiscriminatingCodewords(uint32_t& count) override;

  uint32_t GetFullyDiscriminatingGuess() override {
    // TODO: impl for Metal
    uint32_t discriminatingCount = 0;
    uint32_t* smallOptsOut = getFullyDiscriminatingCodewords(discriminatingCount);
    for (int i = 0; i < discriminatingCount; i++) {
      if (smallOptsOut[i] > 0) {
        return smallOptsOut[i];
      }
    }
    return UINT32_MAX;
  }

  IndexAndScore getBestGuess(uint32_t allCodewordsCount, std::vector<uint32_t>& usedCodewords,
                             uint32_t (*codewordGetter)(uint32_t)) override {
    // TODO: impl for Metal
    uint32_t* maxScoreCounts = getScores();
    bool* remainingIsPossibleSolution = getRemainingIsPossibleSolution();
    bool bestIsPossibleSolution = false;
    uint32_t bestScore = 0;
    uint32_t bestGuessIndex = 0;

    for (int i = 0; i < allCodewordsCount; i++) {
      uint32_t score = maxScoreCounts[i];
      if (score > bestScore || (!bestIsPossibleSolution && remainingIsPossibleSolution[i] && score == bestScore)) {
        uint32_t codeword = codewordGetter(i);
        if (find(usedCodewords.cbegin(), usedCodewords.cend(), codeword) != usedCodewords.end()) {
          continue;  // Ignore codewords we've already used
        }
        bestScore = score;
        bestGuessIndex = i;
        bestIsPossibleSolution = remainingIsPossibleSolution[i];
      }
    }
    return {bestGuessIndex, bestScore, bestIsPossibleSolution};
  }

  std::unordered_map<std::string, std::string> &getGPUInfo() override;
};
