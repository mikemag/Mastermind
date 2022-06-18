// Copyright (c) Michael M. Magruder (https://github.com/mikemag)
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <string>

#include "gpu_interface.hpp"

class NoGPUInterface : public GPUInterface {
 public:
  NoGPUInterface() = default;

  bool gpuAvailable() const override { return false; }

  uint32_t* getAllCodewordsBuffer() override { return nullptr; }
  unsigned __int128* getAllCodewordsColorsBuffer() override { return nullptr; }
  void setAllCodewordsCount(uint32_t count) override {}
  void syncAllCodewords(uint32_t count) override {}

  uint32_t* getPossibleSolutionsBuffer() override { return nullptr; }
  unsigned __int128* getPossibleSolutionsColorsBuffer() override { return nullptr; }
  void setPossibleSolutionsCount(uint32_t count) override {}

  uint32_t* getUsedCodewordsBuffer() override { return nullptr; }
  void setUsedCodewordsCount(uint32_t count) override {}

  void sendComputeCommand() override {}

  uint32_t* getScores() override { return nullptr; }
  bool* getRemainingIsPossibleSolution() override { return nullptr; }

  uint32_t* getFullyDiscriminatingCodewords(uint32_t& count) override { return nullptr; }
  uint32_t getFDGuess() override { return UINT32_MAX; }
  IndexAndScore getBestGuess(uint32_t allCodewordsCount, std::vector<uint32_t>& usedCodewords,
                             uint32_t (*codewordGetter)(uint32_t)) override {
    return {UINT32_MAX, 0, false};
  }

  std::string getGPUName() override { return "None"; }
};
